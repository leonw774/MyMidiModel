import bisect
import os
import sys
from typing import List, Tuple, Union
from time import time
import zipfile

import numpy as np
import psutil
from torch import from_numpy, Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from util.corpus import get_corpus_vocabs, TOKEN_ATTR_INDEX
from util.tokens import BEGIN_TOKEN_STR, END_TOKEN_STR

class MidiDataset(Dataset):

    def __init__(
            self,
            data_dir_path: str,
            max_seq_length: int,
            use_permutable_subseq_loss: bool,
            permute_mps: bool = False,
            permute_track_number: bool = False,
            pitch_augmentation: int = 0,
            sample_from_start: bool = False
        ) -> None:
        """
            Parameters:
            - data_dir_path: Expected to have 'data.npz' and 'vocabs.json'
            - use_permutable_subseq_loss: Whether or not we provide a mps_seperator_indices information at __getitem__
            - permute_mps: Whether or not the dataset should permute all the *maximal permutable subsequences*
              in the sequence before returning in `__getitem__`
            - permute_track_number: Permute all the track numbers, as data augmentation
        """
        npz_path = os.path.join(data_dir_path, 'arrays.npz')
        print('Reading', npz_path)
        available_memory_size = psutil.virtual_memory().available
        npz_zipinfo_list = zipfile.ZipFile(npz_path).infolist()
        array_memory_size = sum([zinfo.file_size for zinfo in npz_zipinfo_list])
        print('available_memory_size:', available_memory_size, 'array_memory_size:', array_memory_size)
        if array_memory_size >= available_memory_size - 4e9: # keep 4G for other things
            # load from disk every time indexing
            print('Memory size not enough, using NPZ mmap.')
            self.pieces = np.load(npz_path)
        else:
            # load into memory to be faster
            print('Loading arrays to memory')
            npz_file = np.load(npz_path)
            self.pieces = {
                str(filenum): npz_file[str(filenum)]
                for filenum in tqdm(range(len(npz_file)))
            }
        print('Processing')

        self.vocabs = get_corpus_vocabs(data_dir_path)
        self.max_seq_length = max_seq_length
        self.use_permutable_subseq_loss = use_permutable_subseq_loss
        self.permute_mps = permute_mps
        self.permute_track_number = permute_track_number
        self.pitch_augmentation = pitch_augmentation
        self.number_of_pieces = len(self.pieces)

        # Seperators are:
        # EOS, BOS, PADDING, first track token (R0), measure tokens (M\w+), position tokens (P\w+)
        # stores their index in event vocab, empty when `use_permutable_subseq_loss` is not True
        self._mps_seperators = list()
        self._note_ids = list()
        self._position_ids = list()
        self._bos_id = self.vocabs.events.text2id[BEGIN_TOKEN_STR]
        self._eos_id = self.vocabs.events.text2id[END_TOKEN_STR]
        # the pad id should be the same across all vocabs (aka zero)
        self._pad_id = self.vocabs.events.text2id[self.vocabs.padding_token]
        self._tempo_ids = set()
        self._measure_ids = set()
        self._track_ids = set()
        for text, index in self.vocabs.events.text2id.items():
            if text[0] == 'N' or text[0] == 'S':
                self._note_ids.append(index)
            elif text[0] == 'T':
                self._tempo_ids.add(index)
            elif text[0] == 'P':
                self._position_ids.append(index)
            elif text[0] == 'M':
                self._measure_ids.add(index)
            elif text[0] == 'R':
                self._track_ids.add(index)
        if use_permutable_subseq_loss or permute_mps:
            self._mps_seperators = set([
                self._bos_id,
                self._eos_id,
                self.vocabs.events.text2id['R0'],
                self._pad_id,
            ])
            self._mps_seperators.update(self._track_ids)
            self._mps_seperators.update(self._measure_ids)
            self._mps_seperators.update(self._position_ids)
            self._mps_seperators = np.sort(np.array(list(self._mps_seperators)))
        # numpy.isin can work on set, turn it into sorted 1d array helps search speed
        self._note_ids = np.sort(np.array(self._note_ids))
        self._position_ids = np.sort(np.array(self._position_ids))

        # preprocessing
        self._sample_from_start = sample_from_start
        self._filenum_indices = [0] * self.number_of_pieces
        self._piece_lengths = [0] * self.number_of_pieces
        self._piece_body_start_index = [0] * self.number_of_pieces
        self._file_mps_sep_indices = [[] for _ in range(self.number_of_pieces)]
        cur_index = 0
        for filenum in tqdm(range(self.number_of_pieces)):
            filename = str(filenum)

            if use_permutable_subseq_loss or permute_mps:
                # find all seperater's index
                mps_sep_indices = np.flatnonzero(np.isin(self.pieces[filename][:, 0], self._mps_seperators))
                for i in range(mps_sep_indices.shape[0] - 1):
                    self._file_mps_sep_indices[filenum].append(mps_sep_indices[i])
                    if mps_sep_indices[i+1] != mps_sep_indices[i] + 1:
                        self._file_mps_sep_indices[filenum].append(mps_sep_indices[i] + 1)

            cur_piece_lengths = self.pieces[filename].shape[0]
            self._piece_lengths[filenum] = cur_piece_lengths

            if cur_piece_lengths > max_seq_length and not sample_from_start:
                cur_index += cur_piece_lengths - max_seq_length + 1 * (self.pitch_augmentation * 2 + 1)
            else:
                cur_index += 1 * (self.pitch_augmentation * 2 + 1)
            self._filenum_indices[filenum] = cur_index
        self._max_index = cur_index
        # cast self._filenum_indices from list of tuples into np array to save space
        self_size = sum(
            sys.getsizeof(getattr(self, attr_name))
            for attr_name in dir(self)
            if not attr_name.startswith('__')
        )
        print('Dataset object size:', self_size, 'bytes')

    def __getitem__(self, index):
        # while True:
        #     if index < 0:
        #         index = self._max_index + index
        #     else:
        #         break
        filenum = bisect.bisect_left(self._filenum_indices, index)
        begin_index_augmented = index if filenum == 0 else index - self._filenum_indices[filenum-1] - 1
        begin_index, pitch_augment = divmod(begin_index_augmented, (self.pitch_augmentation * 2 + 1))
        pitch_augment -= self.pitch_augmentation
        if self._piece_lengths[filenum] <= self.max_seq_length:
            end_index = self._piece_lengths[filenum]
        else:
            end_index = begin_index + self.max_seq_length
        sliced_array = np.array(self.pieces[str(filenum)][begin_index:end_index]) # copy
        sliced_array = sliced_array.astype(np.int32) # was int16 to save space but torch ask for int

        # pitch augmentation
        # assume pitch vocabulary is 0:PAD, 1:0, 2:1, ... 128:127
        if pitch_augment != 0:
            valid_pitches = (sliced_array[:, TOKEN_ATTR_INDEX['pit']]) != 0
            min_pitch = np.min( (sliced_array[:, TOKEN_ATTR_INDEX['pit']])[valid_pitches] ) + pitch_augment
            max_pitch = np.max( (sliced_array[:, TOKEN_ATTR_INDEX['pit']])[valid_pitches] ) + pitch_augment
            if min_pitch >= 1 and max_pitch <= 128:
                (sliced_array[:, TOKEN_ATTR_INDEX['pit']])[valid_pitches] += pitch_augment

        # shift down measure number so that it didn't exceed max_seq_length
        min_measure_num = sliced_array[0, TOKEN_ATTR_INDEX['mea']]
        if min_measure_num > 1:
            sliced_array[:, TOKEN_ATTR_INDEX['mea']] -= (min_measure_num - 1)
        # because EOS also has measure_number = 0
        if sliced_array[-1, TOKEN_ATTR_INDEX['mea']] < 0:
            # set EOS measure number back to zero
            sliced_array[-1, TOKEN_ATTR_INDEX['mea']] = 0

        if self.permute_track_number:
            # amortize the calculation
            if self._piece_body_start_index[filenum] == 0:
                j = 2 # because first token is BOS and second must be track as we excluded all empty midis
                while self.pieces[str(filenum)][j][0] in self._track_ids:
                    j += 1
                self._piece_body_start_index[filenum] = j

            # use self._piece_body_start_index to know how many tracks there are in this piece
            track_count = self._piece_body_start_index[filenum] - 1
            slice_body_begin = self._piece_body_start_index[filenum] - begin_index
            slice_body_begin = max(0, slice_body_begin)
            # add one because there is a padding token at the beginning of the vocab
            perm_array = np.random.permutation(track_count) + 1
            # view body's track number col
            body_trn_column = sliced_array[slice_body_begin:, TOKEN_ATTR_INDEX['trn']]
            body_trn_column_expand = np.asarray([
                (body_trn_column == i + 1) for i in range(track_count)
            ])
            # permute body's track number
            ins_col_index = TOKEN_ATTR_INDEX['ins']
            for i in range(track_count):
                body_trn_column[body_trn_column_expand[i]] = perm_array[i]
            # permute head's track instruments with the inverse of the permutation array
            if slice_body_begin > 0:
                slice_track_begin = 1 if begin_index == 0 else 0
                inv_perm_array = np.empty(track_count, dtype=np.int16)
                inv_perm_array[perm_array-1] = (np.arange(track_count) + 1)
                track_permuted_ins = np.empty(track_count, dtype=np.int16)
                track_permuted_ins = self.pieces[str(filenum)][inv_perm_array, ins_col_index]
                sliced_array[slice_track_begin:slice_body_begin, ins_col_index] = track_permuted_ins[max(0, begin_index-1):]

        mps_sep_indices_list = []
        if self.permute_mps:
            mps_tokens_ranges = []
            mps_sep_indices_list = [
                i - begin_index
                for i in self._file_mps_sep_indices[filenum]
                if begin_index <= i < end_index
            ]
            mps_sep_indices_and_two_ends = [0] + mps_sep_indices_list + [sliced_array.shape[0]]
            for i in range(len(mps_sep_indices_and_two_ends)-1):
                start = mps_sep_indices_and_two_ends[i]
                end = mps_sep_indices_and_two_ends[i+1]
                if end - start >= 2:
                    mps_tokens_ranges.append((start, end))

            # print(mps_tokens_ranges)
            for start, end in mps_tokens_ranges:
                p = np.random.permutation(end-start) + start
                sliced_array[start:end] = sliced_array[p]

        # a numpy array of sequences sep indices would also be returned if self.use_permutable_subseq_loss
        if self.use_permutable_subseq_loss:
            if len(mps_sep_indices_list) == 0:
                mps_sep_indices_list = [
                    i - begin_index
                    for i in self._file_mps_sep_indices[filenum]
                    if begin_index <= i < end_index
                ]
            mps_sep_indices = np.array(
                mps_sep_indices_list,
                dtype=np.int16
            )
            return from_numpy(sliced_array), mps_sep_indices

        return from_numpy(sliced_array), None


    def __len__(self):
        return self._max_index


def collate_mididataset(data: List[Tuple[Tensor, Union[np.ndarray, None]]]) -> Tuple[Tensor, List]:
    """
        for batched sequences: stack them on batch-first to becom 3-d array and pad them
        for others, become 1-d object numpy array if not None
    """
    # print('\n'.join(map(str, [d for d in data])))
    seqs, mps_sep_indices = zip(*data)
    batched_seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    # mps_sep_indices could be None or numpy object array of tuples
    if mps_sep_indices[0] is None:
        batched_mps_sep_indices = []
    else:
        batched_mps_sep_indices = list(mps_sep_indices)
    return batched_seqs, batched_mps_sep_indices
