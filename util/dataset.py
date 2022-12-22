import bisect
import os
import sys
from typing import List, Tuple, Union
import zipfile

import numpy as np
import psutil
from torch import from_numpy, Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from . import tokens
from .corpus import get_corpus_vocabs, TOKEN_ATTR_INDEX

class MidiDataset(Dataset):

    def __init__(
            self,
            data_dir_path: str,
            max_seq_length: int,
            use_permutable_subseq_loss: bool,
            permute_mps: bool = False,
            permute_track_number: bool = False,
            pitch_augmentation: int = 0,
            verbose: bool = False
        ) -> None:
        """
            Parameters:
            - data_dir_path: Expected to have 'data.npz' and 'vocabs.json'
            - use_permutable_subseq_loss: Whether or not we provide a mps_seperator_indices information at __getitem__
            - permute_mps: Whether or not the dataset should permute all the *maximal permutable subsequences*
              in the sequence before returning in `__getitem__`
            - permute_track_number: Permute all the track numbers, as data augmentation
        """

        self.vocabs = get_corpus_vocabs(data_dir_path)
        assert max_seq_length >= 0
        self.max_seq_length = max_seq_length
        assert isinstance(use_permutable_subseq_loss, bool)
        self.use_permutable_subseq_loss = use_permutable_subseq_loss
        assert isinstance(permute_mps, bool)
        self.permute_mps = permute_mps
        assert isinstance(permute_track_number, bool)
        self.permute_track_number = permute_track_number
        assert pitch_augmentation >= 0
        self.pitch_augmentation = pitch_augmentation

        npz_path = os.path.join(data_dir_path, 'arrays.npz')
        if verbose:
            print('Reading', npz_path)
        available_memory_size = psutil.virtual_memory().available
        npz_zipinfo_list = zipfile.ZipFile(npz_path).infolist()
        array_memory_size = sum([zinfo.file_size for zinfo in npz_zipinfo_list])
        if verbose:
            print('available_memory_size:', available_memory_size, 'array_memory_size:', array_memory_size)
        if array_memory_size >= available_memory_size - 4e9: # keep 4G for other things
            # load from disk every time indexing
            if verbose:
                print('Memory size not enough, using NPZ mmap.')
            self.pieces = np.load(npz_path)
        else:
            # load into memory to be faster
            if verbose:
                print('Loading arrays to memory')
            npz_file = np.load(npz_path)
            self.pieces = {
                str(filenum): npz_file[str(filenum)]
                for filenum in tqdm(range(len(npz_file)), disable=not verbose)
            }
        self.number_of_pieces = len(self.pieces)

        if verbose:
            print('Processing')
        # Seperators are:
        # BOS, EOS, SEP, PADDING, first track-instrument token, measure tokens, position tokens and tempo tokens
        # stores their index in event vocab
        # _mps_seperators is empty when `use_permutable_subseq_loss` is not True
        self._mps_seperators = list()
        self._bos_id = self.vocabs.events.text2id[tokens.BEGIN_TOKEN_STR]
        self._sep_id = self.vocabs.events.text2id[tokens.SEP_TOKEN_STR]
        self._eos_id = self.vocabs.events.text2id[tokens.END_TOKEN_STR]
        # the pad id should be the same across all vocabs (aka zero)
        self._pad_id = self.vocabs.events.text2id[self.vocabs.padding_token]
        self._track_ids = set()
        self._measure_ids = set()
        self._tempo_ids = set()
        self._position_ids = list()
        self._note_ids = list()
        for text, index in self.vocabs.events.text2id.items():
            if text[0] == tokens.NOTE_EVENTS_CHAR or text[0] == tokens.MULTI_NOTE_EVENTS_CHAR:
                self._note_ids.append(index)
            elif text[0] == tokens.POSITION_EVENTS_CHAR:
                self._position_ids.append(index)
            elif text[0] == tokens.MEASURE_EVENTS_CHAR:
                self._measure_ids.add(index)
            elif text[0] == tokens.TRACK_EVENTS_CHAR:
                self._track_ids.add(index)
            elif text[0] == tokens.TEMPO_EVENTS_CHAR:
                self._tempo_ids.add(index)
        if use_permutable_subseq_loss or permute_mps:
            self._mps_seperators = set([
                self._pad_id,
                self._bos_id,
                self.vocabs.events.text2id[tokens.TRACK_EVENTS_CHAR+'0'],
                self._sep_id,
                self._eos_id,
            ])
            self._mps_seperators.update(self._track_ids)
            self._mps_seperators.update(self._measure_ids)
            self._mps_seperators.update(self._position_ids)
            # self._mps_seperators.update(self._tempo_ids)
            self._mps_seperators = np.sort(np.array(list(self._mps_seperators)))
        # numpy.isin can work on set, but turn it into sorted 1d array helps search speed
        self._note_ids = np.sort(np.array(self._note_ids))
        self._position_ids = np.sort(np.array(self._position_ids))

        # preprocessing
        self._filenum_indices = [0] * self.number_of_pieces
        self._piece_lengths = [0] * self.number_of_pieces
        self._piece_body_start_index = [0] * self.number_of_pieces
        self._file_mps_sep_indices = [[] for _ in range(self.number_of_pieces)]
        self._augmentable_pitches = [np.empty((0,), dtype=np.bool8) for _ in range(self.number_of_pieces)]
        pitch_augmentation_factor = self.pitch_augmentation * 2 + 1 # length of (-pa, ..., -1, 0, 1, ..., pa)
        cur_index = 0
        for filenum in tqdm(range(self.number_of_pieces), disable=not verbose):
            filename = str(filenum)

            if use_permutable_subseq_loss or permute_mps:
                # find all seperater's index
                mps_sep_indices = np.flatnonzero(
                    np.isin(self.pieces[filename][:, TOKEN_ATTR_INDEX['evt']], self._mps_seperators)
                )
                for i in range(mps_sep_indices.shape[0] - 1):
                    self._file_mps_sep_indices[filenum].append(mps_sep_indices[i])
                    if mps_sep_indices[i+1] != mps_sep_indices[i] + 1:
                        self._file_mps_sep_indices[filenum].append(mps_sep_indices[i] + 1)

            if self.pitch_augmentation != 0:
                is_multinote_token = (self.pieces[filename][:, TOKEN_ATTR_INDEX['pit']]) != 0
                # the pitch attribute of drums / percussion instrument does not means actual note pitch
                # but different percussion sound
                not_percussion_notes = (self.pieces[filename][:, TOKEN_ATTR_INDEX['ins']]) != 129
                self._augmentable_pitches[filenum] = np.logical_and(is_multinote_token, not_percussion_notes)

            cur_piece_lengths = int(self.pieces[filename].shape[0])
            self._piece_lengths[filenum] = cur_piece_lengths
            cur_index += pitch_augmentation_factor
            self._filenum_indices[filenum] = cur_index
        self._max_index = cur_index
        self._pitch_augmentation_factor = pitch_augmentation_factor
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
        pitch_augment = index if filenum == 0 else index - self._filenum_indices[filenum-1] - 1
        if self._piece_lengths[filenum] <= self.max_seq_length:
            end_index = self._piece_lengths[filenum]
        else:
            end_index = self.max_seq_length
        pitch_augment -= self.pitch_augmentation
        sliced_array = np.array(self.pieces[str(filenum)][:end_index]) # copy
        sliced_array = sliced_array.astype(np.int32) # was int16 to save space but torch ask for int

        # pitch augmentation
        # assume pitch vocabulary is 0:PAD, 1:0, 2:1, ... 128:127
        # assume instrument vocabulary is 0:PAD, 1:0, 2:1, ... 128:127, 129:128(drums)
        if pitch_augment != 0:
            sliced_augmentable_pitches = self._augmentable_pitches[filenum][:end_index]
            # sometimes the sliced array does not contain any pitch-augmentable token
            if np.any(sliced_augmentable_pitches):
                if pitch_augment > 0:
                    max_pitch = np.max( (sliced_array[:, TOKEN_ATTR_INDEX['pit']])[sliced_augmentable_pitches] ) + pitch_augment
                    if max_pitch > 128:
                        pitch_augment = 128 - max_pitch
                else: # pitch_augment < 0
                    min_pitch = np.min( (sliced_array[:, TOKEN_ATTR_INDEX['pit']])[sliced_augmentable_pitches] ) + pitch_augment
                    if min_pitch < 1:
                        pitch_augment = 1 - min_pitch
                (sliced_array[:, TOKEN_ATTR_INDEX['pit']])[sliced_augmentable_pitches] += pitch_augment

        if self.permute_track_number:
            # amortize the calculation
            if self._piece_body_start_index[filenum] == 0:
                j = 2 # because first token is BOS and second must be track as we excluded all empty midis
                while self.pieces[str(filenum)][j][0] in self._track_ids:
                    j += 1
                self._piece_body_start_index[filenum] = j

            # use self._piece_body_start_index to know how many tracks there are in this piece
            track_count = self._piece_body_start_index[filenum] - 1
            body_start_index = self._piece_body_start_index[filenum]
            # add one because there is a padding token at the beginning of the vocab
            perm_array = np.random.permutation(track_count) + 1
            # view body's track number col
            body_trn_column = sliced_array[body_start_index:, TOKEN_ATTR_INDEX['trn']]
            body_trn_column_expand = np.asarray([
                (body_trn_column == i + 1) for i in range(track_count)
            ])
            # permute body's track number
            ins_col_index = TOKEN_ATTR_INDEX['ins']
            for i in range(track_count):
                body_trn_column[body_trn_column_expand[i]] = perm_array[i]
            # permute head's track instruments with the inverse of the permutation array
            if body_start_index > 0:
                # head_start_index = 1
                inv_perm_array = np.empty(track_count, dtype=np.int16)
                inv_perm_array[perm_array-1] = (np.arange(track_count) + 1)
                track_permuted_ins = np.empty(track_count, dtype=np.int16)
                track_permuted_ins = self.pieces[str(filenum)][inv_perm_array, ins_col_index]
                sliced_array[1:body_start_index, ins_col_index] = track_permuted_ins

        mps_sep_indices_list = []
        if self.permute_mps:
            mps_tokens_ranges = []
            mps_sep_indices_list = [
                i
                for i in self._file_mps_sep_indices[filenum]
                if i < end_index
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
                    i
                    for i in self._file_mps_sep_indices[filenum]
                    if i < end_index
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
