import bisect
import os
import sys

import numpy as np
import psutil
from torch import from_numpy
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from util import get_corpus_vocabs, FEATURE_INDEX
from util.tokens import BEGIN_TOKEN_STR, END_TOKEN_STR

class MidiDataset(Dataset):

    def __init__(
            self,
            data_dir_path: str,
            max_seq_length: int,
            sample_stride: int,
            use_permutable_subseq_loss: bool,
            permute_mps: bool,
            permute_track_number: bool
        ) -> None:
        """
            Parameters:
            - data_dir_path: Expected to have 'data.npz' and 'vocabs.json'
            - sample_stride: The moving window moves `sample_stride` to right for the next sample.
            - use_permutable_subseq_loss: Whether or not we provide a mps_seperator_indices information at __getitem__
            - permute_mps: Whether or not the dataset should permute all the *maximal permutable subsequences*
              in the sequence before returning in `__getitem__`
            - permute_track_number: Permute all the track numbers, as data augmentation
        """
        npz_path = os.path.join(data_dir_path, 'arrays.npz')
        npz_file = np.load(npz_path)
        print('Loaded', npz_path)
        available_memory_size = psutil.virtual_memory().available
        array_memory_size = 0
        for filenum in range(len(npz_file)):
            filename = str(filenum)
            array_memory_size += np.prod(npz_file[filename].shape) * 2 # times 2 because array dtype is int16
        if array_memory_size >= available_memory_size:
            # load from disk every time indexing
            print('Memory size not enough, using NPZ mmap.')
            self.pieces = npz_file
        else:
            # load into memory to be faster
            self.pieces = {
                str(filenum): npz_file[str(filenum)]
                for filenum in range(len(npz_file))
            }

        self.vocabs = get_corpus_vocabs(data_dir_path)
        self.max_seq_length = max_seq_length
        self.sample_stride = sample_stride
        self.use_permutable_subseq_loss = use_permutable_subseq_loss
        self.permute_mps = permute_mps
        self.permute_track_number = permute_track_number

        # Seperators are:
        # EOS, BOS, PADDING, first track token (R0), measure tokens (M\w+), position tokens (P\w+)
        # stores their index in event vocab, empty when `use_permutable_subseq_loss` is not True
        self._mps_seperators = set()
        self._bos_id = self.vocabs.events.text2id[BEGIN_TOKEN_STR]
        self._eos_id = self.vocabs.events.text2id[END_TOKEN_STR]
        # the pad id should be the same across all vocabs
        self._pad_id = self.vocabs.events.text2id[self.vocabs.padding_token]
        self._note_ids = set()
        self._measure_ids = set()
        self._position_ids = set()
        self._tempo_ids = set()
        self._track_ids = set()
        for text, index in self.vocabs.events.text2id.items():
            if text[0] == 'N' or text[0] == 'S':
                self._note_ids.add(index)
            elif text[0] == 'P':
                self._position_ids.add(index)
            elif text[0] == 'T':
                self._tempo_ids.add(index)
            elif text[0] == 'M':
                self._measure_ids.add(index)
            elif text[0] == 'R':
                self._track_ids.add(index)
        if use_permutable_subseq_loss:
            self._mps_seperators = set([
                self._bos_id,
                self._eos_id,
                self.vocabs.events.text2id['R0'],
                self._pad_id,
            ])
            self._mps_seperators.update(self._measure_ids)
            self._mps_seperators.update(self._position_ids)
        # numpy.isin can work on set, turn it into sorted 1d array helps search speed
        self._note_ids = np.sort(np.array(list(self._note_ids)))
        self._position_ids = np.sort(np.array(list(self._position_ids)))
        self._mps_seperators = np.sort(np.array(list(self._mps_seperators)))

        # preprocessing
        self._filenum_indices = [0] * len(self.pieces)
        self._piece_lengths = [0] * len(self.pieces)
        self._piece_body_start_index = [0] * len(self.pieces) if self.permute_track_number else None
        self._file_mps_sep_indices = [None] * len(self.pieces) if use_permutable_subseq_loss else None
        cur_index = 0
        for filenum in tqdm(range(len(self.pieces))):
            filename = str(filenum)

            if use_permutable_subseq_loss:
                # find all seperater's index
                mps_sep_indices = np.flatnonzero(np.isin(self.pieces[filename][:, 0], self._mps_seperators))
                self._file_mps_sep_indices[filenum] = []
                for i in range(mps_sep_indices.shape[0] - 1):
                    self._file_mps_sep_indices[filenum].append(mps_sep_indices[i])
                    if mps_sep_indices[i+1] != mps_sep_indices[i] + 1:
                        self._file_mps_sep_indices[filenum].append(mps_sep_indices[i] + 1)

            cur_piece_lengths = self.pieces[filename].shape[0]
            self._piece_lengths[filenum] = cur_piece_lengths

            if cur_piece_lengths > max_seq_length:
                cur_index += cur_piece_lengths - max_seq_length + 1
            else:
                cur_index += 1
            self._filenum_indices[filenum] = cur_index
        self._max_index = cur_index
        # cast self._filenum_indices from list of tuples into np array to save space
        print('self._filenum_indices size:', sys.getsizeof(self._filenum_indices), 'bytes')

    def __getitem__(self, index):
        # while True:
        #     if index < 0:
        #         index = self._max_index + index
        #     else:
        #         break
        filenum = bisect.bisect_left(self._filenum_indices, index)
        begin_index = index if filenum == 0 else index - self._filenum_indices[filenum-1] - 1
        if self._piece_lengths[filenum] <= self.max_seq_length:
            end_index = self._piece_lengths[filenum]
        else:
            end_index = begin_index + self.max_seq_length
        sliced_array = np.array(self.pieces[str(filenum)][begin_index:end_index]) # copy
        sliced_array = sliced_array.astype(np.int32) # was int16 to save space but torch ask for int

        # shift down measure number so that it didn't exceed max_seq_length
        min_measure_num = sliced_array[0, FEATURE_INDEX['mea']]
        if min_measure_num > 1:
            sliced_array[:, FEATURE_INDEX['mea']] -= (min_measure_num - 1)
        # because EOS also has measure_number = 0
        if sliced_array[-1, FEATURE_INDEX['mea']] < 0:
            # set EOS measure number back to zero
            sliced_array[-1, FEATURE_INDEX['mea']] = 0

        if self.permute_track_number:
            # amortize the calculation
            if self._piece_body_start_index[filenum] == 0:
                j = 2 # because first token is BOS and second must be track as we excluded all empty midis
                while self.pieces[str(filenum)][j][0] in self._track_ids:
                    j += 1
                self._piece_body_start_index[filenum] = j

            # use self._piece_body_start_index to know how many tracks there are in this piece
            track_count = self._piece_body_start_index[filenum] - 1
            # add one because there is a padding token at the beginning of the vocab
            perm_array = np.random.permutation(track_count) + 1
            track_num_column = sliced_array[:, FEATURE_INDEX['trn']] # view
            track_num_column_expand = np.asarray([
                (track_num_column == i + 1) for i in range(track_count)
            ])
            for i in range(perm_array.shape[0]):
                track_num_column[track_num_column_expand[i]] = perm_array[i]

        if self.permute_mps:
            mps_tokens_ranges = []
            start = None
            is_note = np.isin(sliced_array[:, 0], self._note_ids)
            for i in range(1, sliced_array.shape[0]):
                if (not is_note[i-1]) and (is_note[i]):
                    start = i
                if start is not None and (is_note[i-1]) and (not is_note[i]):
                    if i - start > 1:
                        mps_tokens_ranges.append((start, i))
                    start = None
            if start is not None and sliced_array.shape[0] - start > 1:
                mps_tokens_ranges.append((start, sliced_array.shape[0]))
            # print(mps_tokens_ranges)
            for start, end in mps_tokens_ranges:
                p = np.random.permutation(end-start) + start
                sliced_array[start:end] = sliced_array[p]

        # a numpy array of sequences sep indices would also be returned if self.use_permutable_subseq_loss
        if self.use_permutable_subseq_loss:
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


    # def __del__(self):
        # If self.pieces is a NpzFile object which could be a memory-mapped to a opened file
        # use context management (__enter__, __exit__) would be better but it shouldn't cause big trouble
        # but we will only open the file once in the whole training process so dont bother after all
        # self.pieces.close()


    def __len__(self):
        return self._max_index


def collate_mididataset(data):
    """
        for batched sequences: stack them on batch-first to becom 3-d array and pad them
        for others, become 1-d object numpy array if not None
    """
    # print('\n'.join(map(str, [d for d in data])))
    seqs, mps_sep_indices = zip(*data)
    batched_seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    # mps_sep_indices could be None or numpy object array of tuples
    if mps_sep_indices[0] is None:
        batched_mps_sep_indices = None
    else:
        batched_mps_sep_indices = np.array(mps_sep_indices, dtype=object)
    return batched_seqs, batched_mps_sep_indices
