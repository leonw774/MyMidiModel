import json
import os
import sys
import bisect

import numpy as np
from torch import from_numpy
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from util import to_vocabs_file_path, ATTR_NAME_TO_FEATURE_INDEX
from util.tokens import BEGIN_TOKEN_STR, END_TOKEN_STR

class MidiDataset(Dataset):

    def __init__(
            self,
            data_dir_path: str,
            max_seq_length: int,
            sample_stride: int,
            use_permutable_subseq_loss: bool,
            permute_mps: bool,
            permute_track_number: bool = False,
            measure_number_shift_range: int = 0
        ) -> None:
        """
            Parameters:
            - data_dir_path: Expected to have 'data.npz' and 'vocabs.json'
            - sample_stride: The moving window moves `sample_stride` to right for the next sample.
            - use_permutable_subseq_loss: Whether or not we provide a mps_seperator_indices information at __getitem__
            - permute_mps: Whether or not the dataset should permute all the *maximal permutable subsequences*
              in the sequence before returning in `__getitem__`
            - permute_tracks: Permute all the track numbers, as data augmentation
            - measure_number_shift_range: the measure number of sample will increase or decrease in range of this value,
              as data augmentation
        """
        npz_path = os.path.join(data_dir_path, 'arrays.npz')
        self.piece_files = np.load(npz_path)
        print('Loaded', npz_path)
        # event, duration, velocity, track_number, instrument, tempo, position, measure_number
        with open(to_vocabs_file_path(data_dir_path), 'r', encoding='utf8') as vocabs_file:
            self.vocabs = json.load(vocabs_file)
        self.max_seq_length = max_seq_length
        self.sample_stride = sample_stride
        self.use_permutable_subseq_loss = use_permutable_subseq_loss
        self.permute_mps = permute_mps
        self.permute_track_number = permute_track_number
        self.measure_number_shift_range = measure_number_shift_range

        # Seperators are:
        # EOS, BOS, PAD, first track token (R0), measure tokens (M\w+), position tokens (P\w+)
        # stores thier index in event vocab, empty when `use_permutable_subseq_loss` is not True
        self._mps_seperators = set()
        self._bos_id = self.vocabs['events']['text2id'][BEGIN_TOKEN_STR]
        self._eos_id = self.vocabs['events']['text2id'][END_TOKEN_STR]
        # the pad id is the same across all vocabs
        # self._pad_id = self.vocabs['events']['text2id']['[PAD]']
        self._pad_id = self.vocabs['events']['text2id']['EOS']
        self._padding_token = self.vocabs['padding_token']
        self._note_ids = set()
        self._measure_ids = set()
        self._position_ids = set()
        self._tempo_ids = set()
        self._track_ids = set()
        for text, index in self.vocabs['events']['text2id'].items():
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
                self.vocabs['events']['text2id']['R0'],
                self._pad_id,
            ])
            self._mps_seperators.update(self._measure_ids)
            self._mps_seperators.update(self._position_ids)
        # numpy.isin can work on set, turn it into sorted 1d array helps search speed
        self._note_ids = np.sort(np.array(list(self._note_ids)))
        self._position_ids = np.sort(np.array(list(self._position_ids)))
        self._mps_seperators = np.sort(np.array(list(self._mps_seperators)))

        # preprocessing
        self._filenum_indices = [] # (pieces_array_index, start_index, end_index)
        self._piece_body_start_index = [0] * len(self.piece_files) if self.permute_track_number else None
        self._mps_sep_indices = [0] * len(self.piece_files) if use_permutable_subseq_loss else None
        cur_index = 0
        for filename in tqdm(self.piece_files, total=len(self.piece_files)):
            filenum = int(filename)

            if use_permutable_subseq_loss:
                # find all seperater's index
                mps_sep_indices = np.flatnonzero(np.isin(self.piece_files[filename][:, 0], self._mps_seperators))
                self._mps_sep_indices[filenum] = mps_sep_indices.astype(np.int16)

            if self.piece_files[filename].shape[0] > max_seq_length:
                cur_index += self.piece_files[filename].shape[0] - max_seq_length
            else:
                cur_index += 1
            self._filenum_indices.append(cur_index)
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
        begin_index = self._filenum_indices[filenum] - index
        filename = str(filenum)
        if self.piece_files[filename].shape[0] > self.max_seq_length:
            end_index = self.piece_files[filename].shape[0]
        else:
            end_index = begin_index + self.max_seq_length
        sliced_array = np.array(self.piece_files[filename][begin_index:end_index]) # copy
        sliced_array = sliced_array.astype(np.int64) # was int16 to save space but torch ask for long

        # shift down measure number so that it didn't exceed max_seq_length
        shift_value = 0
        max_measure_num = sliced_array[-1, ATTR_NAME_TO_FEATURE_INDEX['mea']]
        min_measure_num = sliced_array[0, ATTR_NAME_TO_FEATURE_INDEX['mea']]
        if max_measure_num > self.max_seq_length:
            shift_value = max_measure_num - self.max_seq_length
            assert min_measure_num != 0
        # measure number augumentation: move them up or down in range of self.measure_number_shift_range
        if self.measure_number_shift_range != 0:
            if shift_value != 0:
                shiftable_range_upper = 0
            else:
                shiftable_range_upper = min(self.measure_number_shift_range, self.max_seq_length - max_measure_num)
            if min_measure_num != 0:
                shiftable_range_lower = min(self.measure_number_shift_range, min_measure_num - 1)
            else:
                shiftable_range_lower = 0
            shift_value += np.random.randint(-shiftable_range_lower, shiftable_range_upper)
        sliced_array[:, ATTR_NAME_TO_FEATURE_INDEX['mea']] += shift_value

        if self.permute_track_number:
            # amortize the calculation
            if self._piece_body_start_index[filenum] == 0:
                j = 2 # because first token is BOS and second must be track as we excluded all empty midis
                while self.piece_files[filename][j][0] in self._track_ids:
                    j += 1
                self._piece_body_start_index[filenum] = j

            # use self._piece_body_start_index to know how many tracks there are in this piece
            track_count = self._piece_body_start_index[filenum] - 1
            # add one because there is a padding token at the beginning of the vocab
            perm_array = np.random.permutation(track_count) + 1
            track_num_column = sliced_array[:, ATTR_NAME_TO_FEATURE_INDEX['trn']] # view
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
            mps_sep_indices = np.array( # make copy
                self._mps_sep_indices[filenum]
            )
            mps_sep_indices = mps_sep_indices[(mps_sep_indices >= begin_index) & (mps_sep_indices < end_index)]
            mps_sep_indices -= begin_index
            # add begin of note sequence
            note_sequence_begin_indices = np.flatnonzero(np.isin(sliced_array[:, 0], self._position_ids)) + 1
            mps_sep_indices = np.sort(np.concatenate((mps_sep_indices, note_sequence_begin_indices)))
            return from_numpy(sliced_array), mps_sep_indices

        return from_numpy(sliced_array), None


    # def __del__(self):
        # because self.piece_files is a NpzFile object which could be a memory-mapped to a opened file
        # use context management (__enter__, __exit__) would be better but it shouldn't cause big trouble
        # because we will only open the file once in the whole training process
        # self.piece_files.close()


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
