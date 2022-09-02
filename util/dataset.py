import json
import os
import sys

import numpy as np
from torch import from_numpy
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from util import to_vocabs_file_path, ATTR_NAME_TO_FEATURE_INDEX

class MidiDataset(Dataset):

    def __init__(
            self,
            data_dir_path: str,
            max_seq_length: int,
            sample_stride: int,
            use_set_loss: bool,
            permute_mps: bool,
            permute_track_number: bool = False,
            measure_number_shift_range: int = 0
        ) -> None:
        """
            Parameters:
            - data_dir_path: Expected to have 'data.npz' and 'vocabs.json'
            - sample_stride: The moving window moves `sample_stride` to right for the next sample.
            - use_set_loss: Whether or not we provide a mps_seperator_indices information at __getitem__
            - permute_mps: Whether or not the dataset should permute all the *maximal permutable subsequences*
              in the sequence before returning in `__getitem__`
            - permute_tracks: Permute all the track numbers, as data augmentation
            - measure_number_shift_range: the measure number of sample will increase or decrease in range of this value,
              as data augmentation
        """
        npz_path = os.path.join(data_dir_path, 'arrays.npz')
        self.piece_files = np.load(npz_path)
        print('loaded', npz_path)
        # event, duration, velocity, track_number, instrument, tempo, position, measure_number
        with open(to_vocabs_file_path(data_dir_path), 'r', encoding='utf8') as vocabs_file:
            self.vocabs = json.load(vocabs_file)
        self.max_seq_length = self.vocabs['max_seq_length']
        self.sample_stride = sample_stride
        self.use_set_loss = use_set_loss
        self.permute_mps = permute_mps
        self.permute_track_number = permute_track_number
        self.measure_number_shift_range = measure_number_shift_range

        # Seperators are:
        # EOS, BOS, PAD, first track token (R0), measure tokens (M\w+), position tokens (P\w+)
        # stores thier index in event vocab, empty when `use_set_loss` is not True
        self._mps_seperators = set()
        self._bos_id = self.vocabs['events']['text2id']['BOS']
        self._eos_id = self.vocabs['events']['text2id']['EOS']
        # the pad id is the same across all vocabs
        # self._pad_id = self.vocabs['events']['text2id']['[PAD]']
        self._pad_id = 0
        self._special_tokens = self.vocabs['special_tokens']
        self._note_ids = set()
        self._measure_ids = set()
        self._position_ids = set()
        self._tempo_ids = set()
        for text, index in self.vocabs['events']['text2id'].items():
            if text[0] == 'N' or text[0] == 'S':
                self._note_ids.add(index)
            elif text[0] == 'P':
                self._position_ids.add(index)
            elif text[0] == 'M':
                self._measure_ids.add(index)
            elif text[0] == 'T':
                self._tempo_ids.add(index)
        if use_set_loss:
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
        self._piece_body_start_index = [None] * len(self.piece_files)
        self._mps_sep_indices = [None] * len(self.piece_files) if use_set_loss else None
        for filename in tqdm(self.piece_files, total=len(self.piece_files)):
            filenum = int(filename)

            j = 0
            while self.piece_files[filename][j][0] not in self._measure_ids:
                j += 1
            self._piece_body_start_index[filenum] = j

            if use_set_loss:
                # find all seperater's index
                mps_sep_indices = np.flatnonzero(np.isin(self.piece_files[filename][:, 0], self._mps_seperators))
                self._mps_sep_indices[filenum] = mps_sep_indices

            if self.piece_files[filename].shape[0] > max_seq_length:
                for begin_index in range(0, self.piece_files[filename].shape[0] - max_seq_length, sample_stride):
                    self._filenum_indices.append(
                        (filenum, begin_index, begin_index + max_seq_length)
                    )
            else:
                self._filenum_indices.append(
                    (filenum, 0, self.piece_files[filename].shape[0])
                )
        # cast self._filenum_indices from list of tuples into np array to save space
        print('self._mps_sep_indices size:', sys.getsizeof(self._mps_sep_indices) / 1024, 'KB')
        print('self._filenum_indices len:', len(self._filenum_indices))
        print('self._filenum_indices size:', sys.getsizeof(self._filenum_indices) / 1024, 'KB')

    def __getitem__(self, index):
        filenum, begin_index, end_index = self._filenum_indices[index]
        filename = str(filenum)
        sliced_array = np.array(self.piece_files[filename][begin_index:end_index]) # copy

        # measure number augumentation: move them up in range of self.measure_number_shift_range
        if self.measure_number_shift_range != 0:
            max_measure_num = np.max(sliced_array[:, ATTR_NAME_TO_FEATURE_INDEX['mea']])
            increasable_range = min(self.measure_number_shift_range, self.vocabs['max_seq_length']-max_measure_num)
            increase_num = np.random.randint(0, increasable_range)
            sliced_array[:, ATTR_NAME_TO_FEATURE_INDEX['mea']] += increase_num

        if self.permute_track_number:
            # use self.self._piece_body_start_index to know how many tracks there are in this piece
            track_count = self._piece_body_start_index[filenum] - 1
            # trn number as token index starts from len(self._special_tokens)
            perm_array = np.random.permutation(track_count) + len(self._special_tokens)
            track_num_column = sliced_array[:, ATTR_NAME_TO_FEATURE_INDEX['trn']] # view
            track_num_column_expand = np.asarray([
                (track_num_column == i + len(self._special_tokens)) for i in range(track_count)
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

        # a numpy array of sequences sep indices would also be returned if self.use_set_loss
        if self.use_set_loss:
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
        return len(self._filenum_indices)


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
