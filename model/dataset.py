import json
import os
import sys

import numpy as np
from torch import from_numpy
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MidiDataset(Dataset):

    def __init__(
            self,
            data_dir_path: str,
            max_sample_length: int,
            sample_stride: int,
            sample_in_equaltime_sequence: bool,
            sample_always_include_head: bool,
            permute_notes_in_position: bool,
            permute_track_number: bool,
            measure_number_augument_range: int
        ) -> None:
        """
            Parameters:
            - data_dir_path: Expected to have 'data.npz' and 'vocabs.json'
            - max_sample_length: The unit is always token
            - sample_stride: The moving window moves `sample_stride` to right for the next sample.
            - sample_in_equaltime_sequence: Use "equal-time sequence" as unit to split sample. Ignore `sample_stride`.
              "Equal-time sequence" is sequence of consecutive tokens that have same type, same measure and same position
              - ETS of length one: "BOS", "EOS", measure, position and tempo token. 
              - The track tokens in the head section is a ets
              - A consecutive sequence of note tokens is a ets
            - sample_always_include_head: sample will alway have head section no matter is starts with measure tokens or not
            - permute_notes_in_position: Randomly order the notes in a position sequence
            - permute_tracks: Permute all the track numbers
            - measure_number_augument_range: the measure number of sample will increase or decrease in range of it
        """
        self.piece_files = np.load(os.path.join(data_dir_path, 'arrays.npz'))
        print('loaded', str(os.path.join(data_dir_path, 'arrays.npz')))
        # event, duration, velocity, track_number, instrument, tempo, position, measure_number
        with open(os.path.join(data_dir_path, 'vocabs.json'), 'r') as vocabs_file:
            self.vocabs = json.load(vocabs_file)
        self.max_sample_length = max_sample_length
        assert max_sample_length > 0
        self.sample_stride = sample_stride
        self.sample_in_equaltime_sequence = sample_in_equaltime_sequence
        self.sample_always_include_head = sample_always_include_head
        self.permute_notes_in_position = permute_notes_in_position
        self.permute_track_number = permute_track_number
        self.measure_number_augument_range = measure_number_augument_range

        # Seperators are:
        # EOS, BOS, PAD, first track token (R0), tempo tokens (T\w+), measure tokens (M\w+),position tokens (P\w+)
        # stores thier index in event vocab, empty when `sample_in_equaltime_sequence` is not True
        self._equaltime_sequence_seperators = set()
        self._bos_id = self.vocabs['events']['text2id']['BOS']
        self._eos_id = self.vocabs['events']['text2id']['EOS']
        # the pad id is same cross all vocab
        self._pad_id = self.vocabs['events']['text2id']['[PAD]']
        self._special_tokens = self.vocabs['special_tokens']
        self._note_ids = set()
        self._measure_ids = set()
        self._position_ids = set()
        self._tempo_ids = set()
        for text, index in self.vocabs['events']['text2id'].items():
            if text[0] == 'N':
                self._note_ids.add(index)
            elif text[0] == 'P':
                self._position_ids.add(index)
            elif text[0] == 'M':
                self._measure_ids.add(index)
            elif text[0] == 'T':
                self._tempo_ids.add(index)
        if sample_in_equaltime_sequence:
            self._equaltime_sequence_seperators = set([
                self._bos_id,
                self._eos_id,
                self.vocabs['events']['text2id']['R0'],
                self._pad_id,
            ])
            self._equaltime_sequence_seperators.update(self._measure_ids)
            self._equaltime_sequence_seperators.update(self._position_ids)
            self._equaltime_sequence_seperators.update(self._tempo_ids)
        # numpy.isin can work on set, turn it into sorted 1d array helps search speed
        self._note_ids = np.sort(np.array(list(self._note_ids)))
        self._position_ids = np.sort(np.array(list(self._position_ids)))
        self._equaltime_sequence_seperators = np.sort(np.array(list(self._equaltime_sequence_seperators)))

        # for every pieces array, find first occurence of measure token, then cache the head section
        self._max_head_length = 0
        self._piece_heads = [None] * len(self.piece_files)
        for filename in self.piece_files:
            # `filename` is strings of integers
            filenum = int(filename)
            j = 0
            while self.piece_files[filename][j][0] not in self._measure_ids:
                j += 1
            self._max_head_length = max(self._max_head_length, j)
            self._piece_heads[filenum] = np.array(self.piece_files[filename][:j])

        # (pieces_array_index, start_index, end_index)
        self._samples_indices = []
        self._equaltime_sequence_sep_indices = None
        if sample_in_equaltime_sequence:
            self._equaltime_sequence_sep_indices = [None] * len(self.piece_files)
        for filename in self.piece_files:
            filenum = int(filename)
            max_body_length = max_sample_length - self._max_head_length if sample_always_include_head else max_sample_length
            if sample_in_equaltime_sequence:
                # find all split index
                # ets = equl-time sequence
                # this ets_sep_indices does not include the first note of every position
                # because we want position and its note sequence come in pairs
                ets_sep_indices = np.flatnonzero(np.isin(self.piece_files[filename][:, 0], self._equaltime_sequence_seperators))
                self._equaltime_sequence_sep_indices[filenum] = ets_sep_indices
                if sample_always_include_head:
                    begin_ptr = 2
                    end_ptr = 2
                else:
                    begin_ptr = 0
                    end_ptr = 0
                while True:
                    if ets_sep_indices[end_ptr] - ets_sep_indices[begin_ptr] >= max_body_length:
                        begin_index = ets_sep_indices[begin_ptr]
                        end_index = ets_sep_indices[end_ptr-1]
                        if begin_index < end_index:
                            self._samples_indices.append(
                                (filenum, begin_index, end_index, begin_ptr, end_ptr-1)
                            )
                            begin_ptr += 1
                        else:
                            end_ptr += 1
                            begin_ptr += 1
                    else:
                        end_ptr += 1
                        if end_ptr == ets_sep_indices.shape[0]:
                            # because we need a sample that include EOS
                            if ets_sep_indices[begin_ptr] < self.piece_files[filename].shape[0]:
                                self._samples_indices.append(
                                    # end_prt-1 should point to where EOS is
                                    (filenum, ets_sep_indices[begin_ptr], self.piece_files[filename].shape[0], begin_ptr, end_ptr)
                                )
                            break
            else:
                for begin_index in range(
                        self._piece_heads[filenum].shape[0],
                        self.piece_files[filename].shape[0] - max_body_length,
                        sample_stride):
                    self._samples_indices.append(
                        (filenum, begin_index, begin_index + max_body_length)
                    )
        # cast self._samples_indices from list of tuples into np array to save space
        self._samples_indices = np.array(self._samples_indices)
        print('self._equaltime_sequence_sep_indices size:', sys.getsizeof(self._equaltime_sequence_sep_indices) / 1024, 'KB')
        print('self._samples_indices size:', sys.getsizeof(self._samples_indices) / 1024, 'KB')

    def __getitem__(self, index):
        if self.sample_in_equaltime_sequence:
            filenum, begin_index, end_index, ets_sep_begin_ptr, ets_sep_end_ptr = self._samples_indices[index]
        else:
            filenum, begin_index, end_index = self._samples_indices[index]
        filename = str(filenum)
        sliced_array = np.array(self.piece_files[filename][begin_index:end_index]) # copy

        # decrease all measure number so that the first measure number is 1
        min_measure_num = np.min(sliced_array[:,7])
        assert np.max(sliced_array[:,7]) - min_measure_num + 1 < self.vocabs['max_measure_number']
        sliced_array[:,7] -= min_measure_num

        # measure number augumentation: move them up in range of self.measure_number_augument_range
        if self.measure_number_augument_range != 0:
            max_measure_num = np.max(sliced_array[:,7])
            increasable_range = min(self.measure_number_augument_range, self.vocabs['max_measure_number']-max_measure_num)
            increase_num = np.random.randint(0, increasable_range)
            sliced_array[:,7] += increase_num

        if self.sample_always_include_head:
            # do we need a [SEP] token to seperate head and body?
            sliced_array = np.concatenate((self._piece_heads[filenum], sliced_array), axis=0)

        if self.permute_track_number:
            # use self._piece_heads to know how many tracks there are in this piece
            track_count = self._piece_heads[filenum].shape[0] - 1
            # trn number as token index starts from len(self._special_tokens)
            perm_array = np.random.permutation(track_count) + len(self._special_tokens)
            track_num_column = sliced_array[:, 3] # view
            track_num_column_expand = np.asarray([
                (track_num_column == i + len(self._special_tokens)) for i in range(track_count)
            ])
            for i in range(perm_array.shape[0]):
                track_num_column[track_num_column_expand[i]] = perm_array[i]

        if self.permute_notes_in_position:
            ets_note_ranges = []
            start = None
            is_note = np.isin(sliced_array[:, 0], self._note_ids)
            for i in range(1, sliced_array.shape[0]):
                if (not is_note[i-1]) and (is_note[i]):
                    start = i
                if start is not None and (is_note[i-1]) and (not is_note[i]):
                    if i - start > 1:
                        ets_note_ranges.append((start, i))
                    start = None
            if start is not None and sliced_array.shape[0] - start > 1:
                ets_note_ranges.append((start, sliced_array.shape[0]))
            # print(ets_note_ranges)
            for start, end in ets_note_ranges:
                p = np.random.permutation(end-start) + start
                sliced_array[start:end] = sliced_array[p]

        # a numpy array of sequences sep indices would also be returned if self.sample_in_equaltime_sequence
        if self.sample_in_equaltime_sequence:
            ets_sep_indices = np.array( # make copy
                self._equaltime_sequence_sep_indices[filenum][ets_sep_begin_ptr:ets_sep_end_ptr]
            )
            # change to relative indices
            assert begin_index == ets_sep_indices[0]
            if self.sample_always_include_head:
                ets_sep_indices -= (begin_index - self._piece_heads[filenum].shape[0])
                head_ets_sep = np.array([0, 1])
                ets_sep_indices = np.concatenate((head_ets_sep, ets_sep_indices))
            else:
                ets_sep_indices -= begin_index
            # add begin of note sequence
            note_sequence_begin_indices = np.flatnonzero(np.isin(sliced_array[:, 0], self._position_ids)) + 1
            ets_sep_indices = np.sort(np.concatenate((ets_sep_indices, note_sequence_begin_indices)))
            return from_numpy(sliced_array), ets_sep_indices

        return from_numpy(sliced_array), None


    def __del__(self):
        # because self.piece_files is a NpzFile object which could be a memory-mapped to a opened file
        # use context management (__enter__, __exit__) would be better but it shouldn't cause big trouble
        # because we will only access the file once in the whole training process
        self.piece_files.close()


    def __len__(self):
        return len(self._samples_indices)


def collate_mididataset(data):
    """
        for batched samples: stack them on batch-first to becom 3-d array and pad them
        for others, become 1-d object numpy array
    """
    # print('\n'.join(map(str, [d for d in data])))
    samples, ets_sep_indices = zip(*data)
    batched_samples = pad_sequence(samples, batch_first=True, padding_value=0)
    # could be None and should be passed as numpy object array
    batched_ets_sep_indices = np.array(ets_sep_indices, dtype=object)
    return batched_samples, batched_ets_sep_indices
