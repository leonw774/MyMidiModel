import json
import numpy as np
import os
from torch import from_numpy
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MidiMultiHeadDataset(Dataset):

    def __init__(
            self,
            data_dir_path: str,
            max_sample_length: int,
            sample_stride: int,
            sample_in_equaltime_sequence: bool,
            sample_always_include_head: bool,
            permute_notes_in_position: bool,
            puermute_positions_in_measure: bool,
            permute_track_number: bool,
            global_local_masking: bool,
            measure_number_augument_range: int
        ) -> None:
        """
            Parameters:
            - data_dir_path: Expected to have 'data.npz' and 'vocabs.json'
            - max_sample_length: The unit is always token
            - sample_stride: The moving window moves `sample_stride` to right for the next sample.
            - sample_in_equaltime_sequence: Use "equal-time sequence" as unit to split sample. Ignore `sample_stride`.
              An "equal-time sequence" can be:
              - A "BOS", "EOS" or position token. (sequence of length one)
              - The track tokens in the head section.
              - A continuous sequence of measure-related tokens, i.e. "Measure", "Tempo", "TimeSignature", in that order.
              - A continuous sequence of note tokens
            - sample_always_include_head: sample will alway have head section no matter is starts with measure tokens or not
            - permute_notes_in_position: Randomly order the notes in a position sequence
            - puermute_positions_in_measure: Randomly order the position sequences in a measure
            - permute_tracks: Permute all the track numbers
            - measure_number_augument_range: the measure number of sample will increase or decrease in range of it
        """
        self.piece_files = np.load(os.path.join(data_dir_path, 'arrays.npz'))
        # event, duration, velocity, track_number, instrument, tempo, position, measure_number
        self.vocabs = json.load(os.path.join(data_dir_path, 'vocabs.json'))
        self._max_sample_length = max_sample_length
        self._sample_stride = sample_stride
        self._sample_in_equaltime_sequence = sample_in_equaltime_sequence
        self._sample_always_include_head = sample_always_include_head
        self._permute_notes_in_position = permute_notes_in_position
        self._permute_track_number = permute_track_number
        self._global_local_masking = global_local_masking
        self._measure_number_augument_range = measure_number_augument_range

        # Seperators are:
        # EOS, BOS, PAD, first track token (R0), tempo tokens (T\w+), measure tokens (M\w+),position tokens (P\w+)
        # stores thier index in event vocab, empty when `sample_in_equaltime_sequence` is not True
        self._equaltime_sequence_seperators = set()
        self._bos_id = self.vocabs['events']['text2id']['BOS']
        self._eos_id = self.vocabs['events']['text2id']['EOS']
        # the pad id is same cross all vocab
        self._pad_id = self.vocabs['events']['text2id']['PAD']
        self._note_ids = set()
        self._measure_ids = set()
        self._position_ids = set()
        self._tempo_ids = set()
        for key in self.vocabs['events']['text2id']:
            if key[0] == 'N':
                self._note_ids.add(key)
            elif key[0] == 'P':
                self._position_ids.add(key)
            elif key[0] == 'M':
                self._measure_ids.add(key)
            elif key[0] == 'T':
                self._tempo_ids.add(key)
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

        # for every pieces array, find first occurence of measure token, then cache the head section
        self._piece_heads = []
        self._max_head_length = 0
        if sample_always_include_head:
            for filename in self.piece_files:
                j = 0
                while self.piece_files[filename][j][0] != self._measure_id:
                    j += 1
                max_head_length = max(max_head_length, i)
                self._piece_heads.append(np.array(self.piece_files[filename][:j]))

        # (pieces_array_index, start_index, end_index)
        self._samples_indices = []
        self._equaltime_sequence_sep_indices = None
        if sample_in_equaltime_sequence:
            self._equaltime_sequence_sep_indices = {filename: [] for filename in self.piece_files}
        for filename in self.piece_files:
            if sample_in_equaltime_sequence:
                # find all split index
                # ets = equl-time sequence
                ets_sep_indices = np.flatnonzero(np.isin(self.piece_files[filename][:, 0], self._equaltime_sequence_seperators))
                self._equaltime_sequence_sep_indices[filename].append(ets_sep_indices)
                begin_ptr = 0
                end_ptr = 0
                while True:
                    if ets_sep_indices[end_ptr] - ets_sep_indices[begin_ptr] >= max_sample_length-self._max_head_length:
                        begin_index = ets_sep_indices[begin_ptr]
                        end_index = ets_sep_indices[end_ptr-1]
                        self._samples_indices.append(
                            (filename, begin_index, end_index, begin_ptr, end_ptr-1)
                        )
                        begin_ptr += 1
                    else:
                        end_ptr += 1
                        if end_ptr == ets_sep_indices.shape[0]:
                            # because we need a sample that include EOS
                            self.samples_indices.append(
                                # end_prt-1 should point to where EOS is
                                (filename, ets_sep_indices[begin_ptr], self.piece_files[filename].shape[0], begin_ptr, end_ptr)
                            )
                            break
            else:
                for begin_index in range(
                        0,
                        self.piece_files[filename].shape[0] - max_sample_length - self._max_head_length,
                        sample_stride):
                    self._samples_indices.append((filename, begin_index, begin_index+max_sample_length-self._max_head_length))


    def __getitem__(self, index):
        if self._sample_in_equaltime_sequence:
            filename, begin_index, end_index, ets_sep_begin_ptr, ets_sep_end_ptr = self._samples_indices[index]
        else:
            filename, begin_index, end_index = self._samples_indices[index]
        sliced_array = np.array(self.piece_files[filename][begin_index:end_index]) # copy

        if self._sample_always_include_head:
            # do we need a [SEP] token to seperate head and body?
            sliced_array = np.concatenate(self._piece_heads[filename], sliced_array, axis=0)

        if self._permute_track_number:
            # use self._piece_heads to know how many tracks there are in this piece
            track_count = self._piece_heads[filename].shape[0] - 1
            perm_array = np.random.permutation(track_count)
            track_num_column = sliced_array[:, 3] # view
            track_num_column_expand = np.asarray([
                (track_num_column == i) for i in range(track_count)
            ])
            for i in range(perm_array.shape[0]):
                track_num_column[track_num_column_expand[i]] = perm_array[i]

        if self._permute_notes_in_position:
            ets_note_ranges = []
            start = None
            for i in range(sliced_array.shape[0]):
                if start is not None:
                    if sliced_array[i] not in self._note_ids:
                        # assert start < i
                        ets_note_ranges.append((start, i))
                        start = None
                if sliced_array[i][0] in self._position_ids:
                    start = i + 1
            for start, end in ets_note_ranges:
                p = np.random.permutation(end-start) + start
                sliced_array[start:end] = sliced_array[p]

        # decrease all measure number so that the first measure number is 0
        min_measure_num = np.min(sliced_array[:,7])
        # assert max_measure_num - min_measure_num < self.vocabs['max_measure_number']
        sliced_array[:,7] -= min_measure_num

        # measure number augumentation: move them up in range of self._measure_number_augument_range
        if self._measure_number_augument_range != 0:
            max_measure_num = np.max(sliced_array[:,7])
            increasable_range = min(self._measure_number_augument_range, self.vocabs['max_measure_number']-max_measure_num)
            increase_num = np.random.randint(0, increasable_range)
            sliced_array[:,7] += increase_num

        if self._global_local_masking:
            pass

        # a numpy array of sequences sep indices would also be returned if self._sample_in_equaltime_sequence
        if self._sample_in_equaltime_sequence:
            ets_seps_indices = self._equaltime_sequence_sep_indices[filename][ets_sep_begin_ptr:ets_sep_end_ptr]
            return from_numpy(sliced_array), ets_seps_indices

        return from_numpy(sliced_array)


    def __del__(self):
        # because self.piece_files is a NpzFile object which could be a memory-mapped to a opened file
        # use context management (__enter__, __exit__) would be better but it shouldn't cause big trouble
        # because we will only access the file once in the whole training process
        self.piece_files.close()


    def __len__(self):
        return len(self._samples_indices)

# stack batched samples on 3rd dimension and pad them to max_sample_length
def collate_fn(data):
    pass
