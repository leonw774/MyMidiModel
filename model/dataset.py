import json
import numpy as np
import os
from torch.utils.data import Dataset, Dataloader

class MidiMultiHeadDataset(Dataset):

    def __init__(
            self,
            data_dir_path: str,
            lazy_load_data: bool,
            max_sample_length: int,
            sample_stride: int = 1,
            sample_in_equaltime_sequence: bool = False,
            sample_always_include_head: bool = False,
            permute_notes_in_position: bool = True,
            permute_track_number: bool = True,
            measure_number_augument_range: int = 1
        ) -> None:
        """
            Parameters:
            - data_dir_path: Expected to have 'data.npz' and 'vocabs.json'
            - lazy_load_data: Because data is saved in npz file, numpy will return memory-mapped array after load.
              Lazy loading can save space in memory, but will make the __getitem__ method bounded by file system I/O.
              If `lazy_load_data` is false, the entire data will be copyed into memory.
            - max_sample_length: The unit is always token
            - sample_stride: When sampling from a piece, the moving window moves `sample_stride` token(s) for the next sample.
            - sample_in_equaltime_sequence: If true, use "equal-time sequence" as unit to split smaple, and ignore `sample_stride`.
              An "equal-time sequence" can be:
              - A "BOS", "EOS" or position token. (sequence of length one)
              - The track tokens in the head section.
              - A continuous sequence of measure-related tokens, i.e. "Measure", "Tempo", "TimeSignature", in that order.
              - A continuous sequence of note tokens
            - sample_always_include_head: sample will alway have head section no matter is starts with measure tokens or not
            - permute_notes_in_position: Randomly orders the notes in the same position
            - permute_tracks: Permute all the track numbers
            - measure_number_augument_range: the measure number of sample will increase or decrease in range of it
        """
        # event, duration, velocity, track_numer, instrument, position, measure_number
        self.pieces_arrays = np.load(os.path.join(data_dir_path, 'data.npz'))
        self.lazy_load_data = lazy_load_data
        if not lazy_load_data:
            self.pieces_arrays = dict(self.pieces_arrays)

        self.vocabs = json.load(os.path.join(data_dir_path, 'vocabs.json'))
        self.max_sample_length = max_sample_length
        self.sample_stride = sample_stride
        self.sample_in_equaltime_sequence = sample_in_equaltime_sequence
        self.sample_always_include_head = sample_always_include_head
        self.permute_notes_in_position = permute_notes_in_position
        self.permute_track_number = permute_track_number
        self.measure_number_augument_range = measure_number_augument_range

        self.bos_id = self.vocabs['event_tokens']['text2id']['BOS']
        self.eos_id = self.vocabs['event_tokens']['text2id']['EOS']
        self.measure_id = self.vocabs['event_tokens']['text2id']['M']
        # event pad id = duration pad id = velocity pad id = track_number pad id
        self.pad_id = self.vocabs['event_tokens']['text2id']['PAD']
        self.position_ids = {
            self.vocabs['event_tokens']['text2id'][key]
            for key in self.vocabs['event_tokens']['text2id']
            if key[0] == 'P'
        }

        # Seperators are: EOS, BOS, first track token (R0), measure token (M), position tokens (P[0-9A-Z]+) and PAD
        # this set stores thier index in event vocab, and is empty when `sample_in_equaltime_sequence` is not set
        self.equaltime_sequence_seperator = set()
        if sample_in_equaltime_sequence:
            self.equaltime_sequence_seperator = set([
                self.bos_id,
                self.eos_id,
                self.vocabs['event_tokens']['text2id']['R0'],
                self.measure_id,
                self.pad_id,
            ])
            self.equaltime_sequence_seperator.update(self._position_ids)

        # for every pieces array, find first occurence of measure token, then cache the head section
        self.pieces_heads = dict()
        self.max_head_length = 0
        if sample_always_include_head:
            for file in self.pieces_arrays.files:
                i = 0
                while self.pieces_arrays[file][i][0] != self.measure_id:
                    i += 1
                max_head_length = max(max_head_length, i)
                self.pieces_heads[file] = np.array(self.pieces_arrays[file][:i])

        # (pieces_array_key, start_index, end_index)
        self.samples_indices = []
        for filename, array in self.pieces_arrays.items():
            if sample_in_equaltime_sequence:
                for begin_index in range(array.shape[0]):
                    if array[begin_index] in self.equaltime_sequence_seperator:
                        end_index = begin_index + (max_sample_length - self.max_head_length)
                        # if end_index exceed array length, set end_index to last index and stop finding
                        if end_index >= array.shape[0]:
                            self.samples_indices.append((filename, begin_index, array.shape[0]))
                            break
                        # find valid end index
                        while array[end_index] not in self.equaltime_sequence_seperator:
                            end_index -= 1
                        self.samples_indices.append((filename, begin_index, end_index))
            else:
                for begin_index in range(0, array.shape[0]-max_sample_length-self.max_head_length, sample_stride):
                    self.samples_indices.append((filename, begin_index, begin_index+max_sample_length-self.max_head_length))


    def __getitem__(self, index):
        filename, begin_index, end_index = self.samples_indices[index]
        sliced_array = self.pieces_arrays[filename][begin_index:end_index]

        if self.sample_always_include_head:
            pass

        if self.permute_track_number:
            # use self.pieces_heads to know how many tracks there are in this piece
            track_count = self.pieces_heads[filename].shape[0] - 1
            perm_array = np.random.permutation(track_count)
            for i in range(sliced_array.shape[0]):
                sliced_array[i][3] = perm_array[sliced_array[i][3]]

        if self.permute_notes_in_position:
            pass

        # measure number augumentation: move them up or down in range of self.


    def __del__(self):
        # because self.pieces_arrays is a NpzFile object which is memory-mapped to a opened file
        if self.lazy_load_data:
            self.pieces_arrays.close()
    

    def __len__(self):
        return len(self.samples_indices)
