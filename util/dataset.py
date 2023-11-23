import bisect
import math
import os
import sys
from typing import List
import zipfile

import numpy as np
import psutil
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from . import tokens
from .corpus import (
    get_corpus_vocabs,
    to_pathlist_file_path,
    ATTR_NAME_INDEX,
    get_full_array_string
)

class MidiDataset(Dataset):

    def __init__(
            self,
            data_dir_path: str,
            max_seq_length: int,
            excluded_path_list: List[str] = None,
            virtual_piece_step_ratio: float = 0,
            flatten_virtual_pieces: bool = True,
            permute_mps: bool = False,
            permute_track_number: bool = False,
            pitch_augmentation_range: int = 0,
            verbose: bool = False
        ) -> None:
        """
        Parameters:
        - `data_dir_path`: Expect path to corpus directory

        - `virtual_piece_step_ratio`: Should be not less than 0.
           Default is 0.
          - If > 0, over-length pieces will have multiple virtual
            pieces. The start of a virtual pieces will be from the
            measure token that has the smallest index greater than
            `max_seq_length * virtual_piece_step_ratio` * N, where N
            is positive integer. Any virtual piece starting from the
            same index are merged as one.
            If `virtual_piece_step_ratio` is `1 / max_seq_length`,
            then its the same as using all possible measures as the
            starting index.
          - If == 0, each piece has only one virtual piece, starting
            from the index 0.
        
        - `flatten_virtual_pieces`: Default is True.
            - If True, all virtual pieces has a indexing number.
            - If False, a piece, in `__getitem__`, will be randomly
              one of its virtual pieces.

        - `excluded_path_list`: The list of relative or absolute paths
          of the files to exclude.

        - `permute_mps`: Whether or not the dataset should permute all
          the maximal permutable subarrays before returning in
          `__getitem__`, as data augmentation. Default is False.

        - `permute_track_number`: Permute all the track numbers
          relative to the instruments, as data augmentation.
          Default is False.

        - `pitch_augmentation_range`: If set to P, will add a random
          value from -P to +P (all inclusive) on all pitch values as
          data augmentation. Default is 0.
        
        - `verbose`: show tqdm or not.
        """
        self.vocabs = get_corpus_vocabs(data_dir_path)
        assert max_seq_length >= 0
        self.max_seq_length = max_seq_length
        assert virtual_piece_step_ratio >= 0
        self.virtual_piece_step_ratio = virtual_piece_step_ratio
        assert isinstance(flatten_virtual_pieces, bool)
        self.flatten_virtual_pieces = flatten_virtual_pieces
        assert isinstance(permute_mps, bool)
        self.permute_mps = permute_mps
        assert isinstance(permute_track_number, bool)
        self.permute_track_number = permute_track_number
        assert pitch_augmentation_range >= 0
        self.pitch_augmentation_range = pitch_augmentation_range

        npz_path = os.path.join(data_dir_path, 'arrays.npz')
        with open(to_pathlist_file_path(data_dir_path), 'r', encoding='utf8') as pathlist_file:
            all_paths_list = [
                p.strip()
                for p in pathlist_file.readlines()
            ]
        if excluded_path_list is not None and len(excluded_path_list) != 0:
            # assert os.path.exists(test_pathlist_file_path)
            excluded_path_tuple = tuple(p.strip() for p in excluded_path_list)
        else:
            excluded_path_tuple = tuple()
        if verbose:
            print('Reading', npz_path)
        self.included_path_list = []
        self.included_piece_num = set()
        tqdm_enum_all_path_list = tqdm(
            enumerate(all_paths_list),
            desc='Exclude paths',
            disable=not verbose,
            ncols=0
        )
        for piece_num, midi_path in tqdm_enum_all_path_list:
            # use endswith because the pathlist records the relative paths from project's root
            # while excluded_path_tuple may contain relative paths from dataset's root
            if not midi_path.endswith(excluded_path_tuple):
                self.included_path_list.append(midi_path)
                self.included_piece_num.add(piece_num)

        available_memory_size = psutil.virtual_memory().available
        npz_zipinfo_list = zipfile.ZipFile(npz_path).infolist()
        array_memory_size = sum([zinfo.file_size for zinfo in npz_zipinfo_list])
        if array_memory_size >= available_memory_size - 1e9:
            if verbose:
                print(
                    f'Memory not enough. Need {array_memory_size//1000} KB. '
                    f'Only {available_memory_size//1000} KB available.'
                )
            raise OSError('Memory not enough.')
        # load numpy arrays into memory
        with np.load(npz_path) as npz_file:
            tqdm_npz_file = tqdm(
                npz_file,
                desc='Loading arrays',
                disable=not verbose,
                ncols=0
            )
            self.pieces = [
                npz_file[piece_num_str]
                for piece_num_str in tqdm_npz_file
                if int(piece_num_str) in self.included_piece_num
            ]

        self.number_of_pieces = len(self.pieces)
        # The seperators of maximal permutable subarray are:
        # BOS, EOS, SEP, PADDING, track-instrument token, measure tokens, position tokens
        # we stores their index number in _mps_separators
        # _mps_separators is empty when `permute_mps` is False
        self._mps_separators = list()
        self._bos_id = self.vocabs.events.text2id[tokens.BEGIN_TOKEN_STR]
        self._sep_id = self.vocabs.events.text2id[tokens.SEP_TOKEN_STR]
        self._eos_id = self.vocabs.events.text2id[tokens.END_TOKEN_STR]
        # the pad id should be the same across all vocabs (aka zero)
        self._pad_id = self.vocabs.events.text2id[self.vocabs.padding_token]
        self._measure_ids = list()
        for text, index in self.vocabs.events.text2id.items():
            if text[0] == tokens.MEASURE_EVENTS_CHAR:
                self._measure_ids.append(index)
        # numpy.isin can work on set, but turn it into sorted 1d array helps search speed
        self._measure_ids = np.sort(np.array(self._measure_ids))

        # preprocessing
        self._piece_num_indices = [0] * self.number_of_pieces
        self._piece_lengths = [0] * self.number_of_pieces
        self._piece_body_start_indices = [0] * self.number_of_pieces
        self._piece_measures_indices = [0] * self.number_of_pieces

        virtual_piece_step_size = max(1, int(max_seq_length * virtual_piece_step_ratio))
        # zeros will be replaced by body start index
        self._virtual_piece_start_indices = [[0] for _ in range(self.number_of_pieces)]
        self._piece_mps_sep_indices = [[] for _ in range(self.number_of_pieces)]
        self._piece_augmentable_pitches = [
            np.empty((0,), dtype=np.bool8)
            for _ in range(self.number_of_pieces)
        ]
        # length of (-pa, ..., -1, 0, 1, ..., pa)
        self._pitch_augmentation_factor = self.pitch_augmentation_range * 2 + 1

        cur_index = -1 # <-- !!! because the first element we add should become index 0
        tqdm_enum_pieces = tqdm(
            enumerate(self.pieces),
            desc='Processing',
            disable=not verbose,
            ncols=0
        )
        for piece_num, piece_array in tqdm_enum_pieces:

            cur_piece_lengths = int(piece_array.shape[0])
            self._piece_lengths[piece_num] = cur_piece_lengths

            j = 2 # because first token is BOS and second token must be track token
            while piece_array[j][ATTR_NAME_INDEX['evt']] != self._sep_id:
                j += 1
            self._piece_body_start_indices[piece_num] = j + 1
            self._virtual_piece_start_indices[piece_num][0] = j + 1
            self._piece_measures_indices[piece_num] = list(
                np.flatnonzero(
                    np.isin(piece_array[:, ATTR_NAME_INDEX['evt']], self._measure_ids)
                )
            )

            # create more virtual pieces from overlength pieces
            if virtual_piece_step_ratio > 0 and piece_array.shape[0] > max_seq_length:
                cur_vp_start_index = virtual_piece_step_size
                for measure_index in self._piece_measures_indices[piece_num]:
                    if measure_index > cur_vp_start_index:
                        self._virtual_piece_start_indices[piece_num].append(measure_index)
                        # increase cur_vp_start_index until it passes m_idx
                        increased_step = math.ceil(
                            (measure_index - cur_vp_start_index) / virtual_piece_step_size
                        )
                        cur_vp_start_index += increased_step * virtual_piece_step_size
                        # if this virtual piece is already shorter than max_seq_len
                        if measure_index > cur_piece_lengths - self.max_seq_length:
                            break

            if permute_mps:
                # consider sequence: M  P  N  N  N  P  N  N  P  N  M  P  N  N
                #             index: 0  1  2  3  4  5  6  7  8  9  10 11 12 13
                #   mps numbers are: 1  2  3  3  3  4  5  5  6  7  8  9  10 10
                # we want the beginning indices of all mps:
                #                   [0, 1, 2,       5, 6,    8, 9, 10,11,12]
                mps_numbers = piece_array[:, ATTR_NAME_INDEX['mps']]
                # just find where the number increases
                body_mps_sep_indices = np.where(mps_numbers[:-1] < mps_numbers[1:])[0]
                mps_sep_indices = np.concatenate([
                    body_mps_sep_indices,
                    np.array([piece_array.shape[0]]) # the EOS
                ])
                self._piece_mps_sep_indices[piece_num] = mps_sep_indices

            if self.pitch_augmentation_range != 0:
                has_pitch = (piece_array[:, ATTR_NAME_INDEX['pit']]) != 0
                # the pitch attribute of drums / percussion instrument is not actual note pitch
                # but different percussion sound
                # assume instrument vocabulary is 0:PAD, 1:0, 2:1, ... 128:127, 129:128(drums)
                not_percussion = (piece_array[:, ATTR_NAME_INDEX['ins']]) != 129
                augmentable_pitches = np.logical_and(has_pitch, not_percussion)
                self._piece_augmentable_pitches[piece_num] = augmentable_pitches

            if self.flatten_virtual_pieces:
                virtual_piece_number = len(self._virtual_piece_start_indices[piece_num])
                cur_index += virtual_piece_number * self._pitch_augmentation_factor
            else:
                cur_index += self._pitch_augmentation_factor
            self._piece_num_indices[piece_num] = cur_index
        self._max_index = cur_index if cur_index > 0 else 0

        if verbose:
            if permute_mps:
                mps_lengths = []
                for mps_sep_indices in self._piece_mps_sep_indices:
                    mps_lengths.extend([
                        j2 - j1
                        for j1, j2 in zip(mps_sep_indices[:-1], mps_sep_indices[1:])
                        if j2 > j1 + 1
                    ])
                print('Number of non-singleton mps:', len(mps_lengths))
                print('Average mps length:', sum(mps_lengths)/len(mps_lengths))
            self_size = sum(
                sys.getsizeof(getattr(self, attr_name))
                for attr_name in dir(self)
                if not attr_name.startswith('__')
            )
            print('Dataset object size:', self_size, 'bytes')

    def __getitem__(self, index):
        piece_num = bisect.bisect_left(self._piece_num_indices, index)
        body_start_index = self._piece_body_start_indices[piece_num]

        index_offset = (
            index
            if piece_num == 0 else
            index - self._piece_num_indices[piece_num-1] - 1
        )
        if self.flatten_virtual_pieces:
            virtual_piece_num = index_offset // self._pitch_augmentation_factor
            start_index = self._virtual_piece_start_indices[piece_num][virtual_piece_num]
            index_offset -= virtual_piece_num * self._pitch_augmentation_factor
        else:
            start_index = np.random.choice(self._virtual_piece_start_indices[piece_num])

        # preserve space for head
        end_index = start_index + self.max_seq_length - body_start_index
        if end_index < start_index:
            end_index = start_index
            body_start_index = self.max_seq_length

        # dont need to check end_index because if end_index > piece_length,
        # then numpy will just end the slice at piece_length
        head_array = np.array(self.pieces[piece_num][:body_start_index]) # copy
        body_array = np.array(self.pieces[piece_num][start_index:end_index]) # copy
        sampled_array = np.concatenate([head_array, body_array], axis=0)
        # was int16 to save space, but torch want int32
        sampled_array = sampled_array.astype(np.int32)

        # tidy up mps numbers
        # make sure all mps number smaller are than max_mps_number
        # and all of them are non-decreasing
        if body_start_index != self.max_seq_length:
            min_body_mps_number = np.min(
                sampled_array[body_start_index:, ATTR_NAME_INDEX['mps']]
            )
            # magic number 4 because the first measure must have mps number 4
            if min_body_mps_number < 4:
                raise ValueError('MPS number in body is less than 4.')
            sampled_array[body_start_index:, ATTR_NAME_INDEX['mps']] -= (min_body_mps_number - 4)

        # pitch augmentation
        # pitch vocabulary is 0:PAD, 1:0, 2:1, ... 128:127
        pitch_augment = index_offset - self.pitch_augmentation_range
        if pitch_augment != 0 and body_start_index != self.max_seq_length:
            augmentables = self._piece_augmentable_pitches[piece_num][body_start_index:end_index]
            # sometimes the sampled array does not contain any pitch-augmentable token
            body_pitch_col = sampled_array[body_start_index:, ATTR_NAME_INDEX['pit']]
            if np.any(augmentables):
                if pitch_augment > 0:
                    max_pitch = np.max((body_pitch_col)[augmentables])
                    if max_pitch + pitch_augment > 128:
                        pitch_augment -= max_pitch - 128
                else: # pitch_augment < 0
                    min_pitch = np.min((body_pitch_col)[augmentables])
                    if min_pitch + pitch_augment < 1:
                        pitch_augment += 1 - min_pitch
                body_pitch_col[augmentables] += pitch_augment

        if self.permute_track_number:
            max_track_number = self.vocabs.track_numbers.size - 1
            # add one because there is a padding token at the beginning of the vocab
            perm = np.concatenate(([0], np.random.permutation(max_track_number) + 1))
            # view track number col
            trn_column = sampled_array[:, ATTR_NAME_INDEX['trn']]
            for i, trn in enumerate(trn_column):
                trn_column[i] = perm[trn]

        if self.permute_mps and body_start_index != self.max_seq_length:
            mps_sep_indices_list = []
            mps_tokens_ranges = []
            # BOS, first track token and body_start_index-1 is SEP
            mps_sep_indices_list_head = [0, 1, body_start_index - 1]
            mps_sep_indices_list_body = [
                i - start_index + body_start_index
                for i in self._piece_mps_sep_indices[piece_num]
                if start_index <= i < end_index
            ]
            mps_sep_indices_list = mps_sep_indices_list_head + mps_sep_indices_list_body
            mps_sep_indices_and_end = mps_sep_indices_list + [sampled_array.shape[0]]
            for i in range(len(mps_sep_indices_and_end)-1):
                start = mps_sep_indices_and_end[i]
                end = mps_sep_indices_and_end[i+1]
                if end - start >= 2:
                    mps_tokens_ranges.append((start, end))

            # print(mps_tokens_ranges)
            for start, end in mps_tokens_ranges:
                p = np.random.permutation(end-start) + start
                sampled_array[start:end] = sampled_array[p]

        # checking for bad numbers
        for attr_name, attr_idx in ATTR_NAME_INDEX.items():
            if len(attr_name) == 3:
                continue
            assert np.min(sampled_array[:, attr_idx]) >= 0, \
                (f'Number in {attr_name} is below zero\n'
                    f'{get_full_array_string(sampled_array, self.vocabs)}')
            max_attr_value = np.max(sampled_array[:, attr_idx])
            if attr_name == 'mps_numbers':
                vocabs_size = min(self.max_seq_length, self.vocabs.max_mps_number) + 1
                assert max_attr_value < vocabs_size, \
                    (f'Number in {attr_name} larger than vocab size: '
                        f'{max_attr_value} >= {vocabs_size}\n'
                        f'{get_full_array_string(sampled_array, self.vocabs)}'
                    )
            else:
                vocabs_size = getattr(self.vocabs, attr_name).size
                assert max_attr_value < vocabs_size, \
                    (f'Number in {attr_name} not smaller than vocab size: '
                        f'{max_attr_value} >= {vocabs_size}\n'
                        f'{get_full_array_string(sampled_array, self.vocabs)}'
                    )

        return torch.from_numpy(sampled_array)

    def __len__(self):
        return self._max_index + 1


def collate_mididataset(seq_list: List[torch.Tensor]) -> torch.Tensor:
    """Stack batched sequences batch-first to a 3-d array and pad them"""
    # print('\n'.join(map(str, seq_list)))
    batched_seqs = pad_sequence(seq_list, batch_first=True, padding_value=0)
    return batched_seqs
