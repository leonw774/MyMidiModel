import bisect
import math
import os
import sys
from typing import List
import zipfile

import numpy as np
import psutil
from torch import from_numpy, Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from . import tokens
from .corpus import get_corpus_vocabs, to_pathlist_file_path, ATTR_NAME_INDEX, get_full_array_string

class MidiDataset(Dataset):

    def __init__(
            self,
            data_dir_path: str,
            max_seq_length: int,
            excluded_path_list: List[str] = None,
            virtual_piece_step_ratio: float = 0,
            permute_mps: bool = False,
            permute_track_number: bool = False,
            pitch_augmentation_range: int = 0,
            verbose: bool = False
        ) -> None:
        """
            Parameters:
            - `data_dir_path`: Expected a directory that have 'arrays.npz', 'pathlist' and 'vocabs.json'

            - `virtual_piece_step_ratio`: If > 1, will create multiple virtual pieces by splitting overlength pieces.
              The start point of samples will be at the first measure token that has index just greater than
              `max_seq_length` / `virtual_piece_step_ratio` * N, where N is positive integer. Default is 1.
              This value can not be smaller than 1.

            - `excluded_path_list`: The list of paths of the files to exclude.

            - `permute_mps`: Whether or not the dataset should permute all the maximal permutable subarrays
              before returning in `__getitem__`. Default is False.

            - `permute_track_number`: Permute all the track numbers relative to the instruments, as data augmentation.
              Default is False.

            - `pitch_augmentation_range`: If set to P, will add a random value from -P to +P (inclusive)
              on all pitch values as data augmentation. Default is 0.
        """

        self.vocabs = get_corpus_vocabs(data_dir_path)
        assert max_seq_length >= 0
        self.max_seq_length = max_seq_length
        if virtual_piece_step_ratio < 1:
            virtual_piece_step_ratio = 1
        self.virtual_piece_step_ratio = virtual_piece_step_ratio
        assert isinstance(permute_mps, bool)
        self.permute_mps = permute_mps
        assert isinstance(permute_track_number, bool)
        self.permute_track_number = permute_track_number
        assert pitch_augmentation_range >= 0
        self.pitch_augmentation_range = pitch_augmentation_range

        npz_path = os.path.join(data_dir_path, 'arrays.npz')
        all_paths_list = [p.strip() for p in open(to_pathlist_file_path(data_dir_path), 'r', encoding='utf8').readlines()]
        if excluded_path_list is not None and len(excluded_path_list) != 0:
            # assert os.path.exists(test_pathlist_file_path)
            excluded_paths_tuple = tuple(p.strip() for p in excluded_path_list)
        else:
            excluded_paths_tuple = tuple()
        if verbose:
            print('Reading', npz_path)
        self.used_filenames = []
        self.used_midi_paths = []
        for filenum, midi_filepath in tqdm(enumerate(all_paths_list), desc='Exclude paths', disable=not verbose):
            # use endswith because the pathlist records the relative paths to midi files from project's root
            # while excluded_paths_tuple record relative paths to midi files from dataset's root
            if not midi_filepath.endswith(excluded_paths_tuple):
                self.used_filenames.append(str(filenum))
                self.used_midi_paths.append(midi_filepath)

        available_memory_size = psutil.virtual_memory().available
        npz_zipinfo_list = zipfile.ZipFile(npz_path).infolist()
        array_memory_size = sum([zinfo.file_size for zinfo in npz_zipinfo_list])
        if array_memory_size >= available_memory_size - 1e9:
            if verbose:
                print(f'Memory not enough. Need {array_memory_size//1024} KB. Only {available_memory_size//1024} KB available.')
            raise OSError('Memory not enough.')
        # load into memory to be faster
        if verbose:
            print('Loading arrays to memory')
        npz_file = np.load(npz_path)
        self.pieces = {
            str(filenum): npz_file[str(filenum)]
            for filenum in tqdm(self.used_filenames, disable=not verbose, ncols=0)
        }
        self.number_of_pieces = len(self.pieces)

        if verbose:
            print('Processing')
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
        self._filenum_indices = [0] * self.number_of_pieces
        self._piece_lengths = [0] * self.number_of_pieces

        # the default zeros will be replaced by body start index
        self._piece_body_start_index = [0] * self.number_of_pieces
        virtual_piece_step = max(1, int(max_seq_length / virtual_piece_step_ratio))
        self._virtual_piece_start_index = [[0] for _ in range(self.number_of_pieces)]

        self._file_mps_sep_indices = [[] for _ in range(self.number_of_pieces)]

        self._augmentable_pitches = [np.empty((0,), dtype=np.bool8) for _ in range(self.number_of_pieces)]
        self._pitch_augmentation_factor = self.pitch_augmentation_range * 2 + 1 # length of (-pa, ..., -1, 0, 1, ..., pa)

        cur_index = -1 # <-- !!! because the first element we add should become index 0
        for filenum, filename in tqdm(enumerate(self.pieces.keys()), disable=not verbose, ncols=0):

            cur_piece_lengths = int(self.pieces[filename].shape[0])
            self._piece_lengths[filenum] = cur_piece_lengths

            j = 2 # because first token is BOS and second must be track as we excluded all empty midis
            while self.pieces[filename][j][ATTR_NAME_INDEX['evt']] != self._sep_id:
                j += 1
            self._piece_body_start_index[filenum] = j + 1
            self._virtual_piece_start_index[filenum][0] = j + 1

            # create virtual pieces from overlength pieces
            if virtual_piece_step_ratio > 0:
                if self.pieces[filename].shape[0] > max_seq_length:
                    measure_indices = list(
                        np.flatnonzero(
                            np.isin(self.pieces[filename][:, ATTR_NAME_INDEX['evt']], self._measure_ids)
                        )
                    )
                    vp_step_num = 1
                    for m_idx in measure_indices:
                        # the virtual piece can not be shorter than virtual_piece_step
                        if m_idx > cur_piece_lengths - virtual_piece_step:
                            break
                        if m_idx > virtual_piece_step * vp_step_num:
                            self._virtual_piece_start_index[filenum].append(m_idx)
                            vp_step_num += math.ceil((m_idx - (virtual_piece_step * vp_step_num)) / vp_step_num)

            if permute_mps:
                # consider sequence: M  P  N  N  N  P  N  N  P  N  M  P  N  N
                #             index: 0  1  2  3  4  5  6  7  8  9  10 11 12 13
                #   mps numbers are: 1  2  3  3  3  4  5  5  6  7  8  9  10 10
                # we want the beginning indices of all mps:
                #                   [0, 1, 2,       5, 6,    8, 9, 10,11,12]
                mps_numbers = self.pieces[filename][:, ATTR_NAME_INDEX['mps']]
                # just find where the number increases
                body_mps_sep_indices = np.where(mps_numbers[:-1] < mps_numbers[1:])[0]
                mps_sep_indices = np.concatenate([
                    # np.arange(j + 1), # the head # actually don't need this because it is handled in __getitem__
                    body_mps_sep_indices,
                    np.array([self.pieces[filename].shape[0]]) # the EOS
                ])
                self._file_mps_sep_indices[filenum] = mps_sep_indices

            if self.pitch_augmentation_range != 0:
                is_multinote_token = (self.pieces[filename][:, ATTR_NAME_INDEX['pit']]) != 0
                # the pitch attribute of drums / percussion instrument does not means actual note pitch
                # but different percussion sound
                # assume instrument vocabulary is 0:PAD, 1:0, 2:1, ... 128:127, 129:128(drums)
                not_percussion_notes = (self.pieces[filename][:, ATTR_NAME_INDEX['ins']]) != 129
                self._augmentable_pitches[filenum] = np.logical_and(is_multinote_token, not_percussion_notes)

            cur_index += len(self._virtual_piece_start_index[filenum]) * self._pitch_augmentation_factor
            self._filenum_indices[filenum] = cur_index
        self._max_index = cur_index if cur_index > 0 else 0

        if verbose:
            if permute_mps:
                mps_lengths = []
                for mps_sep_indices in self._file_mps_sep_indices:
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

        filenum = bisect.bisect_left(self._filenum_indices, index)
        body_start_index = self._piece_body_start_index[filenum]

        index_offset = index if filenum == 0 else index - self._filenum_indices[filenum-1] - 1
        virtual_piece_number = index_offset // self._pitch_augmentation_factor
        pitch_augment = index_offset - virtual_piece_number * self._pitch_augmentation_factor - self.pitch_augmentation_range

        start_index = self._virtual_piece_start_index[filenum][virtual_piece_number]
        end_index = start_index + self.max_seq_length - body_start_index # preserve space for head
        if end_index < start_index:
            end_index = start_index
            body_start_index = self.max_seq_length
        # dont need to check end_index because if end_index > piece_length than numpy will just end the slice at piece_length
        head_array = np.array(self.pieces[self.used_filenames[filenum]][:body_start_index]) # copy
        body_array = np.array(self.pieces[self.used_filenames[filenum]][start_index:end_index]) # copy
        sampled_array = np.concatenate([head_array, body_array], axis=0)
        sampled_array = sampled_array.astype(np.int32) # was int16 to save space but torch ask for int

        # to make sure mps number is smaller than max_mps_number
        # end_with_eos = sampled_array[-1, ATTR_NAME_INDEX['evt']] == self._eos_id
        # body_end_index = -1 if end_with_eos else sampled_array.shape[0]
        if body_start_index != self.max_seq_length:
            try:
                min_mps_number = np.min(sampled_array[body_start_index:, ATTR_NAME_INDEX['mps']])
            except Exception as e:
                print(sampled_array)
                raise e
            # magic number 4 because the first measure must have mps number 4
            if min_mps_number != 4:
                sampled_array[body_start_index:, ATTR_NAME_INDEX['mps']] -= (min_mps_number - 4)

        # pitch augmentation
        # pitch vocabulary is 0:PAD, 1:0, 2:1, ... 128:127
        if pitch_augment != 0 and body_start_index != self.max_seq_length:
            sampled_augmentable_pitches = self._augmentable_pitches[filenum][start_index:end_index]
            # sometimes the sampled array does not contain any pitch-augmentable token
            sampled_body_pitch_col = sampled_array[body_start_index:, ATTR_NAME_INDEX['pit']]
            if np.any(sampled_augmentable_pitches):
                if pitch_augment > 0:
                    max_pitch = np.max((sampled_body_pitch_col)[sampled_augmentable_pitches])
                    if max_pitch + pitch_augment > 128:
                        pitch_augment -= max_pitch - 128
                else: # pitch_augment < 0
                    min_pitch = np.min((sampled_body_pitch_col)[sampled_augmentable_pitches])
                    if min_pitch + pitch_augment < 1:
                        pitch_augment += 1 - min_pitch
                sampled_body_pitch_col[sampled_augmentable_pitches] += pitch_augment

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
                for i in self._file_mps_sep_indices[filenum]
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
        for attr_name, fidx in ATTR_NAME_INDEX.items():
            if len(attr_name) > 3: # not abbr
                assert np.min(sampled_array[:, fidx]) >= 0, \
                    (f'Number in {attr_name} is below zero\n'
                     f'{get_full_array_string(sampled_array, self.vocabs)}')
                if attr_name == 'mps_numbers':
                    assert np.max(sampled_array[:, fidx]) <= min(self.max_seq_length, self.vocabs.max_mps_number), \
                        (f'Number in {attr_name} larger than vocab size: '
                         f'{np.max(sampled_array[:, fidx])} >= {min(self.max_seq_length, self.vocabs.max_mps_number)}\n'
                         f'{get_full_array_string(sampled_array, self.vocabs)}')
                else:
                    assert np.max(sampled_array[:, fidx]) < getattr(self.vocabs, attr_name).size, \
                        (f'Number in {attr_name} not smaller than vocab size: '
                         f'{np.max(sampled_array[:, fidx])} >= {getattr(self.vocabs, attr_name).size}\n'
                         f'{get_full_array_string(sampled_array, self.vocabs)}')

        return from_numpy(sampled_array)


    def __len__(self):
        return self._max_index


def collate_mididataset(seq_list: List[Tensor]) -> Tensor:
    """Stack batched sequences with batch-first to becom 3-d array and pad them"""
    # print('\n'.join(map(str, seq_list)))
    batched_seqs = pad_sequence(seq_list, batch_first=True, padding_value=0)
    return batched_seqs
