from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader

from util.midi import piece_to_midi
from util.corpus import get_input_array_format_string, array_to_text_list
from util.dataset import MidiDataset, collate_mididataset
# from util.model import MidiTransformerDecoder

parser = ArgumentParser()
parser.add_argument(
    '--max-seq-length',
    type=int,
    default=48
)
parser.add_argument(
    '--use-permutable-subseq-loss',
    action='store_true'
)
parser.add_argument(
    '--measure-sample-step-ratio',
    type=float,
    default=0.25
)
parser.add_argument(
    '--permute-mps',
    action='store_true'
)
parser.add_argument(
    '--permute-track-number',
    action='store_true'
)
parser.add_argument(
    '--pitch-augmentation-range',
    type=int,
    default=0
)
parser.add_argument(
    '--batch-size',
    type=int,
    default=4
)
parser.add_argument(
    '--output-midi',
    action='store_true'
)
parser.add_argument(
    '--seed',
    type=int,
    default=None
)
parser.add_argument(
    'corpus_dir_path',
    type=str,
    nargs='?',
    default='data/corpus/test_midis_nth32_r32_d32_v16_t24_200_16'
)
args = parser.parse_args()
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

dataset = MidiDataset(
    data_dir_path=args.corpus_dir_path,
    max_seq_length=args.max_seq_length,
    use_permutable_subseq_loss=args.use_permutable_subseq_loss,
    measure_sample_step_ratio=args.measure_sample_step_ratio,
    permute_mps=args.permute_mps,
    permute_track_number=args.permute_track_number,
    pitch_augmentation_range=args.pitch_augmentation_range,
    verbose=True
)
vocabs = dataset.vocabs
print('max_seq_length:', dataset.max_seq_length)
print('use_permutable_subseq_loss:', dataset.use_permutable_subseq_loss)
print('permute_mps:', dataset.permute_mps)
print('permute_track_number:', dataset.permute_track_number)

train_len = int(len(dataset) * 0.8)
valid_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, valid_len])
print(len(dataset), len(train_set), len(val_set))

# print('TEST RANDOM SPLIT')

# train_sample, train_mps_sep_indices = train_set[0]
# train0 = train_sample.numpy()
# print('TRAIN[0]')
# print(get_input_array_format_string(train0, train_mps_sep_indices, vocabs))

# val_sample, val_mps_sep_indices = val_set[0]
# val0 = val_sample.numpy()
# print('VAL[0]')
# print(get_input_array_format_string(val0, val_mps_sep_indices, vocabs))

print('TEST DATA CORRECTNESS')

train_dataloader = DataLoader(
    dataset=train_set,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_mididataset
)
batched_samples, batched_mps_sep_indices = next(iter(train_dataloader))
if len(batched_mps_sep_indices) == 0:
    batched_mps_sep_indices = [None] * batched_samples.shape[0]
for i, (s, e) in enumerate(zip(batched_samples, batched_mps_sep_indices)):
    print(f'BATCHED[{i}]')
    print(get_input_array_format_string(s.numpy(), e, vocabs))
    piece = ' '.join(array_to_text_list(s.numpy(), vocabs))
    print(piece)
    if args.output_midi:
        if not piece.endswith('EOS'):
            piece += ' EOS'
        piece_to_midi(piece, vocabs.paras['nth']).dump(f'debug_dataset_batched{i}.mid')
