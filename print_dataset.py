from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader

from util.midi import piece_to_midi
from util.corpus import get_full_array_string, array_to_text_list
from util.dataset import MidiDataset, collate_mididataset
# from util.model import MidiTransformerDecoder

parser = ArgumentParser()
parser.add_argument(
    '--max-seq-length',
    type=int,
    default=48,
    help='Default is %(default)s'
)
parser.add_argument(
    '--virtual-piece-step-ratio',
    type=float,
    default=0,
    help='Default is %(default)s'
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
    default=0,
    help='Default is %(default)s'
)
parser.add_argument(
    '--batch-size',
    type=int,
    default=4,
    help='Default is %(default)s'
)
parser.add_argument(
    '--output-midi',
    action='store_true'
)
parser.add_argument(
    '--seed',
    type=int,
    default=None,
    help='Default is %(default)s'
)
parser.add_argument(
    'data_dir_path',
    type=str
)
args = parser.parse_args()
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

data_args = {
    k: v
    for k, v in vars(args).items()
    if k not in {'batch_size', 'output_midi', 'seed'}
}
print(data_args)

dataset = MidiDataset(
    **data_args,
    verbose=True
)
vocabs = dataset.vocabs
print('max_seq_length:', dataset.max_seq_length)
print('permute_mps:', dataset.permute_mps)
print('permute_track_number:', dataset.permute_track_number)

train_len = int(len(dataset) * 0.8)
valid_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, valid_len])
print(len(dataset), len(train_set), len(val_set))

# print('RANDOM SPLIT')

# train_sample, train_mps_sep_indices = train_set[0]
# train0 = train_sample.numpy()
# print('TRAIN[0]')
# print(get_input_array_format_string(train0, train_mps_sep_indices, vocabs))

# val_sample, val_mps_sep_indices = val_set[0]
# val0 = val_sample.numpy()
# print('VAL[0]')
# print(get_input_array_format_string(val0, val_mps_sep_indices, vocabs))

print('FIRST BATCH OF TRAIN_DATALOADER')

train_dataloader = DataLoader(
    dataset=train_set,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_mididataset
)
batched_samples = next(iter(train_dataloader))
for i, s in enumerate(batched_samples):
    print(f'BATCHED[{i}]')
    print(get_full_array_string(s.numpy(), vocabs))
    piece = ' '.join(array_to_text_list(s.numpy(), vocabs))
    print(piece)
    if args.output_midi:
        if not piece.endswith('EOS'):
            piece += ' EOS'
        piece_to_midi(piece, vocabs.paras['nth']).dump(f'debug_dataset_batched{i}.mid')
