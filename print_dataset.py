from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader

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
    default=1,
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
print('len(dataset):', len(dataset))
# print('included_path_list:', dataset.included_path_list)
# print('included_piece_num:', dataset.included_piece_num)
# print('_piece_lengths:', dataset._piece_lengths)
# print('_piece_mps_sep_indices:', dataset._piece_mps_sep_indices)
# print('_piece_measures_indices:', dataset._piece_measures_indices)
# print('_piece_body_start_indices:', dataset._piece_body_start_indices)
# print('_virtual_piece_start_indices:', dataset._virtual_piece_start_indices)


print('FIRST BATCH OF DATALOADER')
dataloader = DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_mididataset
)
batched_samples = next(iter(dataloader))
for i, s in enumerate(batched_samples):
    print(f'BATCHED[{i}]')
    print(get_full_array_string(s.numpy(), vocabs))
    piece = ' '.join(array_to_text_list(s.numpy(), vocabs))
    print(piece)
    if args.output_midi:
        if not piece.endswith('EOS'):
            piece += ' EOS'
        piece_to_midi(piece, vocabs.paras['tpq']).dump(
            f'print_dataset_batch{i}.mid'
        )
