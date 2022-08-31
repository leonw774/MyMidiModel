from argparse import ArgumentParser, Namespace
import json

from torch import rsqrt
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam, lr_scheduler

from util import get_input_array_debug_string
from util.model import MidiTransformerDecoder
from util.dataset import MidiDataset, collate_mididataset
from util.corpus import to_vocabs_file_path
# from model.model import MidiTransformerDecoder


def parse_args():
    data_parser = ArgumentParser()
    data_parser.add_argument(
        '--max-seq-length',
        type=int,
        default=1024
    )
    data_parser.add_argument(
        '--sample-stride',
        type=int,
        default=1
    )
    data_parser.add_argument(
        '--use-set-loss',
        action='store_true'
    )
    data_parser.add_argument(
        '--permute-mps',
        action='store_true'
    )
    data_parser.add_argument(
        '--permute-track-number',
        action='store_true'
    )
    data_parser.add_argument(
        '--measure-number-shift-range',
        type=int,
        default=0
    )

    model_parser = ArgumentParser()
    model_parser.add_argument(
        '--layers-number',
        type=int,
        default=12
    )
    model_parser.add_argument(
        '--d-model',
        type=int,
        default=512
    )

    train_parser = ArgumentParser()
    train_parser.add_argument(
        '--split-ratio',
        type=float,
        nargs=2,
        default=[8, 2],
        help='The split ratio for training and validation. Default is 8:2.'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=8
    )
    train_parser.add_argument(
        '--steps',
        type=int,
        default=100000
    )
    train_parser.add_argument(
        '--validation-frequency',
        type=int,
        default=1000
    )

    train_parser.add_argument(
        '--learning-rate',
        type=float,
        default=5.0e-3
    )
    train_parser.add_argument(
        '--lr-warmup-steps',
        type=int,
        default=4000
    )
    train_parser.add_argument(
        '--lr-decay-end-ratio',
        type=float,
        default=0.1
    )
    train_parser.add_argument(
        '--lr-decay-end-steps',
        type=int,
        default=40000
    )

    train_parser.add_argument(
        '--early-stop',
        action='store_true'
    )
    train_parser.add_argument(
        '--ealry-stop-tolerance',
        type=int,
        default=10
    )

    global_parser = ArgumentParser()
    global_parser.add_argument(
        'corpus_dir_path',
        nargs='+' # store as list and at least one
    )
    global_parser.add_argument(
        '--log',
        dest='log_file_path',
        default='',
    )
    global_parser.add_argument(
        '--use-device',
        choices=['cpu', 'gpu'],
        default='cpu'
    )
    # make them into dicts
    global_args = vars(global_parser.parse_known_args()[0])
    global_args['data_args'] = data_parser.parse_known_args()[0]
    global_args['model_args'] = model_parser.parse_known_args()[0]
    global_args['train_args'] = train_parser.parse_known_args()[0]
    # turnback into Namespace
    return Namespace(**global_args)


# def vanilla_lr(step_num: int, warmup_steps: int, d_model: int) -> float:
#     return rsqrt(d_model) * min(rsqrt(step_num), step_num * warmup_steps ** (-1.5))

def lr_warmup_and_linear_decay(step_num: int, warmup_steps: int, decay_end_ratio: float, decay_end_steps: int):
    if step_num < warmup_steps:
        return (step_num + 1) / warmup_steps
    r = min(1, ((step_num - warmup_steps) / decay_end_steps))
    return 1 - r * (1 - decay_end_ratio)


def valid(model, validation_dataset, args):
    pass

def save_checkpoint(model):
    pass

def check_early_stopping(losses):
    pass

def train(model, complete_dataset, args):

    train_len, valid_len = (
        len(complete_dataset) * r / sum(args.train_args.split_ratio)
        for r in args.train_args.split_ratio
    )

    train_dataset, valid_dataset = random_split(complete_dataset, (train_len, valid_len))
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_args.batch_size,
        shuffle=True,
        collate_fn=collate_mididataset
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.train_args.batch_size,
        shuffle=False,
        collate_fn=collate_mididataset
    )
    train_dataloader_iter = iter(train_dataloader)
    valid_dataloader_iter = iter(valid_dataloader)

    optimizer = Adam(model.parameters(), args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_warmup_and_linear_decay(
            step,
            args.train_args.lr_warmup_steps,
            args.train_args.lr_decay_end_ratio,
            args.train_args.lr_decay_end_steps
        )
    )

    model.train()
    for step in range(args.train_args.steps):
        batched_input_array, batched_mps_sep_indices = next(train_dataloader_iter)


def main():
    args = parse_args()
    print(args)

    vocabs_dict = json.load(open(to_vocabs_file_path(args.corpus_dir_path), 'r', encoding='utf8'))
    model = MidiTransformerDecoder(
        vocabs=vocabs_dict,
        **vars(args.model_args)
    )

    complete_dataset = None
    if len(args.corpus_dir_path) == 1:
        complete_dataset = MidiDataset(
            data_dir_path=args.corpus_dir_path[0],
            **args.data_args
        )
    else:
        raise NotImplementedError('ConcatDataset for multiple corpus not implemented yet.')

    train(model, complete_dataset, args)

    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
