from argparse import ArgumentParser, Namespace
import json

import torch
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam, lr_scheduler

from util import get_input_array_debug_string
from util.model import MidiTransformerDecoder, get_seq_mask
from util.dataset import MidiDataset, collate_mididataset
from util.corpus import ATTR_NAME_TO_FEATURE_INDEX, to_vocabs_file_path
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
        '--attn-heads-number',
        type=int,
        default=8
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
        '--learning-rate', '--lr',
        dest='learning_rate',
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
        default=0.5
    )
    train_parser.add_argument(
        '--lr-decay-end-steps',
        type=int,
        default=12000
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
#     return torch.rsqrt(d_model) * min(torch.rsqrt(step_num), step_num * warmup_steps ** (-1.5))

def lr_warmup_and_linear_decay(step_num: int, warmup_steps: int, decay_end_ratio: float, decay_end_steps: int):
    if step_num < warmup_steps:
        return (step_num + 1) / warmup_steps
    r = min(1, ((step_num - warmup_steps) / decay_end_steps))
    return 1 - r * (1 - decay_end_ratio)


def set_loss():
    pass

def valid(model, validation_dataset, args):
    pass

def save_checkpoint(model):
    pass

def check_early_stopping(losses):
    pass

def train(model, train_dataloader, loss_func, args):
    model.train()
    train_dataloader_iter = iter(train_dataloader)

    for step in range(args.train_args.validation_frequency):
        try:
            batched_input_seqs, batched_mps_sep_indices = next(train_dataloader_iter)
        except StopIteration:
            train_dataloader_iter = iter(train_dataloader)
            batched_input_seqs, batched_mps_sep_indices = next(train_dataloader_iter)

        batchwed_seq_mask = get_seq_mask(batched_input_seqs.shape[1])
        print(batched_input_seqs.shape[1])
        print(batchwed_seq_mask.shape)
        prediction = model(batched_input_seqs, batchwed_seq_mask)

        if args.data_args.use_set_loss:
            raise NotImplementedError('Use_set_loss not implemented yet.')
        else:
            ground_truth = batched_input_seqs[:, :, args.output_attr_indices]
            loss = model.calc_loss(prediction, ground_truth)
            loss.backward()


def main():
    args = parse_args()
    print(args)

    vocabs_dict = json.load(open(to_vocabs_file_path(args.corpus_dir_path), 'r', encoding='utf8'))

    model = MidiTransformerDecoder(
        vocabs=vocabs_dict,
        **vars(args.model_args)
    )

    # make dataset
    complete_dataset = None
    if len(args.corpus_dir_path) == 1:
        complete_dataset = MidiDataset(
            data_dir_path=args.corpus_dir_path[0],
            **args.data_args
        )
    else:
        raise NotImplementedError('ConcatDataset for multiple corpus not implemented yet.')
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
        shuffle=True,
        collate_fn=collate_mididataset
    )
    output_attr_indices = [
        ATTR_NAME_TO_FEATURE_INDEX[attr_name]
        for attr_name in ['evt', 'pit', 'dur', 'vel', 'trn', 'ins', 'pos']
    ]
    if vocabs_dict['position_method'] == 'event':
        output_attr_indices.pop()
    args.output_attr_indices = output_attr_indices

    # make optimizer
    optimizer = Adam(model.parameters(), args.train_args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_warmup_and_linear_decay(
            step,
            args.train_args.lr_warmup_steps,
            args.train_args.lr_decay_end_ratio,
            args.train_args.lr_decay_end_steps
        )
    )

    # make loss function
    if args.train_args.use_set_loss:
        loss_func = set_loss()
    else:
        loss_func = torch.nn.CrossEntropyLoss()
    

    # train
    train_losses_history = []
    valid_losses_history = []
    early_stop_counter = 0
    for valid_epoch in range(args.train_args.steps // args.train_args.validation_frequency):
        train_losses = train(model, train_dataloader, args)
        valid_losses = valid(model, valid_dataloader, args)
        train_losses_history.append(train_losses)
        valid_losses_history.append(valid_losses)
        if args.train_args.early_stop:
            pass

    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
