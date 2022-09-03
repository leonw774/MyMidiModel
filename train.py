from argparse import ArgumentParser, Namespace
import json
import logging
import os
import shutil
from time import strftime, time

from tqdm import tqdm
import torch
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam, lr_scheduler

from util.model import MidiTransformerDecoder, get_seq_mask, calc_loss, calc_permutable_subseq_loss
from util.dataset import MidiDataset, collate_mididataset
from util.corpus import to_vocabs_file_path


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
        '--use-permutable-subseq-loss',
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
        '--validation-interval',
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
        "--grad-norm-clip",
        default=1.0,
        type=float
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
        nargs='+', # store as list and at least one
        type=str
    )
    global_parser.add_argument(
        '--checkpoint-dir-path',
        type=str,
        default=''
    )
    global_parser.add_argument(
        '--dataloader-worker-number',
        type=int,
        default=8
    )
    global_parser.add_argument(
        '--log',
        dest='log_file_path',
        default='',
    )
    global_parser.add_argument(
        '--loss-file-path',
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


def log(cur_step, start_time, scheduler, train_loss_list, valid_loss_list, loss_logger):
    avg_train_loss = sum(train_loss_list)/len(train_loss_list)
    avg_valid_loss = sum(valid_loss_list)/len(valid_loss_list)
    logging(
        f'Step: {cur_step}, Time: {time()-start_time}, '\
        f'Learning rate: {scheduler.get_last_lr()[0]}, '\
        f'Avg train loss: {avg_train_loss}, '\
        f'Avg valid loss: {avg_valid_loss}'
    )
    if loss_logger:
        loss_logger.info(f'{avg_train_loss},{avg_valid_loss}')


def save_checkpoint(cur_step, model, checkpoint_dir_path):
    model_file_name = os.path.join(checkpoint_dir_path, f'model_{cur_step}.pt')
    torch.save(model.state_dict(), model_file_name)
    logging.info(f"Saved model at: {model_file_name}")
    return model_file_name


def valid(model, valid_dataloader, args):
    model.eval()
    loss_list = []
    for batch_seqs, batch_mps_sep_indices in valid_dataloader:
        batch_input_seqs = batch_seqs[:, :-1]
        batch_target_seqs = model.to_output_features(batch_seqs[:, 1:])
        batch_input_seq_mask = get_seq_mask(batch_input_seqs.shape[1])
        prediction = model(batch_input_seqs, batch_input_seq_mask)

        if args.data_args.use_permutable_subseq_loss:
            loss = calc_permutable_subseq_loss(prediction, batch_target_seqs, batch_mps_sep_indices)
        else:
            loss = calc_loss(prediction, batch_target_seqs)
        loss_list.append(float(loss))
    return loss_list

def train(model, train_dataloader, optimizer, scheduler, args):
    model.train()
    train_dataloader_iter = iter(train_dataloader)
    loss_list = []
    for _ in tqdm(range(args.train_args.validation_interval)):
        try:
            batch_seqs, batch_mps_sep_indices = next(train_dataloader_iter)
        except StopIteration:
            train_dataloader_iter = iter(train_dataloader)
            batch_seqs, batch_mps_sep_indices = next(train_dataloader_iter)

        batch_input_seqs = batch_seqs[:, :-1]
        batch_target_seqs = model.to_output_features(batch_seqs[:, 1:])
        batch_input_seq_mask = get_seq_mask(batch_input_seqs.shape[1])
        prediction = model(batch_input_seqs, batch_input_seq_mask)

        if args.data_args.use_permutable_subseq_loss:
            loss = calc_permutable_subseq_loss(prediction, batch_target_seqs, batch_mps_sep_indices)
        else:
            loss = calc_loss(prediction, batch_target_seqs)
        loss_list.append(float(loss))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
        optimizer.step()
        scheduler.step()
    return loss_list


def main():
    args = parse_args()
    print(args)

    # logging setting
    # root logger
    loglevel = logging.INFO
    if args.log_file_path:
        logging.basicConfig(
            filename=args.log_file_path,
            filemode='a',
            level=loglevel,
            format='%(message)s',
        )
        console = logging.StreamHandler()
        console.setLevel(loglevel)
        logging.getLogger().addHandler(console)
    else:
        logging.basicConfig(
            level=loglevel,
            format='%(message)s'
        )
    logging.info(strftime('----train.py start at %Y%m%d-%H%M---'))

    # loss file logger
    loss_logger = None
    if args.loss_file_path:
        handler = logging.FileHandler(args.loss_file_path)
        handler.setFormatter('%(message)s')
        loss_logger = logging.getLogger('loss logger')
        loss_logger.setLevel(loglevel)
        loss_logger.addHandler(handler)
        logging.info('Set loss file: %s', args.loss_file_path)

    if not os.path.isdir(args.checkpoint_dir_path):
        logging.info('Invalid dir path: %s', args.checkpoint_dir_path)
        return 1

    # make model
    vocabs_dict = json.load(open(to_vocabs_file_path(args.corpus_dir_path), 'r', encoding='utf8'))
    model = MidiTransformerDecoder(
        vocabs=vocabs_dict,
        **vars(args.model_args)
    )

    # make dataset
    complete_dataset = None
    if len(args.corpus_dir_path) == 1:
        complete_dataset = MidiDataset(data_dir_path=args.corpus_dir_path[0], **args.data_args)
    else:
        raise NotImplementedError('ConcatDataset for multiple corpus not implemented yet.')
    train_len, valid_len = (
        len(complete_dataset) * r / sum(args.train_args.split_ratio)
        for r in args.train_args.split_ratio
    )
    train_dataset, valid_dataset = random_split(complete_dataset, (train_len, valid_len))
    train_dataloader = DataLoader(
        dataset=train_dataset,
        num_workers=args.dataloader_worker_number,
        batch_size=args.train_args.batch_size,
        shuffle=True,
        collate_fn=collate_mididataset
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        num_workers=args.dataloader_worker_number,
        batch_size=args.train_args.batch_size,
        shuffle=True,
        collate_fn=collate_mididataset
    )

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

    # train
    logging.info('Begin training')
    complete_train_loss_list = []
    complete_valid_loss_list = []
    min_avg_valid_loss = float('inf')
    early_stop_counter = 0
    start_time = time()
    for start_step in range(0, args.train_args.steps, args.train_args.validation_interval):
        train_loss_list = train(model, train_dataloader, optimizer, scheduler, args)
        valid_loss_list = valid(model, valid_dataloader, args)
        complete_train_loss_list.extend(train_loss_list)
        complete_valid_loss_list.extend(valid_loss_list)
        cur_step = start_step + args.train_args.validation_interval
        log(cur_step, start_time, scheduler, train_loss_list, valid_loss_list, loss_logger)
        model_file_name = save_checkpoint(cur_step, model, args.checkpoint_dir_path)
        if args.train_args.early_stop:
            avg_valid_loss = sum(valid_loss_list) / len(valid_loss_list)
            if min_avg_valid_loss >= avg_valid_loss:
                early_stop_counter += 1
                if early_stop_counter >= args.early_stop_tolerance:
                    logging.info('Early stopped at step %d for no improvement in %d validations.', cur_step, args.early_stop_tolerance)
                    break
            else:
                early_stop_counter = 0
                min_avg_valid_loss = avg_valid_loss
                shutil.copyfile(model_file_name, os.path.join(args.checkpoint_dir_path, 'model_best.pt'))
                logging('New best model.')

    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
