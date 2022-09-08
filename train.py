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
import torchinfo

from util.model import MidiTransformerDecoder, get_seq_mask, calc_losses, calc_permutable_subseq_losses
from util.dataset import MidiDataset, collate_mididataset
from util.corpus import text_list_to_array, to_vocabs_file_path
from util.midi import piece_to_midi
from util.tokens import BEGIN_TOKEN_STR


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
        '--embedding-dim',
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
        "--grad-norm-clip",
        default=1.0,
        type=float
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
        '--lr-decay-end-steps',
        type=int,
        default=16000
    )
    train_parser.add_argument(
        '--lr-decay-end-ratio',
        type=float,
        default=0.3 # approx. 1 - sqrt(2)^-1
    )
    train_parser.add_argument(
        '--early-stop',
        type=int,
        default=10,
        help='If this value <= 0, no early stoping will perform.'
    )

    global_parser = ArgumentParser()
    global_parser.add_argument(
        '--dataloader-worker-number',
        type=int,
        default=1 # npz file cannot handle multiprocessing, sad
    )
    global_parser.add_argument(
        '--use-device',
        choices=['cpu', 'cuda'],
        default='cpu'
    )
    global_parser.add_argument(
        '--log',
        dest='log_file_path',
        default='',
    )
    global_parser.add_argument(
        '--checkpoint-dir-path',
        type=str,
        required=True
    )
    global_parser.add_argument(
        '--log-head-losses',
        action='store_true'
    )
    global_parser.add_argument(
        'corpus_dir_path',
        type=str
    )
    # make them as dicts first
    global_args = dict()
    global_args['data_args'], others = data_parser.parse_known_args()
    # print(others)
    global_args['model_args'], others = model_parser.parse_known_args(others)
    # print(others)
    global_args['train_args'], others = train_parser.parse_known_args(others)
    # print(others)
    global_args.update(vars(global_parser.parse_known_args(others)[0]))
    # then turn into Namespace
    return Namespace(**global_args)


# def vanilla_lr(step_num: int, warmup_steps: int, d_model: int) -> float:
#     return torch.rsqrt(d_model) * min(torch.rsqrt(step_num), step_num * warmup_steps ** (-1.5))

def lr_warmup_and_linear_decay(step_num: int, warmup_steps: int, decay_end_ratio: float, decay_end_steps: int):
    if step_num < warmup_steps:
        return (step_num + 1) / warmup_steps
    r = min(1, ((step_num - warmup_steps) / decay_end_steps))
    return 1 - r * (1 - decay_end_ratio)


def log(
        cur_step:int,
        start_time: int,
        scheduler: lr_scheduler,
        train_loss_list: list,
        valid_loss_list: list,
        loss_logger):
    avg_train_losses = [ sum(head_loss_tuple) for head_loss_tuple in zip(*train_loss_list) ]
    avg_valid_losses = [ sum(head_loss_tuple)for head_loss_tuple in zip(*valid_loss_list) ]
    avg_train_losses_str = ', '.join([f'{l:.4f}' for l in avg_train_losses])
    avg_valid_losses_str = ', '.join([f'{l:.4f}' for l in avg_valid_losses])
    logging.info(
        'Step: %d, Time: %d, Learning rate: %.6f',
        cur_step, time()-start_time, scheduler.get_last_lr()[0]
    )
    logging.info(
        'Avg. train losses: %s, Avg. accu. train loss: %.4f, Avg. valid losses: %s Avg. accu. valid loss: %.4f', 
        avg_train_losses_str, sum(avg_train_losses), avg_valid_losses_str, sum(avg_valid_losses)
    )
    if loss_logger:
        loss_logger.debug('%s,%s', avg_train_losses_str, avg_valid_losses_str)


def save_checkpoint(cur_step, model, checkpoint_dir_path):
    model_file_name = os.path.join(checkpoint_dir_path, f'model_{cur_step}.pt')
    torch.save(model.state_dict(), model_file_name)
    logging.info("Saved model at: %s", model_file_name)
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
            head_losses = calc_permutable_subseq_losses(prediction, batch_target_seqs, batch_mps_sep_indices)
        else:
            head_losses = calc_losses(prediction, batch_target_seqs)
        loss_list.append([float(hl) for hl in head_losses])
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
            head_losses = calc_permutable_subseq_losses(prediction, batch_target_seqs, batch_mps_sep_indices)
        else:
            head_losses = calc_losses(prediction, batch_target_seqs)
        loss_list.append([float(hl) for hl in head_losses])
        loss = sum(head_losses)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.train_args.grad_norm_clip)
        optimizer.step()
        scheduler.step()
    return loss_list


def main():
    args = parse_args()

    if not os.path.isdir(args.checkpoint_dir_path):
        logging.info('Invalid checkpoint dir path: %s', args.checkpoint_dir_path)
        return 1

    # logging setting
    # root logger
    if args.log_file_path:
        logging.basicConfig(
            filename=args.log_file_path,
            filemode='a',
            level=logging.INFO,
            format='%(message)s',
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
    logging.info(strftime('----train.py start at %Y%m%d-%H%M---'))

    data_args_str = '\n'.join([f'{k}:{v}' for k, v in vars(args.data_args).items()])
    model_args_str = '\n'.join([f'{k}:{v}' for k, v in vars(args.model_args).items()])
    train_args_str = '\n'.join([f'{k}:{v}' for k, v in vars(args.train_args).items()])
    args_str = '\n'.join([f'{k}:{v}' for k, v in vars(args).items() if not isinstance(v, Namespace)])
    logging.info(data_args_str)
    logging.info(model_args_str)
    logging.info(train_args_str)
    logging.info(args_str)

    # loss logger
    # loss file is in the checkpoint directory
    loss_file_path = os.path.join(args.checkpoint_dir_path, 'loss.csv')
    loss_handler = logging.FileHandler(loss_file_path, 'w+', encoding='utf8')
    loss_formatter = logging.Formatter('%(message)s')
    loss_handler.setFormatter(loss_formatter)
    loss_logger = logging.getLogger('loss_logger')
    loss_logger.setLevel(logging.INFO)
    loss_logger.addHandler(loss_handler)
    logging.info('Created loss.csv file at %s', loss_file_path)

    # make model
    vocabs_dict = json.load(open(to_vocabs_file_path(args.corpus_dir_path), 'r', encoding='utf8'))
    model = MidiTransformerDecoder(
        vocabs=vocabs_dict,
        max_seq_length=args.data_args.max_seq_length,
        **vars(args.model_args)
    )
    torchinfo.summary(
        model,
        input_size=[
            (args.train_args.batch_size, args.data_args.max_seq_length, len(model.input_features_name)), # x
            (args.data_args.max_seq_length, args.data_args.max_seq_length) # mask
        ],
        dtypes=[torch.long, torch.float64]
    )

    loss_csv_head = ', '.join(
        ['train_' + n for n in model.output_features_name]
            + ['train_sum']
            + ['valid_' + n for n in model.output_features_name]
            + ['valid_sum']
    )
    loss_logger.debug(loss_csv_head)

    # make dataset
    complete_dataset = MidiDataset(data_dir_path=args.corpus_dir_path, **vars(args.data_args))
    train_len = int(len(complete_dataset) * args.train_args.split_ratio[0] / sum(args.train_args.split_ratio))
    valid_len = len(complete_dataset) - train_len
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
    # was int16 to save space but torch ask for long
    uncond_gen_prompt = torch.from_numpy(text_list_to_array([BEGIN_TOKEN_STR], vocabs_dict)).unsqueeze(0).long()
    for start_step in range(0, args.train_args.steps, args.train_args.validation_interval):
        train_loss_list = train(model, train_dataloader, optimizer, scheduler, args)
        valid_loss_list = valid(model, valid_dataloader, args)

        complete_train_loss_list.extend(train_loss_list)
        complete_valid_loss_list.extend(valid_loss_list)
        cur_step = start_step + args.train_args.validation_interval
        log(cur_step, start_time, scheduler, train_loss_list, valid_loss_list, loss_logger)

        model_file_name = save_checkpoint(cur_step, model, args.checkpoint_dir_path)

        print('Generating unconditional generation sample for checkpoint')
        uncond_gen_text_list = model.generate(uncond_gen_prompt, args.data_args.max_seq_length)
        uncond_gen_piece = ' '.join(uncond_gen_text_list)
        open(os.path.join(args.checkpoint_dir_path, f'uncond_gen_{cur_step}'), 'w+', encoding='utf8').write(uncond_gen_piece)
        piece_to_midi(uncond_gen_piece, vocabs_dict['paras']['nth']).dump(
            os.path.join(args.checkpoint_dir_path, f'uncond_gen_{cur_step}.mid')
        )

        if args.train_args.early_stop > 0:
            avg_valid_loss = sum([sum(valid_head_losses) for valid_head_losses in valid_loss_list]) / len(valid_loss_list)
            if avg_valid_loss >= min_avg_valid_loss:
                early_stop_counter += 1
                if early_stop_counter >= args.train_args.early_stop:
                    logging.info(
                        'Early stopped @ step %d: No improvement for %d validations.',
                        cur_step, args.train_args.early_stop
                    )
                    break
            else:
                early_stop_counter = 0
                min_avg_valid_loss = avg_valid_loss
                shutil.copyfile(model_file_name, os.path.join(args.checkpoint_dir_path, 'model_best.pt'))
                logging.info('New best model.')

    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
