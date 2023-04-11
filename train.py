from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
import glob
import logging
import os
import shutil
from time import strftime, time
from traceback import format_exc
from typing import List

import accelerate
import numpy as np

from tqdm import tqdm
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import random_split, DataLoader
from torch.optim import AdamW, lr_scheduler
# from torch.profiler import profile, record_function, ProfilerActivity
import torchinfo

from util.midi import piece_to_midi, get_first_k_measures
from util.corpus import COMPLETE_ATTR_NAME, OUTPUT_ATTR_NAME, get_corpus_vocabs, array_to_text_list, text_list_to_array
from util.dataset import MidiDataset, collate_mididataset
from util.model import (
    MyMidiTransformer,
    generate_sample,
    compute_losses,
    compute_permutable_subseq_losses
)

def parse_args():
    data_parser = ArgumentParser()
    data_parser.add_argument(
        '--max-seq-length',
        type=int
    )
    data_parser.add_argument(
        '--use-permutable-subseq-loss',
        action='store_true'
    )
    data_parser.add_argument(
        '--measure-sample-step-ratio',
        type=float,
        default=0.25
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
        '--pitch-augmentation-range',
        type=int,
        default=0
    )

    model_parser = ArgumentParser()
    model_parser.add_argument(
        '--use-linear-attn',
        action='store_true'
    )
    model_parser.add_argument(
        '--layers-number',
        type=int
    )
    model_parser.add_argument(
        '--attn-heads-number',
        type=int
    )
    model_parser.add_argument(
        '--embedding-dim',
        type=int
    )
    model_parser.add_argument(
        '--input-context',
        action='store_true',
    )
    model_parser.add_argument(
        '--input-instruments',
        action='store_true',
    )
    model_parser.add_argument(
        '--output-instruments',
        action='store_true',
    )

    train_parser = ArgumentParser()
    train_parser.add_argument(
        '--split-ratio',
        type=float,
        nargs=2,
        default=[9, 1],
        help='The split ratio for training and validation. \
            If one is set to -1 and the other N, for exmaple (-1, N) it means (len(complete_dataset) - N, N) \
            Default is %(default)s.'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=8
    )
    train_parser.add_argument(
        '--max-updates',
        type=int
    )
    train_parser.add_argument(
        '--validation-interval',
        type=int,
    )
    train_parser.add_argument(
        '--generate-sample-interval',
        type=int
    )
    train_parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help='Set the max_norm of nn.util.clip_grad_norm_(). \
            If this value is zero, gradient clipping will not be used. \
            Default is %(desult)s.'
    )
    train_parser.add_argument(
        '--lr-peak',
        type=float
    )
    train_parser.add_argument(
        '--lr-warmup-updates',
        type=int
    )
    train_parser.add_argument(
        '--lr-decay-end-updates',
        type=int
    )
    train_parser.add_argument(
        '--lr-decay-end-ratio',
        type=float
    )
    train_parser.add_argument(
        '--early-stop',
        type=int,
        help='If this value <= 0, no early stoping will perform.'
    )

    global_parser = ArgumentParser()
    global_parser.add_argument(
        '--dataloader-worker-number',
        type=int,
        default=4
    )
    global_parser.add_argument(
        '--use-device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda'
    )
    global_parser.add_argument(
        '--use-parallel',
        action='store_true'
    )
    global_parser.add_argument(
        '--max-pieces-per-gpu',
        type=int,
        default=256,
        help='Set this to reasonable value to prevent OOM.'
    )
    global_parser.add_argument(
        '--log',
        dest='log_file_path',
        type=str,
        default='',
    )
    global_parser.add_argument(
        '--nucleus-sampling-threshold', '--nu',
        type=float,
        default=1.0,
        help='The probability threshold nuclues sampling. Default is %(default)s.'
    )
    global_parser.add_argument(
        '--primer-measure-length',
        type=int,
        default=4
    )
    global_parser.add_argument(
        '--seed',
        type=int,
        default=413
    )
    global_parser.add_argument(
        'corpus_dir_path',
        type=str
    )
    global_parser.add_argument(
        'model_dir_path',
        type=str
    )
    # make them as dicts first
    global_args = dict()
    global_args['data'], others = data_parser.parse_known_args()
    # print(others)
    global_args['model'], others = model_parser.parse_known_args(others)
    # print(others)
    global_args['train'], others = train_parser.parse_known_args(others)
    # print(others)
    global_args.update(
        vars(global_parser.parse_known_args(others)[0])
    )
    # then turn into Namespace
    return Namespace(**global_args)


# def vanilla_lr(step_num: int, warmup_steps: int, d_model: int) -> float:
#     return torch.rsqrt(d_model) * min(torch.rsqrt(step_num), step_num * warmup_steps ** (-1.5))

def lr_warmup_and_linear_decay(step_num: int, warmup_steps: int, decay_end_ratio: float, decay_end_steps: int):
    if step_num < warmup_steps:
        return (step_num + 1) / warmup_steps
    r = min(1, ((step_num - warmup_steps) / decay_end_steps))
    return 1 - r * (1 - decay_end_ratio)


def log_losses(
        cur_num_updates: int,
        train_loss_list: List[List[float]],
        valid_loss_list: List[List[float]],
        loss_file_path: str):
    avg_train_losses = [ sum(head_loss_tuple) / len(head_loss_tuple) for head_loss_tuple in zip(*train_loss_list) ]
    avg_valid_losses = [ sum(head_loss_tuple) / len(head_loss_tuple) for head_loss_tuple in zip(*valid_loss_list) ]
    sum_avg_train_losses = sum(avg_train_losses)
    sum_avg_valid_losses = sum(avg_valid_losses)
    logging.info(
        'Avg. train head losses: %s Avg. train loss: %.6f \nAvg. valid head losses: %s Avg. valid loss: %.6f',
        ', '.join([f'{l:.3f}' for l in avg_train_losses]), sum_avg_train_losses,
        ', '.join([f'{l:.3f}' for l in avg_valid_losses]), sum_avg_valid_losses
    )
    if loss_file_path:
        valid_len = len(train_loss_list)
        with open(loss_file_path, 'a', encoding='utf8') as loss_file:
            for i, train_head_losses in enumerate(train_loss_list):
                idx = cur_num_updates - valid_len + i + 1 # count from 1
                if idx != cur_num_updates:
                    loss_file.write(
                        f'{idx},'
                        + ','.join([f'{l:.3f}' for l in train_head_losses]) + ',' # train head losses
                        + f'{sum(train_head_losses):.6f},' # train loss (sum)
                        # NO valid head losses and sum
                        + '\n'
                    )
                else:
                    loss_file.write(
                        f'{idx},'
                        + ','.join([f'{l:.3f}' for l in train_head_losses]) + ',' # train head losses
                        + f'{sum(train_head_losses):.6f},' # train loss (sum)
                        + ','.join([f'{l:.3f}' for l in avg_valid_losses]) + ',' # avg valid head losses
                        + f'{sum_avg_valid_losses:.6f}' # avg valid loss (sum)
                        + '\n'
                    )


def main():

    ######## Check args and print

    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        args.use_device = 'cpu'
        args.use_parallel = False
    args.use_device = torch.device(args.use_device)

    parallel_devices_count = len(os.getenv('CUDA_VISIBLE_DEVICES').split(',')) if args.use_parallel else 1
    if args.use_parallel and parallel_devices_count == 1:
        args.use_parallel = False

    gradient_accumulation_steps = int(args.train.batch_size / (args.max_pieces_per_gpu * parallel_devices_count))
    if gradient_accumulation_steps > 1:
        args.train.batch_size = args.max_pieces_per_gpu * parallel_devices_count
    if gradient_accumulation_steps == 0:
        gradient_accumulation_steps = 1

    if args.use_parallel:
        accelerator = accelerate.Accelerator(
            # `split_batches=True indicate that the batch size will always stay the same
            # no matter how many number of GPUs you run your script on
            split_batches=True, 
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        is_main_process = accelerator.is_main_process
    else:
        accelerator = None
        is_main_process = True

    # root logger
    if args.log_file_path != '':
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

    ckpt_dir_path = os.path.join(args.model_dir_path, 'ckpt')
    if is_main_process:
        logging.info(strftime('==== train.py start at %Y%m%d-%H%M%S ===='))
        if not os.path.isdir(args.model_dir_path):
            logging.info('Invalid model dir path: %s', args.model_dir_path)
            return 1
        if not os.path.isdir(ckpt_dir_path):
            logging.info('Invalid model ckpt dir path: %s', ckpt_dir_path)
            return 1

        data_args_str = '\n'.join([f'{k}:{v}' for k, v in vars(args.data).items()])
        model_args_str = '\n'.join([f'{k}:{v}' for k, v in vars(args.model).items()])
        train_args_str = '\n'.join([f'{k}:{v}' for k, v in vars(args.train).items()])
        args_str = '\n'.join([f'{k}:{v}' for k, v in vars(args).items() if not isinstance(v, Namespace)])
        logging.info(data_args_str)
        logging.info(model_args_str)
        logging.info(train_args_str)
        logging.info('gradient_accumulation_steps:%d', gradient_accumulation_steps)
        logging.info(args_str)

    ######## Prepare loss.csv

    vocabs = get_corpus_vocabs(args.corpus_dir_path)

    # loss csv file is in the root of model directory
    loss_file_path = os.path.join(args.model_dir_path, 'loss.csv')
    with open(loss_file_path, 'w+', encoding='utf8') as loss_file:
        loss_csv_head = 'step,'
        train_output_attr_name = ['train_' + n for n in OUTPUT_ATTR_NAME]
        valid_output_attr_name = ['valid_' + n for n in OUTPUT_ATTR_NAME]
        if not args.model.output_instruments:
            # remove instruments from output attribute names
            train_output_attr_name = train_output_attr_name[:-1]
            valid_output_attr_name = valid_output_attr_name[:-1]
        loss_csv_head += ','.join(
            train_output_attr_name + ['train_sum']
            + valid_output_attr_name + ['valid_sum']
        )
        loss_file.write(loss_csv_head+'\n')
    if is_main_process:
        logging.info('Created loss.csv file at %s', loss_file_path)

    if args.use_device.type == 'cuda':
        if is_main_process and not args.use_parallel:
            logging.info(
                'Torch sees %d CUDA devices. Current device is #%d',
                torch.cuda.device_count(), torch.cuda.current_device()
            )

    ######## Make dataset

    complete_dataset = MidiDataset(data_dir_path=args.corpus_dir_path, **vars(args.data), verbose=is_main_process)
    if is_main_process:
        logging.info('Made MidiDataset')
    train_ratio, valid_ratio = args.train.split_ratio
    if train_ratio == valid_ratio == -1:
        raise ValueError('split_ratio (-1, -1) is not allowed')
    else:
        if train_ratio == -1:
            train_ratio = len(complete_dataset) - valid_ratio
        if valid_ratio == -1:
            valid_ratio = len(complete_dataset) - train_ratio
    train_len = int(len(complete_dataset) * train_ratio / (train_ratio + valid_ratio))
    valid_len = len(complete_dataset) - train_len
    train_dataset, valid_dataset = random_split(complete_dataset, (train_len, valid_len))
    if is_main_process:
        logging.info('Size of training set: %d', len(train_dataset))
        logging.info('Size of validation set: %d', len(valid_dataset))

    ######## Get random conditional primer from complete_dataset

    while True:
        try:
            cond_primer_index = np.random.randint(len(complete_dataset.pieces))
            cond_primer_array = complete_dataset.pieces[str(cond_primer_index)]
            cond_primer_text_list = array_to_text_list(cond_primer_array, vocabs)
            cond_primer_text_list = get_first_k_measures(cond_primer_text_list, args.primer_measure_length)
            cond_primer_array = text_list_to_array(cond_primer_text_list, vocabs).astype(np.int32)
            break
        except Exception:
            continue
    cond_primer_array = torch.from_numpy(np.expand_dims(cond_primer_array, axis=0))
    if is_main_process:
        logging.info('Conditional generation primer is #%d', cond_primer_index)

    ######## Make dataloader

    # cannot handle multiprocessing if use npz mmap
    if not isinstance(complete_dataset.pieces, dict):
        args.dataloader_worker_number = 1
    train_dataloader = DataLoader(
        dataset=train_dataset,
        num_workers=args.dataloader_worker_number,
        batch_size=args.train.batch_size // parallel_devices_count,
        shuffle=False, # no need to shuffle because random_split did
        collate_fn=collate_mididataset
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        num_workers=args.dataloader_worker_number,
        batch_size=args.train.batch_size // parallel_devices_count,
        shuffle=False,
        collate_fn=collate_mididataset
    )
    if is_main_process:
        logging.info('Made DataLoaders')

    ######## Make model

    model = MyMidiTransformer(
        vocabs=vocabs,
        max_seq_length=args.data.max_seq_length,
        permute_mps=args.data.permute_mps,
        **vars(args.model)
    )
    if is_main_process:
        logging.info('Embedding size:')
        logging.info('\n'.join([
            f'{i} - {COMPLETE_ATTR_NAME[idx]} {vsize}'
            for i, (idx, vsize) in enumerate(zip(model.input_attrs_indices, model.embedding_vocabs_size))
        ]))
    to_input_attrs = model.to_input_attrs
    to_output_attrs = model.to_output_attrs

    ######## Use torchinfo

    if is_main_process:
        summary_str = str(torchinfo.summary(
            model,
            input_size=[
                (args.train.batch_size, args.data.max_seq_length, len(model.input_attrs_indices))
            ],
            dtypes=[torch.long],
            device=args.use_device,
            verbose=0
        ))
        logging.info(summary_str)

    ######## Make optimizer

    optimizer = AdamW(model.parameters(), args.train.lr_peak, betas=(0.9, 0.98), eps=1e-6, weight_decay=1e-2)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_warmup_and_linear_decay(
            step,
            args.train.lr_warmup_updates,
            args.train.lr_decay_end_ratio,
            args.train.lr_decay_end_updates
        )
    )

    ######## Move model to devices

    if args.use_parallel:
        model, optimizer, scheduler, train_dataloader, valid_dataloader = accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, valid_dataloader
        )
    else:
        model = model.to(args.use_device)

    ######## Training start

    if is_main_process:
        logging.info('Begin training')
    train_dataloader_iter = iter(train_dataloader)
    valid_dataloader_iter = iter(valid_dataloader)
    min_avg_valid_loss = float('inf')
    early_stop_counter = 0

    start_time = time()
    for start_num_updates in range(0, args.train.max_updates, args.train.validation_interval):
        model.train()
        train_loss_list: List[List[float]] = []
        # forward_time = 0
        # backward_time = 0
        training_tqdm = tqdm(
            range(args.train.validation_interval*gradient_accumulation_steps),
            disable=not is_main_process,
            desc=f'Training:{start_num_updates}~{start_num_updates+args.train.validation_interval}'
        )
        for step in training_tqdm:
            with accelerator.accumulate(model) if args.use_parallel else nullcontext():
                try:
                    batch_seqs, batch_mps_sep_indices = next(train_dataloader_iter)
                except StopIteration:
                    train_dataloader_iter = iter(train_dataloader)
                    batch_seqs, batch_mps_sep_indices = next(train_dataloader_iter)

                # batch_seqs has shape: (batch_size, seq_size, complete_attr_num)
                batch_input_seqs = to_input_attrs(batch_seqs[:, :-1])
                batch_target_seqs = to_output_attrs(batch_seqs[:, 1:])
                if not args.use_parallel:
                    batch_input_seqs = batch_input_seqs.to(args.use_device)
                    batch_target_seqs = batch_target_seqs.to(args.use_device)
                # start_forward_time = time()
                prediction = model(batch_input_seqs)
                # forward_time += time() - start_forward_time

                # start_backward_time = time()
                if args.data.use_permutable_subseq_loss:
                    loss, head_losses = compute_permutable_subseq_losses(
                        prediction,
                        batch_target_seqs,
                        batch_mps_sep_indices
                    )
                    # print('compute_permutable_subseq_losses use time:', time() - start_backward_time)
                else:
                    loss, head_losses = compute_losses(
                        prediction,
                        batch_target_seqs
                    )
                    # print('compute_losses use time:', time() - start_backward_time)
                # assert all(not torch.isnan(head_l).any() for head_l in head_losses)

                if is_main_process:
                    if step % gradient_accumulation_steps == 0: # still need to manually control
                        train_loss_list.append([0. for _ in train_output_attr_name])
                    # note that this only record the loss calculated on main process
                    train_loss_list[-1] = [
                        accu_l + head_l.item() / gradient_accumulation_steps
                        for accu_l, head_l in zip(train_loss_list[-1], head_losses)
                    ]
                    # print(train_loss_list[-1])

                if args.use_parallel:
                    accelerator.backward(loss)
                else:
                    loss.backward()

                if args.train.grad_clip_norm > 0:
                    if args.use_parallel:
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(model.parameters(), args.train.grad_clip_norm)
                    else:
                        clip_grad_norm_(model.parameters(), args.train.grad_clip_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            # print('loss + back propagate use time:', time() - start_backward_time)
            # backward_time += time() - start_backward_time
            # end with accelerator.accumulate
        # print('Forward time', forward_time, 'Backward time', backward_time)

        model.eval()
        valid_loss_list = []
        with torch.no_grad():
            for batch_seqs, batch_mps_sep_indices in tqdm(valid_dataloader, disable=not is_main_process, desc='Validation'):
                batch_input_seqs = to_input_attrs(batch_seqs[:, :-1])
                batch_target_seqs = to_output_attrs(batch_seqs[:, 1:])
                if not args.use_parallel:
                    batch_input_seqs = batch_input_seqs.to(args.use_device)
                    batch_target_seqs = batch_target_seqs.to(args.use_device)
                prediction = model(batch_input_seqs)

                if args.data.use_permutable_subseq_loss:
                    loss, head_losses = compute_permutable_subseq_losses(
                        prediction,
                        batch_target_seqs,
                        batch_mps_sep_indices,
                    )
                else:
                    loss, head_losses = compute_losses(
                        prediction,
                        batch_target_seqs,
                    )
                # need to gather, otherwise each process see different losses
                gathered_head_losses = accelerator.gather(head_losses)
                # gathered_head_losses: List[tensor.Tensor]
                # dim 0 is process dimension, dim 1 ~ last are original dimensions
                gathered_head_losses = torch.mean(torch.stack(gathered_head_losses), dim=1)
                valid_loss_list.append([head_l.item() for head_l in gathered_head_losses])


        cur_num_updates = start_num_updates + args.train.validation_interval

        if is_main_process:
            logging.info(
                'Progress: %d/%d, Time: %d, Learning rate: %.6f',
                cur_num_updates,
                args.train.max_updates,
                time()-start_time,
                scheduler.get_last_lr()[0]
            )
            assert train_loss_list
            log_losses(cur_num_updates, train_loss_list, valid_loss_list, loss_file_path)

        ckpt_model_file_path = os.path.join(ckpt_dir_path, f'{cur_num_updates}.pt')
        unwrapped_model = None
        if args.use_parallel:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model, ckpt_model_file_path) # don't need is_main_process
        else:
            torch.save(model, ckpt_model_file_path)

        if args.train.early_stop > 0:
            avg_valid_loss = sum([sum(valid_head_losses) for valid_head_losses in valid_loss_list]) / len(valid_loss_list)
            if avg_valid_loss >= min_avg_valid_loss:
                early_stop_counter += 1
                if early_stop_counter >= args.train.early_stop:
                    if is_main_process:
                        logging.info('Early stopped: No improvement for %d validations.', args.train.early_stop)
                    break
            else:
                early_stop_counter = 0
                min_avg_valid_loss = avg_valid_loss
                if is_main_process:
                    shutil.copyfile(ckpt_model_file_path, os.path.join(args.model_dir_path, 'best_model.pt'))
                    logging.info('New best model.')

        if is_main_process:
            if cur_num_updates % args.train.generate_sample_interval == 0:
                print('Generating conditional and unconditional sample for checkpoint')
                uncond_gen_text_list = generate_sample(
                    unwrapped_model if args.use_parallel else model,
                    steps=args.data.max_seq_length,
                    nucleus_sampling_threshold=args.nucleus_sampling_threshold
                )
                uncond_gen_piece = ' '.join(uncond_gen_text_list)
                with open(os.path.join(ckpt_dir_path, f'{cur_num_updates}_uncond.txt'), 'w+', encoding='utf8') as uncond_file:
                    uncond_file.write(uncond_gen_piece)
                try:
                    midiobj = piece_to_midi(uncond_gen_piece, vocabs.paras['nth'], ignore_pending_note_error=True)
                    midiobj.dump(os.path.join(ckpt_dir_path, f'{cur_num_updates}_uncond.mid'))
                except Exception:
                    print('Error when dumping uncond gen MidiFile object')
                    print(format_exc())

                cond_gen_text_list = generate_sample(
                    unwrapped_model if args.use_parallel else model,
                    steps=args.data.max_seq_length,
                    start_seq=cond_primer_array,
                    nucleus_sampling_threshold=args.nucleus_sampling_threshold
                )
                cond_gen_piece = ' '.join(cond_gen_text_list)
                with open(os.path.join(ckpt_dir_path, f'{cur_num_updates}_cond.txt'), 'w+', encoding='utf8') as cond_file:
                    cond_file.write(cond_gen_piece)
                try:
                    midiobj = piece_to_midi(cond_gen_piece, vocabs.paras['nth'], ignore_pending_note_error=True)
                    midiobj.dump(os.path.join(ckpt_dir_path, f'{cur_num_updates}_cond.mid'))
                except Exception:
                    print('Error when dumping cond gen MidiFile object')
                    print(format_exc())

    ######## Training end

    # if args.use_parallel:
        # Don't need this unless we use trackers in accelerator
        # accelerator.end_training()

    ######## Remove all checkpoints

    if is_main_process:
        ckpt_file_paths = glob.glob(os.path.join(ckpt_dir_path, '*.pt'), recursive=True)
        for ckpt_file_path in ckpt_file_paths:
            os.remove(ckpt_file_path)
        logging.info('==== train.py exit ====')
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
