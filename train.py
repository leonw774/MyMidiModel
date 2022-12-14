from argparse import ArgumentParser, Namespace
import glob
import json
import logging
import os
import shutil
from time import strftime, time
from traceback import format_exc
from typing import List, Dict

import accelerate
import numpy as np
from pandas import Series
from tqdm import tqdm
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import random_split, DataLoader
from torch.optim import AdamW, lr_scheduler
# from torch.profiler import profile, record_function, ProfilerActivity
import torchinfo
# import torchviz


from util.midi import piece_to_midi, get_first_k_measures
from util.corpus import COMPLETE_ATTR_NAME, OUTPUT_ATTR_NAME, get_corpus_vocabs, array_to_text_list, text_list_to_array
from util.dataset import MidiDataset, collate_mididataset
from util.model import (
    MyMidiTransformer,
    generate_sample,
    calc_losses,
    calc_permutable_subseq_losses
)
from util.evaluations import EVAL_FEATURE_NAMES, piece_to_features

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
        '--permute-mps',
        action='store_true'
    )
    data_parser.add_argument(
        '--permute-track-number',
        action='store_true'
    )
    data_parser.add_argument(
        '--pitch-augmentation',
        type=int,
        default=0
    )
    data_parser.add_argument(
        '--sample-from-start',
        action='store_true'
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
        '--input-no-tempo',
        action='store_true',
    )
    model_parser.add_argument(
        '--input-no-time-signature',
        action='store_true',
    )

    train_parser = ArgumentParser()
    train_parser.add_argument(
        '--split-ratio',
        type=float,
        nargs=2,
        default=[9, 1],
        help='The split ratio for training and validation. \
            If one is set to -1, for example (-1, 200), it means (len(complete_dataset) - 200, 200) \
            Default is %(default)s.'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=8
    )
    train_parser.add_argument(
        '--steps',
        type=int
    )
    train_parser.add_argument(
        '--validation-interval',
        type=int,
    )
    train_parser.add_argument(
        '--validation-steps',
        type=int,
    )
    train_parser.add_argument(
        "--grad-norm-clip",
        type=float,
        default=1.0,
        help='Set the max_norm of nn.util.clip_grad_norm_(). \
            If this value is zero, gradient clipping will not be used. \
            Default is %(desult)s.'
    )
    train_parser.add_argument(
        '--learning-rate', '--lr',
        dest='learning_rate',
        type=float
    )
    train_parser.add_argument(
        '--lr-warmup-steps',
        type=int
    )
    train_parser.add_argument(
        '--lr-decay-end-steps',
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
        default=2
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
        '--log',
        dest='log_file_path',
        default='',
    )
    global_parser.add_argument(
        '--eval-sample-number',
        type=int,
        default=64
    )
    global_parser.add_argument(
        '--ckpt-cond-primer-measures',
        type=int,
        default=4
    )
    global_parser.add_argument(
        '--model-dir-path',
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


def log_metrics(
        cur_step: int,
        start_time: float,
        scheduler: lr_scheduler.LambdaLR,
        gpu_mem_alloc_bytes: int,
        train_loss_list: List[List[float]],
        valid_loss_list: List[List[float]],
        loss_file_path: str):
    avg_train_losses = [ sum(head_loss_tuple) / len(head_loss_tuple) for head_loss_tuple in zip(*train_loss_list) ]
    avg_valid_losses = [ sum(head_loss_tuple) / len(head_loss_tuple) for head_loss_tuple in zip(*valid_loss_list) ]
    avg_train_losses_str = ', '.join([f'{l:.6f}' for l in avg_train_losses])
    avg_valid_losses_str = ', '.join([f'{l:.6f}' for l in avg_valid_losses])
    lr = scheduler.get_last_lr()[0]
    logging.info(
        'Time: %d, GPU Mem.: %.3f MB, Learning rate: %.6f',
        time()-start_time, gpu_mem_alloc_bytes/1e6, lr
    )
    logging.info(
        'Avg. train losses: %s Avg. accum. train loss: %.6f \nAvg. valid losses: %s Avg. accum. valid loss: %.6f',
        avg_train_losses_str, sum(avg_train_losses), avg_valid_losses_str, sum(avg_valid_losses)
    )
    if loss_file_path:
        with open(loss_file_path, 'a', encoding='utf8') as loss_file:
            loss_file.write(
                f'{cur_step}, {time()-start_time:.6f}, {gpu_mem_alloc_bytes} {lr:.6f}, '
                + f'{avg_train_losses_str}, {sum(avg_train_losses):.6f}, {avg_valid_losses_str}, {sum(avg_valid_losses):.6f}\n'
            )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        args.use_device = 'cpu'
        args.use_parallel = False
    args.use_device = torch.device(args.use_device)

    accelerator = accelerate.Accelerator() if args.use_parallel else None
    is_main_process = accelerator.is_main_process if args.use_parallel else True
    parallel_devices_count = len(os.getenv('CUDA_VISIBLE_DEVICES').split(',')) if args.use_parallel else 1

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
    eval_dir_path = os.path.join(args.model_dir_path, 'eval_samples')
    if is_main_process:
        logging.info(strftime('=== train.py start at %Y%m%d-%H%M%S ==='))
        if not os.path.isdir(args.model_dir_path):
            logging.info('Invalid model dir path: %s', args.model_dir_path)
            return 1
        if not os.path.isdir(ckpt_dir_path):
            logging.info('Invalid model ckpt dir path: %s', ckpt_dir_path)
            return 1
        if not os.path.isdir(eval_dir_path):
            logging.info('Invalid model eval samples dir path: %s', eval_dir_path)
            return 1

        data_args_str = '\n'.join([f'{k}:{v}' for k, v in vars(args.data_args).items()])
        model_args_str = '\n'.join([f'{k}:{v}' for k, v in vars(args.model_args).items()])
        train_args_str = '\n'.join([f'{k}:{v}' for k, v in vars(args.train_args).items()])
        args_str = '\n'.join([f'{k}:{v}' for k, v in vars(args).items() if not isinstance(v, Namespace)])
        logging.info(data_args_str)
        logging.info(model_args_str)
        logging.info(train_args_str)
        logging.info(args_str)

    vocabs = get_corpus_vocabs(args.corpus_dir_path)

    # loss csv file is in the checkpoint directory
    loss_file_path = os.path.join(args.model_dir_path, 'loss.csv')
    with open(loss_file_path, 'w+', encoding='utf8') as loss_file:
        loss_csv_head = 'step, time, gpu_memory_allocated_bytes, learning_rate, '
        train_output_attr_name = ['train_' + n for n in OUTPUT_ATTR_NAME]
        valid_output_attr_name = ['valid_' + n for n in OUTPUT_ATTR_NAME]
        if vocabs.paras['position_method'] == 'event':
            train_output_attr_name = train_output_attr_name[:-1]
            valid_output_attr_name = valid_output_attr_name[:-1]
        loss_csv_head += ', '.join(
            train_output_attr_name + ['train_total']
            + valid_output_attr_name + ['valid_total']
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

    # make dataset
    complete_dataset = MidiDataset(data_dir_path=args.corpus_dir_path, **vars(args.data_args), verbose=is_main_process)
    if is_main_process:
        logging.info('Made MidiDataset')
    train_ratio, valid_ratio = args.train_args.split_ratio
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
        logging.info('Legnth of training set: %d', len(train_dataset))
        logging.info('Legnth of validation set: %d', len(valid_dataset))

    # get random conditional primer from complete_dataset
    cond_primer_index = np.random.randint(len(complete_dataset.pieces))
    cond_primer_array = complete_dataset.pieces[str(cond_primer_index)]
    cond_primer_text_list = array_to_text_list(cond_primer_array, vocabs)
    cond_primer_text_list = get_first_k_measures(cond_primer_text_list, args.ckpt_cond_primer_measures)
    cond_primer_array = text_list_to_array(cond_primer_text_list, vocabs).astype(np.int32)
    cond_primer_array = torch.from_numpy(np.expand_dims(cond_primer_array, axis=0))

    # make dataloader
    # cannot handle multiprocessing if use npz mmap
    if not isinstance(complete_dataset.pieces, dict):
        args.dataloader_worker_number = 1
    train_dataloader = DataLoader(
        dataset=train_dataset,
        num_workers=args.dataloader_worker_number,
        batch_size=args.train_args.batch_size // parallel_devices_count,
        shuffle=False, # no need to shuffle because random_split did
        collate_fn=collate_mididataset
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        num_workers=args.dataloader_worker_number,
        batch_size=args.train_args.batch_size // parallel_devices_count,
        shuffle=False,
        collate_fn=collate_mididataset
    )
    if is_main_process:
        logging.info('Made DataLoaders')

    # make model
    model = MyMidiTransformer(
        vocabs=vocabs,
        max_seq_length=args.data_args.max_seq_length,
        **vars(args.model_args)
    )
    if is_main_process:
        logging.info('Embedding size:')
        logging.info('\n'.join([
            f'{i} - {name} {vsize}' for i, (name, vsize) in enumerate(zip(COMPLETE_ATTR_NAME, model.embedding_vocabs_size))
        ]))
    to_input_attrs = model.to_input_attrs
    to_output_attrs = model.to_output_attrs

    # use torchinfo
    if is_main_process:
        summary_str = str(torchinfo.summary(
            model,
            input_size=[
                (args.train_args.batch_size, args.data_args.max_seq_length, len(model.input_attrs_indices))
            ],
            dtypes=[torch.long],
            device=args.use_device,
            verbose=0
        ))
        logging.info(summary_str)

    # make optimizer
    optimizer = AdamW(model.parameters(), args.train_args.learning_rate, betas=(0.9, 0.98), eps=1e-8)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_warmup_and_linear_decay(
            step,
            args.train_args.lr_warmup_steps,
            args.train_args.lr_decay_end_ratio,
            args.train_args.lr_decay_end_steps
        )
    )

    # move things to devices
    if args.use_parallel:
        model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, valid_dataloader
        )
    else:
        model = model.to(args.use_device)

    # training start
    if is_main_process:
        logging.info('Begin training')
    train_dataloader_iter = iter(train_dataloader)
    valid_dataloader_iter = iter(valid_dataloader)
    complete_train_loss_list = []
    complete_valid_loss_list = []
    min_avg_valid_loss = float('inf')
    early_stop_counter = 0

    start_time = time()
    for start_step in range(0, args.train_args.steps, args.train_args.validation_interval):
        if is_main_process:
            logging.info('Training: %d/%d', start_step, args.train_args.steps)
        model.train()
        train_loss_list = []
        train_loss_list: List[List[float]]
        # forward_time = 0
        # backward_time = 0
        for _ in tqdm(range(args.train_args.validation_interval), disable=not is_main_process):
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
            # assert all(not torch.isnan(h).any() for h in prediction), [torch.isnan(h).nonzero() for h in prediction]
            # forward_time += time() - start_forward_time
            # start_backward_time = time()
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            if args.data_args.use_permutable_subseq_loss:
                head_losses = calc_permutable_subseq_losses(prediction, batch_target_seqs, batch_mps_sep_indices)
                # print('\ncalc_permutable_subseq_losses use time:', time() - start_backward_time)
            else:
                head_losses = calc_losses(prediction, batch_target_seqs)
                    # print('\ncalc_losses use time:', time() - start_backward_time)
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=32))
            # assert all(not torch.isnan(hl).any() for hl in head_losses), [torch.isnan(head).nonzero() for hl in head_losses]
            train_loss_list.append([hl.item() for hl in head_losses])
            # print(train_loss_list[-1])
            loss = torch.mean(torch.stack(head_losses))
            # dot=torchviz.make_dot(loss, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
            # dot.render(filename='lossbackward_mps', format='png')s
            if args.use_parallel:
                accelerator.backward(loss)
            else:
                loss.backward()
            # print(torch.cuda.memory_allocated()/1e6, 'MB')
            if args.train_args.grad_norm_clip > 0:
                if args.use_parallel:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.train_args.grad_norm_clip)
                else:
                    clip_grad_norm_(model.parameters(), args.train_args.grad_norm_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # print('loss + back propagate use time:', time() - start_backward_time)
            # torch.cuda.empty_cache() # use only when oom did happen
            # backward_time += time() - start_backward_time
        # print('Forward time', forward_time, 'Backward time', backward_time)

        if is_main_process:
            print('Validation')
        model.eval()
        valid_loss_list = []
        with torch.no_grad():
            for _ in tqdm(range(min(args.train_args.validation_steps, len(valid_dataloader))), disable=not is_main_process):
                try:
                    batch_seqs, batch_mps_sep_indices = next(valid_dataloader_iter)
                except StopIteration:
                    valid_dataloader_iter = iter(valid_dataloader)
                    batch_seqs, batch_mps_sep_indices = next(valid_dataloader_iter)

                batch_input_seqs = to_input_attrs(batch_seqs[:, :-1])
                batch_target_seqs = to_output_attrs(batch_seqs[:, 1:])
                if not args.use_parallel:
                    batch_input_seqs = batch_input_seqs.to(args.use_device)
                    batch_target_seqs = batch_target_seqs.to(args.use_device)
                prediction = model(batch_input_seqs)

                if args.data_args.use_permutable_subseq_loss:
                    head_losses = calc_permutable_subseq_losses(prediction, batch_target_seqs, batch_mps_sep_indices)
                else:
                    head_losses = calc_losses(prediction, batch_target_seqs)
                valid_loss_list.append([hl.item() for hl in head_losses])

        complete_train_loss_list.extend(train_loss_list)
        complete_valid_loss_list.extend(valid_loss_list)
        cur_step = start_step + args.train_args.validation_interval

        gpu_mem_alloc_bytes = -1
        if parallel_devices_count == 1:
            gpu_mem_alloc_bytes = torch.cuda.memory_allocated(0)
        else:
            gpu_mem_alloc_bytes = sum(
                map(torch.cuda.memory_allocated, range(torch.cuda.device_count()))
            )

        ckpt_model_file_path = os.path.join(ckpt_dir_path, f'{cur_step}.pt')
        unwrapped_model = None
        if args.use_parallel:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model, ckpt_model_file_path) # don't need is_main_process
        else:
            torch.save(model, ckpt_model_file_path)

        if is_main_process:
            log_metrics(cur_step, start_time, scheduler, gpu_mem_alloc_bytes, train_loss_list, valid_loss_list, loss_file_path)
            print('Generating conditional and unconditional sample for checkpoint')
            uncond_gen_text_list = generate_sample(
                unwrapped_model if args.use_parallel else model,
                steps=args.data_args.max_seq_length
            )
            uncond_gen_piece = ' '.join(uncond_gen_text_list)
            with open(os.path.join(ckpt_dir_path, f'{cur_step}_uncond.txt'), 'w+', encoding='utf8') as uncond_file:
                uncond_file.write(uncond_gen_piece)
            try:
                midiobj = piece_to_midi(uncond_gen_piece, vocabs.paras['nth'], ignore_pending_note_error=True)
                midiobj.dump(os.path.join(ckpt_dir_path, f'{cur_step}_uncond.mid'))
            except Exception:
                print('Error when dumping uncond gen MidiFile object')
                print(format_exc())
            cond_gen_text_list = generate_sample(
                unwrapped_model if args.use_parallel else model,
                steps=args.data_args.max_seq_length,
                start_seq=cond_primer_array
            )
            cond_gen_piece = ' '.join(cond_gen_text_list)
            with open(os.path.join(ckpt_dir_path, f'{cur_step}_cond.txt'), 'w+', encoding='utf8') as cond_file:
                cond_file.write(cond_gen_piece)
            try:
                midiobj = piece_to_midi(cond_gen_piece, vocabs.paras['nth'], ignore_pending_note_error=True)
                midiobj.dump(os.path.join(ckpt_dir_path, f'{cur_step}_cond.mid'))
            except Exception:
                print('Error when dumping cond gen MidiFile object')
                print(format_exc())

            if args.train_args.early_stop > 0:
                avg_valid_loss = sum([sum(valid_head_losses) for valid_head_losses in valid_loss_list]) / len(valid_loss_list)
                if avg_valid_loss >= min_avg_valid_loss:
                    early_stop_counter += 1
                    if early_stop_counter >= args.train_args.early_stop:
                        logging.info('Early stopped: No improvement for %d validations.', args.train_args.early_stop)
                        break
                else:
                    early_stop_counter = 0
                    min_avg_valid_loss = avg_valid_loss
                    shutil.copyfile(ckpt_model_file_path, os.path.join(args.model_dir_path, 'best_model.pt'))
                    logging.info('New best model.')
    # training end
    if args.use_parallel:
        accelerate.end_training()

    # remove all checkpoints
    if is_main_process:
        ckpt_file_paths = glob.glob(os.path.join(ckpt_dir_path, '*.pt'), recursive=True)
        for ckpt_file_path in ckpt_file_paths:
            os.remove(ckpt_file_path)

    # evaluation
    if is_main_process:
        logging.info('Generating unconditional generation sample for evaluation')
        best_model = torch.load(os.path.join(args.model_dir_path, 'best_model.pt'))
        uncond_gen_piece_list = []
        uncond_gen_start_time = time()
        uncond_gen_total_token_length = 0
        for _ in range(args.eval_sample_number):
            uncond_gen_text_list = generate_sample(best_model, args.data_args.max_seq_length)
            uncond_gen_total_token_length += len(uncond_gen_text_list)
            uncond_gen_piece = ' '.join(uncond_gen_text_list)
            uncond_gen_piece_list.append(uncond_gen_piece)
        logging.info(
            'Done. Generating %d pieces with max_length %d takes %.3f seconds',
            args.eval_sample_number,
            args.data_args.max_seq_length,
            time() - uncond_gen_start_time
        )
        logging.info('Avg. tokens# in the samples are %.3f', uncond_gen_total_token_length / args.eval_sample_number)

        eval_sample_features_per_piece = []
        eval_sample_features_per_piece: List[ Dict[str, float] ]
        for i, uncond_gen_piece in enumerate(uncond_gen_piece_list):
            open(os.path.join(eval_dir_path, f'{i}.txt'), 'w+', encoding='utf8').write(uncond_gen_piece)
            try:
                piece_to_midi(uncond_gen_piece, vocabs.paras['nth'], ignore_pending_note_error=True).dump(
                    os.path.join(eval_dir_path, f'{i}.mid')
                )
                eval_sample_features_per_piece.append(
                    piece_to_features(uncond_gen_piece, nth=vocabs.paras['nth'], max_pairs_number=int(10e6))
                )
            except (AssertionError, ValueError):
                print('Error when dumping eval uncond gen MidiFile object')
                print(format_exc())

        eval_sample_features = {
            fname: [
                fs[fname]
                for fs in eval_sample_features_per_piece
            ]
            for fname in EVAL_FEATURE_NAMES
        }

        eval_sample_features_stats = dict()
        for fname in EVAL_FEATURE_NAMES:
            fname_description = dict(Series(eval_sample_features[fname]).dropna().describe())
            fname_description: Dict[str, np.float64]
            eval_sample_features_stats[fname] = {
                k : float(v) for k, v in fname_description.items()
            }
        with open(os.path.join(args.model_dir_path, 'eval_sample_feature_stats.json'), 'w+', encoding='utf8') as eval_stat_file:
            json.dump(eval_sample_features_stats, eval_stat_file)
        logging.info('==== train.py exit ====')
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
