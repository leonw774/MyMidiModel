from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
import glob
import json
import logging
import os
import shutil
from time import strftime, time
# from traceback import format_exc
from typing import List, Union

import accelerate
import numpy as np

from miditoolkit import MidiFile
from tqdm import tqdm
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
# from torch.profiler import profile, record_function, ProfilerActivity
import torchinfo

from util.corpus import (
    get_corpus_vocabs, to_pathlist_file_path,
    ALL_ATTR_NAMES, OUTPUT_ATTR_NAMES
)
from util.dataset import MidiDataset, collate_mididataset
from util.evaluations import (
    midi_list_to_features,
    piece_list_to_features,
    compare_with_ref,
    EVAL_DISTRIBUTION_FEATURE_NAMES,
    EVAL_SCALAR_FEATURE_NAMES
)
from util.generation import generate
from util.model import (
    MyMidiTransformer, compute_losses,
    LOSS_PADDING_ARG_CHOICES, LOSS_PADDING_ARG_CHOICES_DEFAULT
)
from util.type_wrappers import or_none


def parse_args():
    parser = ArgumentParser()

    data_group = parser.add_argument_group('data')
    data_group.add_argument(
        '--test-paths-file',
        dest='test_paths_file_path',
        type=str,
        default='',
        help='The path to the text file recording the testing files of the dataset'
    )
    data_group.add_argument(
        '--valid-paths-file',
        dest='valid_paths_file_path',
        type=str,
        default='',
        help='The path to the text file recording the testing files of the dataset'
    )
    data_group.add_argument(
        '--max-seq-length',
        type=int
    )
    data_group.add_argument(
        '--virtual-piece-step-ratio',
        type=float,
        default=0
    )
    data_group.add_argument(
        '--permute-mps',
        action='store_true'
    )
    data_group.add_argument(
        '--permute-track-number',
        action='store_true'
    )
    data_group.add_argument(
        '--pitch-augmentation-range',
        type=int,
        default=0
    )
    data_keys = set(action.dest for action in data_group._group_actions)

    model_group = parser.add_argument_group('model')
    model_group.add_argument(
        '--use-linear-attn',
        action='store_true'
    )
    model_group.add_argument(
        '--layers-number',
        type=int
    )
    model_group.add_argument(
        '--attn-heads-number',
        type=int
    )
    model_group.add_argument(
        '--embedding-dim',
        type=int
    )
    model_group.add_argument(
        '--not-use-mps-number',
        action='store_true',
    )
    model_keys = set(action.dest for action in model_group._group_actions)

    train_group = parser.add_argument_group('train')
    train_group.add_argument(
        '--batch-size',
        type=int,
        default=8
    )
    train_group.add_argument(
        '--max-updates',
        type=int,
        required=True
    )
    train_group.add_argument(
        '--validation-interval',
        type=int,
        required=True
    )
    train_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help='The max_norm of nn.util.clip_grad_norm_(). \
            If this value is zero, gradient clipping will not be used. \
            Default is %(default)s.'
    )
    train_group.add_argument(
        '--loss-padding',
        type=str,
        choices=LOSS_PADDING_ARG_CHOICES,
        default=LOSS_PADDING_ARG_CHOICES_DEFAULT
    )
    train_group.add_argument(
        '--lr-peak',
        type=float
    )
    train_group.add_argument(
        '--lr-warmup-updates',
        type=int
    )
    train_group.add_argument(
        '--lr-decay-end-updates',
        type=int
    )
    train_group.add_argument(
        '--lr-decay-end-ratio',
        type=float
    )
    train_group.add_argument(
        '--early-stop',
        type=int,
        help='If this value <= 0, no early stoping will perform.'
    )
    train_keys = set(action.dest for action in train_group._group_actions)

    eval_group = parser.add_argument_group('evaluation')
    eval_group.add_argument(
        '--softmax-temperature',
        type=float,
        nargs='+',
        default=[1.0],
        help='Control the temperature of softmax before multinomial sampling. \
            Default is %(default)s.'
    )
    eval_group.add_argument(
        '--sample-function',
        type=str,
        nargs='?',
        choices=('none', 'top-k', 'top-p', 'nucleus'),
        const='none',
        default='none',
        help='The sample function to used. Choice "top-p" is the same as "nucleus". \
            Default is %(default)s'
    )
    eval_group.add_argument(
        '--sample-threshold',
        type=float,
        nargs='+',
        default=[1.0],
        help='The probability threshold of nucleus sampling. \
            Default is %(default)s.'
    )
    eval_group.add_argument(
        '--valid-eval-sample-number',
        type=int,
        nargs='?',
        const=0,
        default=0,
        help='If set to 0, no eval on valid set will be performed. \
            Default is %(default)s'
    )
    eval_group.add_argument(
        '--valid-eval-worker-number',
        type=int,
        nargs='?',
        const=4,
        default=4,
        help='If set to 0, no eval on valid set will be performed. \
            Default is %(default)s'
    )
    eval_keys = set(action.dest for action in eval_group._group_actions)

    global_group = parser.add_argument_group('others')
    global_group.add_argument(
        '--dataloader-worker-number',
        type=int,
        default=4
    )
    global_group.add_argument(
        '--use-device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda'
    )
    global_group.add_argument(
        '--use-parallel',
        action='store_true'
    )
    global_group.add_argument(
        '--max-pieces-per-gpu',
        type=or_none(int),
        nargs='?',
        const=None,
        default=None,
        help='Set this to reasonable value to prevent OOM. If not set, assume no limit.'
    )
    global_group.add_argument(
        '--log',
        dest='log_file_path',
        type=str,
        default='',
    )
    global_group.add_argument(
        '--seed',
        type=or_none(int),
        nargs='?',
        const=None,
        default=None
    )
    global_group.add_argument(
        'midi_dir_path',
        type=str
    )
    global_group.add_argument(
        'corpus_dir_path',
        type=str
    )
    global_group.add_argument(
        'model_dir_path',
        type=str
    )
    global_keys = set(action.dest for action in global_group._group_actions)

    all_args_dict = vars(parser.parse_args())
    data_args = Namespace(**dict(
        (k, v)
        for k, v in all_args_dict.items()
        if k in data_keys
    ))
    model_args = Namespace(**dict(
        (k, v)
        for k, v in all_args_dict.items()
        if k in model_keys
    ))
    train_args = Namespace(**dict(
        (k, v)
        for k, v in all_args_dict.items()
        if k in train_keys
    ))
    eval_args = Namespace(**dict(
        (k, v)
        for k, v in all_args_dict.items()
        if k in eval_keys
    ))

    global_args_dict = dict(
        (k, v)
        for k, v in all_args_dict.items()
        if k in global_keys
    )
    global_args_dict['data'] = data_args
    global_args_dict['model'] = model_args
    global_args_dict['train'] = train_args
    global_args_dict['eval'] = eval_args

    # then turn into Namespace
    return Namespace(**global_args_dict)


# def vanilla_lr(step_num: int, warmup_steps: int, d_model: int) -> float:
#     return torch.rsqrt(d_model) * min(
#         torch.rsqrt(step_num), step_num * warmup_steps ** (-1.5)
#     )

def lr_warmup_and_linear_decay(
        step_num: int,
        warmup_steps: int,
        decay_end_ratio: float,
        decay_end_steps: int) -> float:
    if step_num < warmup_steps:
        return (step_num + 1) / warmup_steps
    r = min(1, ((step_num - warmup_steps) / decay_end_steps))
    return 1 - r * (1 - decay_end_ratio)


def log_losses(
        cur_num_updates: int,
        train_loss_list: List[float],
        train_head_losses_list: List[List[float]],
        valid_loss_list: List[float],
        valid_head_losses_list: List[List[float]],
        loss_file_path: str):
    avg_train_head_losses = [
        sum(head_losses) / len(head_losses)
        for head_losses in zip(*train_head_losses_list)
    ]
    avg_valid_head_losses = [
        sum(head_losses) / len(head_losses)
        for head_losses in zip(*valid_head_losses_list)
    ]
    avg_train_loss = sum(train_loss_list) / len(train_loss_list)
    avg_valid_loss = sum(valid_loss_list) / len(valid_loss_list)
    logging.info(
        'Avg. train head losses: %s Avg. train loss: %.6f',
        ', '.join([f'{l:.6f}' for l in avg_train_head_losses]), avg_train_loss
    )
    logging.info(
        'Avg. valid head losses: %s Avg. valid loss: %.6f',
        ', '.join([f'{l:.6f}' for l in avg_valid_head_losses]), avg_valid_loss
    )
    if not loss_file_path:
        return
    valid_len = len(train_head_losses_list)
    with open(loss_file_path, 'a', encoding='utf8') as loss_file:
        for i, (train_loss, train_head_losses) in enumerate(
                zip(train_loss_list, train_head_losses_list)):
            idx = cur_num_updates - valid_len + i + 1 # count from 1
            line = (
                f'{idx},'
                + ','.join([f'{l:.6f}' for l in train_head_losses])
                + f',{train_loss:.6f},'
            )
            if idx == cur_num_updates:
                line += (
                    ','.join([f'{l:.6f}' for l in avg_valid_head_losses])
                    + f',{avg_valid_loss:.6f}'
                )
            loss_file.write(line + '\n')


def generate_valid_sample_and_get_eval_features(
        model: MyMidiTransformer,
        sample_number: int,
        valid_eval_features: dict,
        softmax_temperature: float,
        sample_function: str,
        sample_threshold: float) -> dict:
    generated_text_list_list = [
        generate(
            model=model,
            max_generation_step=model.max_seq_length,
            primer_seq=None,
            softmax_temperature=softmax_temperature,
            sample_function=sample_function,
            sample_threshold=sample_threshold,
            show_tqdm=False
        )
        for _ in range(sample_number)
    ]
    generated_piece_list = [' '.join(t) for t in generated_text_list_list]
    generated_aggr_eval_features = piece_list_to_features(
        generated_piece_list,
        model.vocabs.paras['tpq']
    )
    compare_with_ref(generated_aggr_eval_features, valid_eval_features)
    return generated_aggr_eval_features

def log_generated_aggr_eval_features(generated_aggr_eval_features):
    logging.info('\t'.join([
        f'{fname[:15]}'
        for fname in EVAL_SCALAR_FEATURE_NAMES
    ]))
    logging.info('\t'.join([
        f'{generated_aggr_eval_features[fname]["mean"]:.12f}'
        for fname in EVAL_SCALAR_FEATURE_NAMES
    ]))
    logging.info('\t'.join([
        f'{fname[:10]}{suffix}'
        for suffix in ('_KLD', '_OA', '_HI')
        for fname in EVAL_DISTRIBUTION_FEATURE_NAMES
    ]))

    logging.info('\t'.join([
        f'{generated_aggr_eval_features[fname+suffix]:.12f}'
        for suffix in ('_KLD', '_OA', '_HI')
        for fname in EVAL_DISTRIBUTION_FEATURE_NAMES
    ]))


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

    parallel_devices_count = 1
    if args.use_parallel:
        parallel_devices_count = len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))
    if args.use_parallel and parallel_devices_count == 1:
        args.use_parallel = False

    gradient_accumulation_steps = 1
    if args.use_device != 'cpu':
        if args.max_pieces_per_gpu is not None: # if gpu memory is limited
            # effective batch size = gradient_accumulation_steps * batch_size * device_count
            gradient_accumulation_steps = int(
                args.train.batch_size / (args.max_pieces_per_gpu * parallel_devices_count)
            )
            if gradient_accumulation_steps > 1:
                args.train.batch_size = args.max_pieces_per_gpu
            if gradient_accumulation_steps == 0:
                gradient_accumulation_steps = 1

    accelerator: Union[accelerate.Accelerator , None]
    if args.use_parallel:
        accelerator = accelerate.Accelerator()
        is_main_process: bool = accelerator.is_main_process
    else:
        accelerator = None
        is_main_process: bool = True

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

    # log arguments
    ckpt_dir_path = os.path.join(args.model_dir_path, 'ckpt')
    if is_main_process:
        logging.info(strftime('==== train.py start at %Y%m%d-%H%M%S ===='))
        data_args_str  = '\n'.join([f'{k}:{v}' for k, v in vars(args.data).items()])
        model_args_str = '\n'.join([f'{k}:{v}' for k, v in vars(args.model).items()])
        eval_args_str  = '\n'.join([f'{k}:{v}' for k, v in vars(args.eval).items()])
        train_args_str = '\n'.join([f'{k}:{v}' for k, v in vars(args.train).items()])
        other_args_str = '\n'.join([
            f'{k}:{v}' for k, v in vars(args).items() if not isinstance(v, Namespace)
        ])
        logging.info(data_args_str)
        logging.info(model_args_str)
        logging.info(eval_args_str)
        logging.info(train_args_str)
        logging.info('gradient_accumulation_steps:%d', gradient_accumulation_steps)
        logging.info(other_args_str)

        if not os.path.isdir(args.model_dir_path):
            logging.info('Invalid model dir path: %s', args.model_dir_path)
            return 1
        if not os.path.isdir(ckpt_dir_path):
            logging.info('Invalid model ckpt dir path: %s', ckpt_dir_path)
            return 1

    ######## Prepare loss.csv

    vocabs = get_corpus_vocabs(args.corpus_dir_path)

    # loss csv file is in the root of model directory
    loss_file_path = os.path.join(args.model_dir_path, 'loss.csv')
    with open(loss_file_path, 'w+', encoding='utf8') as loss_file:
        loss_csv_head = 'step,'
        train_output_attr_name = ['train_' + n for n in OUTPUT_ATTR_NAMES]
        valid_output_attr_name = ['valid_' + n for n in OUTPUT_ATTR_NAMES]
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

    with open(args.data.test_paths_file_path, 'r', encoding='utf8') as test_paths_file:
        test_path_list = [p.strip() for p in test_paths_file.readlines()]
    with open(args.data.valid_paths_file_path, 'r', encoding='utf8') as valid_paths_file:
        valid_path_list = [p.strip() for p in valid_paths_file.readlines()]
    del args.data.test_paths_file_path
    del args.data.valid_paths_file_path

    if is_main_process:
        logging.info('Making training dataset')
    excluded_path_list = valid_path_list + test_path_list
    train_dataset = MidiDataset(
        data_dir_path=args.corpus_dir_path,
        excluded_path_list=excluded_path_list,
        **vars(args.data),
        verbose=is_main_process
    )
    del excluded_path_list

    if is_main_process:
        logging.info('Making valid dataset')
    excluded_path_list_for_valid = train_dataset.included_path_list + test_path_list
    # to prevent inconsistency, set valid dataset's virtual piece step to zero
    args.data.virtual_piece_step_ratio = 0
    valid_dataset = MidiDataset(
        data_dir_path=args.corpus_dir_path,
        excluded_path_list=excluded_path_list_for_valid,
        **vars(args.data),
        verbose=is_main_process
    )
    del excluded_path_list_for_valid

    if is_main_process:
        logging.info('Size of training set: %d', len(train_dataset))
        logging.info('Size of validation set: %d', len(valid_dataset))

    # if we want to generate and eval samples at each validation
    valid_eval_features = dict()
    if is_main_process and args.eval.valid_eval_sample_number > 0:
        valid_eval_features_json_path = os.path.join(
            args.midi_dir_path,
            'valid_eval_features.json'
        )
        if os.path.isfile(valid_eval_features_json_path):
            logging.info(
                'Getting valid set eval features from %s',
                valid_eval_features_json_path
            )
            with open(valid_eval_features_json_path, 'r', encoding='utf8') as f:
                valid_eval_features = json.load(f)
        else:
            logging.info('Computing valid set eval features')
            # copy validation midi files into model_dir
            pathlist_file_path = to_pathlist_file_path(args.corpus_dir_path)
            with open(pathlist_file_path, 'r', encoding='utf8') as pathlist_file:
                all_paths_list = [
                    p.strip()
                    for p in pathlist_file.readlines()
                ]
            valid_file_path_tuple = tuple(valid_path_list) # relative to dataset root
            valid_file_path_list = [ # relative to project root
                p
                for p in all_paths_list # this also filter out the un-processsable ones
                if p.endswith(valid_file_path_tuple)
            ]
            del valid_file_path_tuple
            for valid_path in valid_file_path_list:
                shutil.copy(valid_path, ckpt_dir_path) # use ckpt as temporary dir

            # get valid features
            valid_file_path_list = glob.glob(f'{ckpt_dir_path}/*.mid')
            tqdm_valid_file_path_list = tqdm(
                valid_file_path_list,
                desc='Reading valid set midi files',
                ncols=0
            )
            valid_midi_list = [
                MidiFile(p)
                for p in tqdm_valid_file_path_list
            ]
            _, valid_eval_features = midi_list_to_features(valid_midi_list, use_tqdm=True)
            del valid_midi_list

            # save json for reuse
            with open(valid_eval_features_json_path, 'w+', encoding='utf8') as f:
                json.dump(valid_eval_features, f)

            # delete the temporary validation midi files
            for p in valid_file_path_list:
                os.remove(p)
            del valid_file_path_list

    ######## Make dataloader

    train_dataloader = DataLoader(
        dataset=train_dataset,
        num_workers=args.dataloader_worker_number,
        batch_size=args.train.batch_size,
        shuffle=True,
        collate_fn=collate_mididataset
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        num_workers=args.dataloader_worker_number,
        batch_size=args.train.batch_size,
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
        permute_track_number=args.data.permute_track_number,
        **vars(args.model)
    )
    if is_main_process:
        logging.info('Embedding size:')
        logging.info('\n'.join([
            f'{i} - {ALL_ATTR_NAMES[idx]} {vsize}'
            for i, (idx, vsize) in enumerate(
                zip(model.input_attrs_indices, model.embedding_vocabs_size)
            )
        ]))
    to_input_attrs = model.to_input_attrs
    to_output_attrs = model.to_output_attrs

    ######## Use torchinfo

    if is_main_process:
        summary_str = str(torchinfo.summary(
            model,
            input_size=[
                (args.train.batch_size, args.data.max_seq_length, len(ALL_ATTR_NAMES))
            ],
            dtypes=[torch.long],
            device=args.use_device,
            verbose=0
        ))
        logging.info(summary_str)

    ######## Make optimizer

    optimizer = AdamW(
        model.parameters(),
        args.train.lr_peak,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=1e-2
    )
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
        model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, valid_dataloader
        )
    else:
        model = model.to(args.use_device)

    ######## Training start

    if is_main_process:
        logging.info('Begin training')
    train_dataloader_iter = iter(train_dataloader)
    # valid_dataloader_iter = iter(valid_dataloader)
    min_avg_valid_loss = float('inf')
    early_stop_counter = 0

    start_time = time()
    valid_interval = args.train.validation_interval
    for start_num_updates in range(0, args.train.max_updates, valid_interval):
        model.train()
        train_loss_list: List[float] = []
        train_head_losses_list: List[List[float]] = []
        training_tqdm = tqdm(
            range(valid_interval),
            disable=not is_main_process,
            desc=f'Training:{start_num_updates}~{start_num_updates+valid_interval}',
            ncols=100
        )
        for _ in training_tqdm:
            train_loss_list.append(0.0)
            train_head_losses_list.append([0.0 for _ in train_output_attr_name])
            for ga_step in range(gradient_accumulation_steps):
                # if use parallel and gradient accumulation step is not the last
                # then we can use no sync
                if args.use_parallel and ga_step + 1 != gradient_accumulation_steps:
                    parallel_no_sync_context = accelerator.no_sync(model)
                else:
                    parallel_no_sync_context = nullcontext()

                with parallel_no_sync_context:
                    try:
                        batch_seqs = next(train_dataloader_iter)
                    except StopIteration:
                        train_dataloader_iter = iter(train_dataloader)
                        batch_seqs = next(train_dataloader_iter)

                    # batch_seqs has shape: (batch_size, seq_size, complete_attr_num)
                    batch_input_seqs = to_input_attrs(batch_seqs[:, :-1])
                    batch_target_seqs = to_output_attrs(batch_seqs[:, 1:])
                    if not args.use_parallel:
                        batch_input_seqs = batch_input_seqs.to(args.use_device)
                        batch_target_seqs = batch_target_seqs.to(args.use_device)
                    prediction = model(batch_input_seqs)

                    loss, head_losses = compute_losses(
                        prediction,
                        batch_target_seqs,
                        args.train.loss_padding
                    )
                    loss = loss / gradient_accumulation_steps

                    if is_main_process:
                        # this only record the loss calculated on main process
                        # we assume they are close enough to "real" loss
                        train_loss_list[-1] += loss.item()
                        train_head_losses_list[-1] = [
                            acc_hl + hl.item() / gradient_accumulation_steps
                            for acc_hl, hl in zip(train_head_losses_list[-1], head_losses)
                        ]

                    if args.use_parallel:
                        accelerator.backward(loss)
                    else:
                        loss.backward()
            # end for gradient_accumulation_steps

            if args.train.max_grad_norm > 0:
                if args.use_parallel:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            model.parameters(),
                            args.train.max_grad_norm
                        )
                else:
                    clip_grad_norm_(model.parameters(), args.train.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        # end training interval

        model.eval()
        valid_loss_list: List[float] = []
        valid_head_losses_list: List[List[float]] = []
        with torch.no_grad():
            validation_tqdm = tqdm(
                valid_dataloader,
                disable=not is_main_process,
                desc='Validation',
                ncols=100
            )
            for batch_seqs in validation_tqdm:
                batch_input_seqs = to_input_attrs(batch_seqs[:, :-1])
                batch_target_seqs = to_output_attrs(batch_seqs[:, 1:])
                if not args.use_parallel:
                    batch_input_seqs = batch_input_seqs.to(args.use_device)
                    batch_target_seqs = batch_target_seqs.to(args.use_device)
                prediction = model(batch_input_seqs)

                loss, head_losses = compute_losses(
                    prediction,
                    batch_target_seqs,
                    args.train.loss_padding
                )
                if args.use_parallel:
                    # need to gather, otherwise each process see different losses
                    gather_loss: torch.Tensor = accelerator.gather(loss)
                    gather_head_losses: List[torch.Tensor] = accelerator.gather(head_losses)
                    # gather_head_losses: List[torch.Tensor]
                    # dim 0 is process dimension, dim 1 ~ last are original dimensions
                    gather_loss = gather_loss.mean()
                    gather_head_losses = torch.stack(gather_head_losses).mean(dim=1)
                    valid_loss_list.append(gather_loss.item())
                    valid_head_losses_list.append([hl.item() for hl in gather_head_losses])
                else:
                    valid_loss_list.append(loss.item())
                    valid_head_losses_list.append([hl.item() for hl in head_losses])

        cur_num_updates = start_num_updates + args.train.validation_interval

        if is_main_process:
            logging.info(
                'Progress: %d/%d, Time: %d, Learning rate: %.4e',
                cur_num_updates,
                args.train.max_updates,
                time()-start_time,
                scheduler.get_last_lr()[0]
            )
            log_losses(
                cur_num_updates,
                train_loss_list,
                train_head_losses_list,
                valid_loss_list,
                valid_head_losses_list,
                loss_file_path
            )

        ckpt_model_file_path = os.path.join(ckpt_dir_path, f'{cur_num_updates}.pt')
        unwrapped_model = None
        if args.use_parallel:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model, ckpt_model_file_path)
        else:
            torch.save(model, ckpt_model_file_path)

        if is_main_process and args.eval.valid_eval_sample_number > 0:
            generated_aggr_eval_features = generate_valid_sample_and_get_eval_features(
                model=torch.load(ckpt_model_file_path, map_location=args.use_device),
                sample_number=args.eval.valid_eval_sample_number,
                valid_eval_features=valid_eval_features,
                softmax_temperature=args.eval.softmax_temperature,
                sample_function=args.eval.sample_function,
                sample_threshold=args.eval.sample_threshold
            )
            log_generated_aggr_eval_features(generated_aggr_eval_features)

        avg_valid_loss = sum(valid_loss_list) / len(valid_loss_list)
        if avg_valid_loss >= min_avg_valid_loss:
            early_stop_counter += 1
            if args.train.early_stop >= 0 and early_stop_counter >= args.train.early_stop:
                if is_main_process:
                    logging.info(
                        'Early stopped: No improvement for %d validations.',
                        args.train.early_stop
                    )
                break
        else:
            early_stop_counter = 0
            min_avg_valid_loss = avg_valid_loss
            if is_main_process:
                shutil.copyfile(
                    ckpt_model_file_path,
                    os.path.join(args.model_dir_path, 'best_model.pt')
                )
                logging.info('New best model.')

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
    try:
        EXIT_CODE = main()
        exit(EXIT_CODE)
    except KeyboardInterrupt:
        if accelerate.state.AcceleratorState().is_main_process:
            logging.info('Training stopped by KeyboardInterrupt')
            logging.info('==== train.py exit ====')
        exit(1)
    except Exception as e:
        raise e
        # exit(1)
