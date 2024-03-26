from argparse import ArgumentParser, Namespace
from functools import partial
import os
import subprocess
import tempfile
from time import time
from traceback import format_exc
from typing import List
from tqdm import tqdm

import numpy as np
import torch
from miditoolkit import MidiFile

from util.tokens import BEGIN_TOKEN_STR, END_TOKEN_STR
from util.midi import (
    midi_to_piece, piece_to_midi, get_first_k_measures, get_first_k_ticks
)
from util.corpus import (
    to_corpus_file_path, to_paras_file_path, dump_corpus_paras
)
from util.arrays import text_list_to_array, get_full_array_string

from util.model import MyMidiTransformer
from util.generation import generate, permute_track_number
from util.argparse_helper import or_none, MyHelpFormatter

def read_args():
    parser = ArgumentParser(formatter_class=MyHelpFormatter)

    parser.add_argument(
        '--primer', '-p',
        type=str,
        default=None,
        help='A path to MIDI file, or a file containing list of paths of MIDI \
            files that would be used as the primer in conditional generation. \
            If the extension is "*.mid" or "*.midi", try to parse it as MIDI. \
            Otherwise, try to parse it as list of paths to midi files, \
            separated by newlines. If this option is not set, unconditional \
            generation will be performed.'
    )
    parser.add_argument(
        '--primer-length', '-l',
        type=int,
        default=0,
        help='How long from the start the primer music should be used. \
            Default is %(default)s. The unit of these number is controlled \
            with "--unit" option.'
    )
    parser.add_argument(
        '--unit', '-u',
        choices=['measure', 'tick', 'token'],
        default='measure',
        help='Specify the unit of PRIMER_LENGTH: \\n\
            - "measure": use the first PRIMER_LENGTH measures of the piece \
            no matter how long it actually is. \\n\
            - "tick": tick is the time unit. The length of tick is determined \
            by parameters of the corpus on which the model is trained. \\n\
            - "token": use the first PRIMER_LENGTH tokens of the piece. \\n\
            Default is %(default)s'
    )
    parser.add_argument(
        '--sample-number', '-n',
        type=int,
        default=0,
        help='How many sample will be generated. Default is %(default)s. \\n\
            If no primer used, it simply generate SAMPLE_NUMBER samples. \\n\
            If primer is a midi file, it generate SAMPLE_NUMBER samples \
            using that primer repeatly. \\n\
            If primer is list of midi file paths, it generates samples with \
            the first SAMPLE_NUMBER primers. When SAMPLE_NUMBER == 0, \
            it using all of them'
    )
    parser.add_argument(
        '--output-text', '-o',
        action='store_true'
    )
    parser.add_argument(
        '--output-array-text', '-a',
        action='store_true'
    )
    parser.add_argument(
        '--max-generation-step', '--step', '-s',
        type=int,
        default=None,
        help='The maximum TOKENS in each sample would be generated. \
            Default is the model\'s max sequence length.'
    )
    parser.add_argument(
        '--no-adjust-logit',
        action='store_true',
        help='Trun off the logit adjustment that helps generate \
            next token that satisfy the format rule.'
    )
    parser.add_argument(
        '--softmax-temperature', '-t',
        type=float,
        nargs='+',
        default=[1.0],
        help='Control the temperature of softmax before sampling. \
            Default is %(default)s.'
    )
    parser.add_argument(
        '--sample-function', '-f',
        type=str,
        nargs='?',
        choices=('none', 'top-k', 'top-p', 'nucleus'),
        const='none',
        default='none',
        help='The sample function to used. \
            Choice "nucleus" is the same as "top-p". \
            Default is %(default)s'
    )
    parser.add_argument(
        '--sample-threshold', '--threshold', '-e',
        type=float,
        nargs='+',
        default=[1.0],
        help='The probability threshold of the sample function. \
            Default is %(default)s.'
    )
    parser.add_argument(
        '--try-count-limit',
        type=int,
        default=100,
        help='The model may fail to generate next token that satisfy \
            the format rule if --no-adjust-prob is set. \
            The generation ends when its trying times pass this limit. \
            Default is %(default)s.'
    )
    parser.add_argument(
        '--use-device',
        type=str,
        nargs='?',
        const='cuda',
        default='cuda',
        help='What device the model would be on.'
    )
    parser.add_argument(
        '--worker-number', '-w',
        type=int,
        default=8,
        help='Number of workers for applying BPE vocabs if used.'
    )
    parser.add_argument(
        '--print-exception',
        action='store_true',
        help='When model fail to generate next token that satisfy \
            the format rule. Print out the exception message.'
    )
    parser.add_argument(
        '--no-sample-tqdm', '--no-tqdm',
        action='store_true',
        help='No tqdm progress bar for single sample generation.'
    )
    parser.add_argument(
        '--no-total-tqdm',
        action='store_true',
        help='No tqdm progress bar for all samples generation.'
    )
    parser.add_argument(
        '--seed',
        type=or_none(int),
        default=None
    )
    parser.add_argument(
        'model_file_path',
        type=str
    )
    parser.add_argument(
        'output_file_path',
        type=str,
        help='The base of the path of generated MIDI file(s). \
            If SAMPLE_NUMBER == 1, the path is "{OUTPUT_FILE_PATH}.mid". \
            Otherwise, the paths are "{OUTPUT_FILE_PATH}_{i}.mid", \
            where i is 1 ~ SAMPLE_NUMBER.'
    )

    return parser.parse_args()


def gen_handler(
        model: MyMidiTransformer,
        primer_seq,
        args: Namespace,
        output_file_path: str):
    try:
        gen_text_list = generate(
            model,
            max_generation_step=args.max_generation_step,
            primer_seq=primer_seq,
            softmax_temperature=args.softmax_temperature,
            try_count_limit=args.try_count_limit,
            use_adjust_logit=(not args.no_adjust_logit),
            sample_function=args.sample_function,
            sample_threshold=args.sample_threshold,
            print_exception=args.print_exception,
            show_tqdm=(not args.no_sample_tqdm)
        )
        if gen_text_list == BEGIN_TOKEN_STR + ' ' + END_TOKEN_STR:
            print(
                f'{output_file_path}: generated empty piece. '
                f'Will not output file.'
            )
        else:
            if args.output_text:
                out_path = f'{output_file_path}.txt'
                with open(out_path, 'w+', encoding='utf8') as f:
                    f.write(' '.join(gen_text_list))
            if args.output_array_text:
                out_path = f'{output_file_path}_array.txt'
                with open(out_path, 'w+', encoding='utf8') as f:
                    f.write(
                        get_full_array_string(
                            text_list_to_array(gen_text_list, model.vocabs),
                            model.vocabs
                        )
                    )
            midi = piece_to_midi(
                piece=' '.join(gen_text_list),
                tpq=model.vocabs.paras['tpq']
            )
            midi.dump(f'{output_file_path}.mid')
    except Exception:
        print('Generation failed becuase the following exception:')
        print(format_exc())


def cut_primer_piece(
        primer_piece: str,
        primer_length: int,
        length_unit: str,
        tpq: int) -> str:
    primer_text_list = primer_piece.split(' ')
    if length_unit == 'measure':
        primer_text_list = get_first_k_measures(
            text_list=primer_text_list,
            k=primer_length
        )
    elif length_unit == 'tick':
        primer_text_list = get_first_k_ticks(
            text_list=primer_text_list,
            tpq=tpq,
            k=primer_length
        )
    if primer_text_list[-1] != END_TOKEN_STR:
        primer_text_list.append(END_TOKEN_STR)
    processed_primer_piece = ' '.join(primer_text_list)
    return processed_primer_piece


def midi_file_list_to_text_list_list(
        primer_paths: List[str],
        primer_length: int,
        length_unit: str,
        vocabs,
        worker_number: int = 1) -> List[List[str]]:
    """
    1. Read the midi files
    2. cut them into primers
    3. write them into corpus
    4. apply BPE
    5. return the primers in text_list form
    """
    primer_piece_list: List[str] = []
    tqdm_primer_paths = tqdm(
        primer_paths,
        desc='Encoding midi files to text representation',
        ncols=0
    )
    for midi_path in tqdm_primer_paths:
        try:
            p = midi_to_piece(MidiFile(midi_path), **vocabs.paras)
            primer_piece_list.append(p)
        except Exception:
            pass
    # apply measure and tick cut
    partial_cut_primer_piece = partial(
        cut_primer_piece,
        primer_length=primer_length,
        length_unit=length_unit,
        tpq=vocabs.paras['tpq']
    )
    primer_piece_list = list(map(partial_cut_primer_piece, primer_piece_list))

    if len(vocabs.bpe_shapes_list) > 0 and primer_length > 0:
        print('Applying BPE')
        # model use BPE, then apply it
        with tempfile.TemporaryDirectory() as in_dir_path, \
             tempfile.TemporaryDirectory() as out_dir_path:
            # make tmp corpus, paras, and shape_vocab files
            in_corpus_path = to_corpus_file_path(in_dir_path)
            with open(in_corpus_path, 'w+', encoding='utf8') as in_corpus_file:
                in_corpus_file.write('\n'.join(primer_piece_list) + '\n')

            in_paras_path = to_paras_file_path(in_dir_path)
            with open(in_paras_path, 'w+', encoding='utf8') as in_paras_file:
                in_paras_file.write(dump_corpus_paras(vocabs.paras))

            in_sv_path = os.path.join(in_dir_path, 'shape_vocab')
            with open(in_sv_path, 'w+', encoding='utf8') as in_sv_file:
                in_sv_file.write('\n'.join(vocabs.bpe_shapes_list) + '\n')

            # make sure this script is runing at project's root
            abs_cwd = os.path.abspath(os.getcwd())
            abs_project_root = os.path.dirname(os.path.abspath(__file__))
            assert abs_cwd == abs_project_root

            # make sure the program is there and new
            subprocess.run(
                ['make', '-C', './bpe'],
                check=True,
                stdout=subprocess.DEVNULL
            )

            out_corpus_path = to_corpus_file_path(out_dir_path)
            apply_args = [
                './bpe/apply_vocab',
                # '-log',
                # '-clearLine',
                in_dir_path,
                out_corpus_path,
                in_sv_path
            ]
            if worker_number > 1:
                apply_args.append(str(worker_number))
            subprocess.run(apply_args, check=True, stdout=subprocess.DEVNULL)

            # get content from output
            with open(out_corpus_path, 'r', encoding='utf8') as out_corpus_file:
                merged_piece_list = [
                    line.strip()
                    for line in out_corpus_file.readlines()
                ]

        primer_text_list_list = [
            merged_piece.split(' ')
            for merged_piece in merged_piece_list
        ]
    else:
        primer_text_list_list = [
            piece.split(' ')
            for piece in primer_piece_list
        ]

    # remove end-of-sequence token
    for text_list in primer_text_list_list:
        if text_list[-1] == END_TOKEN_STR:
            text_list.pop()

    # apply token cut
    if length_unit == 'token':
        primer_text_list_list = [
            text_list[:primer_length]
            for text_list in primer_text_list_list
            if len(text_list) >= primer_length
        ]

    return primer_text_list_list


def main():
    args = read_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    overhead_time_begin = time()

    # device
    # if is empty string, use default: cuda
    if args.use_device == '':
        args.use_device = 'cuda'
    if not args.use_device.startswith('cuda') and args.use_device != 'cpu':
        raise ValueError(f'Bad device name: "{args.use_device}"')
    if not torch.cuda.is_available():
        print(
            '--use-device is \'cuda\' but found no CUDA device.',
            'Changed to \'cpu\'.'
        )
        args.use_device = 'cpu'
    assert args.sample_number >= 0

    # model
    model = torch.load(
        args.model_file_path,
        map_location=torch.device(args.use_device)
    )
    assert isinstance(model, MyMidiTransformer)
    if args.max_generation_step is None:
        args.max_generation_step = model.max_seq_length

    if args.output_file_path.endswith('.mid'):
        args.output_file_path = args.output_file_path[:-4]

    overhead_time = time() - overhead_time_begin
    primer_process_time_begin = time()

    # primer
    primer_seq_list = []
    if args.primer is  None:
        if args.sample_number == 0:
            args.sample_number = 1
        primer_seq_list = [None] * args.sample_number
    else:
        print('Processing primer')
        assert args.primer_length >= 0
        if not os.path.isfile(args.primer):
            print('Primer file not exists')
            raise FileNotFoundError()

        encode_args = {
            'primer_length': args.primer_length,
            'length_unit': args.unit,
            'vocabs': model.vocabs,
            'worker_number': args.worker_number
        }
        if args.primer.endswith('.mid') or args.primer.endswith('.midi'):
            print('From midi file:', args.primer)
            if args.sample_number == 0:
                args.sample_number = 1
            primer_path_list = [args.primer]
            primer_text_list_list = midi_file_list_to_text_list_list(
                primer_path_list,
                **encode_args
            )
            primer_text_list_list = primer_text_list_list * args.sample_number
        else: # we guess it is a list of paths to midi files
            print('From path list:', args.primer)

            with open(args.primer, 'r', encoding='utf8') as primer_file:
                primer_path_list = primer_file.readlines()
            print(f'Read {len(primer_path_list)} lines')

            # keep first n line if n > 0
            if args.sample_number != 0:
                if len(primer_path_list) > args.sample_number:
                    print(
                        f'Only keep first {len(primer_path_list)} lines',
                        f'as sample number is {args.sample_number}'
                    )
                    print('(Set sample number to 0 to keep all primers)')
                    primer_path_list = primer_path_list[:args.sample_number]
            else:
                args.sample_number = len(primer_path_list)

            # check if file exists and have MIDI extension
            primer_path_list = [p.strip() for p in primer_path_list]
            assert all(
                p.endswith('.mid') or p.endswith('.MID') or p.endswith('.midi')
                for p in primer_path_list
            )
            assert all(os.path.isfile(p) for p in primer_path_list)

            primer_text_list_list = midi_file_list_to_text_list_list(
                primer_path_list,
                **encode_args
            )
            if len(primer_text_list_list) == 0:
                print('No file processed. Generation end.')
                return 0

            print(f'Processed {len(primer_text_list_list)} files.')
            if len(primer_text_list_list) < args.sample_number:
                print(
                    'Processed primer number is less than required',
                    f'{args.sample_number}.'
                )
                print(f'Will generate {len(primer_text_list_list)} pieces.')
                args.sample_number = len(primer_text_list_list)

        for primer_text_list in primer_text_list_list:
            # turn primer text list into array
            primer_seq = text_list_to_array(
                text_list=primer_text_list,
                vocabs=model.vocabs
            )
            if model.permute_track_number:
                primer_seq = permute_track_number(
                    primer_seq,
                    model.vocabs.track_numbers.size
                )
            primer_seq = np.expand_dims(primer_seq, axis=0).astype(np.int32)
            primer_seq = torch.from_numpy(primer_seq)
            primer_seq_list.append(primer_seq)

    primer_process_time = time() - primer_process_time_begin
    print('Begin Generation')
    generation_time_begin = time()

    if args.sample_number == 1:
        gen_handler(model, primer_seq_list[0], args, args.output_file_path)
    else:
        tqdm_primer_seq_list = tqdm(
            desc='Total samples',
            total=len(primer_seq_list),
            ncols=0
        )
        for i, primer_seq in enumerate(primer_seq_list):
            gen_handler(
                model,
                primer_seq,
                args,
                f'{args.output_file_path}_{i+1}'
            )
            # print(f'generated {args.output_file_path}_{i+1}')
            tqdm_primer_seq_list.update()
        tqdm_primer_seq_list.close()

    generation_time = time() - generation_time_begin
    total_time = overhead_time + primer_process_time + generation_time
    print(f'overhead_time:{overhead_time:g} \
        primer_process_time:{primer_process_time:g} \
        generation_time:{generation_time:g} \
        total_time:{total_time:g}'
    )

    return 0


if __name__ == '__main__':
    EXIT_CODE = main()
    exit(EXIT_CODE)
