from argparse import ArgumentParser, Namespace
from functools import partial
import os
import subprocess
import tempfile
from time import time
from tqdm import tqdm
from traceback import format_exc
from typing import List

import numpy as np
import torch
from miditoolkit import MidiFile

from util.tokens import BEGIN_TOKEN_STR, END_TOKEN_STR
from util.midi import midi_to_piece, piece_to_midi, get_first_k_measures, get_first_k_nths
# from util.corpus_reader import CorpusReader
from util.corpus import text_list_to_array, to_corpus_file_path, to_paras_file_path, dump_corpus_paras, get_full_array_string
from util.model import MyMidiTransformer
from util.generation import generate_piece, permute_track_number

def read_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--primer', '-p',
        type=str,
        default=None,
        help='A MIDI file, or a text file containing a list of MIDI file paths \
            that would be used as the primer in conditional generation. \
            If the extension is "*.mid" or "*.midi", I will try to parse it as MIDI. \
            Otherwise, I will try to parse it as a list of paths to midi files, separated by newlines. \
            If this option is not set, unconditional generation will be performed.'
    )
    parser.add_argument(
        '--primer-length', '-l',
        type=int,
        default=0,
        help='How long from the start the primer music should be used. Default is %(default)s.\
            The unit of these number is controlled with "--unit" option. \
            When --unit is "nth", the the length is counted with the token\'s start time.'
    )
    parser.add_argument(
        '--unit', '-u',
        choices=['measure', 'nth', 'token'],
        default='measure',
        help='Specify the unit of PRIMER_LENGTH. \
            If use "measure", primer will be the first PRIMER_LENGTH measures of the piece no matter how long it actually is. \
            If use "nth", the length of a "nth"-note is the unit. \
            The value of "nth" is determined by the setting of the dataset on which the model trained. \
            If use "tokens", primer will be the first PRIMER_LENGTH tokens of the piece.'
    )
    parser.add_argument(
        '--sample-number', '-n',
        type=int,
        default=1,
        help='How many sample will be generated. Default is %(default)s. \
            If no primer used, it simply generate SAMPLE_NUMBER samples. \
            If primer is a midi file, it generate SAMPLE_NUMBER samples using that primer repeatly. \
            If primer is list of midi file paths, it generate samples with the first SAMPLE_NUMBER primers. \
            Set it to 0 to use all primers in the list.'
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
        default=-1,
        help='The maximum TOKENS in each sample would be generated. Default is the model\'s max sequence length.'
    )
    parser.add_argument(
        '--no-prob-adjustment',
        action='store_true'
    )
    parser.add_argument(
        '--softmax-temperature', '-t',
        type=float,
        nargs='+',
        default=[1.0],
        help='Control the temperature of softmax before multinomial sampling. Default is %(default)s.'
    )
    parser.add_argument(
        '--sample-function', '-f',
        type=str,
        nargs='?',
        choices=('none', 'top-k', 'top-p', 'nucleus'),
        const='none',
        default='none',
        help='The sample function to used. Choice "top-p" is the same as "nucleus". Default is %(default)s'
    )
    parser.add_argument(
        '--sample-threshold', '--threshold', '-e',
        type=float,
        nargs='+',
        default=[1.0],
        help='The probability threshold of nucleus sampling. Default is %(default)s.'
    )
    parser.add_argument(
        '--try-count-limit',
        type=int,
        default=100,
        help='The model may fail to generate next token that satisfy the format rule.\
            The generation ends when its trying times pass this limit.\
            Default is %(default)s.'
    )
    parser.add_argument(
        '--use-device',
        type=str,
        default='cuda',
        help='What device the model would be on.'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=8,
        help='Number of workers for applying BPE vocabs if used.'
    )
    parser.add_argument(
        '--print-exception',
        action='store_true',
        help='When model fail to generate next token that satisfy the format rule. Print out the exception message.'
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
        type=int,
        default=None
    )
    parser.add_argument(
        'model_file_path',
        type=str
    )
    parser.add_argument(
        'output_file_path',
        type=str,
        help='The path of generated MIDI file(s). \
            If SAMPLE_NUMBER == 1 (default), the path will be "{OUTPUT_FILE_PATH}.mid". \
            If SAMPLE_NUMBER > 1, the paths will be "{OUTPUT_FILE_PATH}_{i}.mid", where i is 1 ~ SAMPLE_NUMBER.'
    )

    return parser.parse_args()


def gen_handler(model: MyMidiTransformer, primer_seq, args: Namespace, output_file_path: str):
    try:
        gen_text_list = generate_piece(
            model,
            steps=args.max_generation_step,
            start_seq=primer_seq,
            softmax_temperature=args.softmax_temperature,
            try_count_limit=args.try_count_limit,
            use_prob_adjustment=(not args.no_prob_adjustment),
            sample_function=args.sample_function,
            sample_threshold=args.sample_threshold,
            print_exception=args.print_exception,
            show_tqdm=(not args.no_sample_tqdm)
        )
        if gen_text_list == BEGIN_TOKEN_STR + ' ' + END_TOKEN_STR:
            print(f'{output_file_path}: generated empty piece. will not output file.')
        else:
            if args.output_text:
                with open(f'{output_file_path}.txt', 'w+', encoding='utf8') as f:
                    f.write(' '.join(gen_text_list))
            if args.output_array_text:
                with open(f'{output_file_path}_array.txt', 'w+', encoding='utf8') as f:
                    f.write(get_full_array_string(text_list_to_array(gen_text_list, model.vocabs), model.vocabs))
            midi = piece_to_midi(' '.join(gen_text_list), model.vocabs.paras['nth'])
            midi.dump(f'{output_file_path}.mid')
    except Exception:
        print('Generation failed becuase the following exception:')
        print(format_exc())


def cut_primer_piece(primer_piece: str, primer_length: int, length_unit: str, nth: int) -> str:
    primer_text_list = primer_piece.split(' ')
    if length_unit == 'measure':
        primer_text_list = get_first_k_measures(primer_text_list, primer_length)
    elif length_unit == 'nth':
        primer_text_list = get_first_k_nths(primer_text_list, nth, primer_length)
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
        Read the midi files, cut them, write them into corpus, apply BPE, return the primers in text_list form
    """
    primer_piece_list: List[str] = []
    for midi_path in tqdm(primer_paths, desc='Encoding midi files to text representation', ncols=0):
        try:
            p = midi_to_piece(MidiFile(midi_path), **vocabs.paras)
            primer_piece_list.append(p)
        except Exception:
            pass
    # apply measure and nth cut
    partial_cut_primer_piece = partial(
        cut_primer_piece,
        primer_length=primer_length, length_unit=length_unit, nth=vocabs.paras['nth']
    )
    primer_piece_list = list(map(partial_cut_primer_piece, primer_piece_list))

    if len(vocabs.bpe_shapes_list) > 0 and primer_length > 0:
        print('Applying BPE')
        # model use BPE, then apply it
        with tempfile.TemporaryDirectory() as tmp_in_corpus_dir_path:
            # make tmp corpus
            with open(to_corpus_file_path(tmp_in_corpus_dir_path), 'w+', encoding='utf8') as tmp_corpus_file:
                tmp_corpus_file.write('\n'.join(primer_piece_list) + '\n')
            with open(to_paras_file_path(tmp_in_corpus_dir_path), 'w+', encoding='utf8') as tmp_paras_file:
                tmp_paras_file.write(dump_corpus_paras(vocabs.paras))
            # make tmp shape_vocab
            tmp_shape_vocab_file_path = os.path.join(tmp_in_corpus_dir_path, 'shape_vocab')
            with open(tmp_shape_vocab_file_path, 'w+', encoding='utf8') as shape_vocab_file:
                shape_vocab_file.write('\n'.join(vocabs.bpe_shapes_list) + '\n')
            # make sure the current working directory is correct
            assert os.path.abspath(os.getcwd()) == os.path.dirname(os.path.abspath(__file__))
            # make sure the program is there and new
            subprocess.run(['make', '-C', './bpe'], check=True, stdout=subprocess.DEVNULL)
            with tempfile.TemporaryDirectory() as tmp_out_corpus_dir_path:
                tmp_out_corpus_file_path = to_corpus_file_path(tmp_out_corpus_dir_path)
                # ./apply_vocab [-log] [-clearLine] inCorpusDirPath outCorpusFilePath shapeVocabularyFilePath
                apply_args = [
                    './bpe/apply_vocab',
                    # '-log',
                    # '-clearLine',
                    tmp_in_corpus_dir_path,
                    tmp_out_corpus_file_path,
                    tmp_shape_vocab_file_path
                ]
                if worker_number > 1:
                    apply_args.append(str(worker_number))
                subprocess.run(apply_args, check=True, stdout=subprocess.DEVNULL)
                # get content from output
                with open(to_corpus_file_path(tmp_out_corpus_dir_path), 'r', encoding='utf8') as tmp_out_corpus:
                    merged_piece_list = [
                        line.strip()
                        for line in tmp_out_corpus.readlines()
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
    if not args.use_device.startswith('cuda') and args.use_device != 'cpu':
        raise ValueError(f'Bad device name {args.use_device}')
    if not torch.cuda.is_available():
        print('--use-device is set to \'cuda\' but found no CUDA device. Changed to CPU.')
        args.use_device = 'cpu'
    assert args.sample_number > 0

    # model
    model = torch.load(args.model_file_path, map_location=torch.device(args.use_device))
    assert isinstance(model, MyMidiTransformer)
    if args.max_generation_step == -1:
        args.max_generation_step = model.max_seq_length

    if args.output_file_path.endswith('.mid'):
        args.output_file_path = args.output_file_path[:-4]

    overhead_time = time() - overhead_time_begin
    primer_process_time_begin = time()

    # primer
    primer_seq_list = []
    if args.primer is  None:
        assert args.sample_number > 0
        primer_seq_list = [None] * args.sample_number
    else:
        print('Processing primer')
        if not os.path.isfile(args.primer):
            print('Primer file not exists')
            raise FileNotFoundError()

        encode_args = {
            'primer_length': args.primer_length,
            'length_unit': args.unit,
            'vocabs': model.vocabs,
            'worker_number': args.workers
        }
        if args.primer.endswith('.mid') or args.primer.endswith('.midi'):
            print('From midi file:', args.primer)
            primer_path_list = [args.primer]
            primer_text_list_list = midi_file_list_to_text_list_list(primer_path_list, **encode_args)
            primer_text_list_list = primer_text_list_list * args.sample_number
        else: # we guess it is a list of paths to midi files
            print('From path list:', args.primer)

            primer_path_list = open(args.primer, 'r', encoding='utf8').readlines()
            print(f'Read {len(primer_path_list)} lines')

            if args.sample_number != 0:
                primer_path_list = primer_path_list[:args.sample_number]
            primer_path_list = [p.strip() for p in primer_path_list]
            assert all(p.endswith('.mid') or p.endswith('.midi') for p in primer_path_list)
            assert all(os.path.isfile(p) for p in primer_path_list)
            print(f'Keep first {len(primer_path_list)} lines with sample number setting to {args.sample_number}')
            print('Hint: Set sample number to 0 to keep all primers in the primer list.')

            primer_text_list_list = midi_file_list_to_text_list_list(primer_path_list, **encode_args)
            assert len(primer_text_list_list) != 0
            print(f'Processed {len(primer_text_list_list)} files successfully.')
            if len(primer_text_list_list) < args.sample_number:
                print(
                    f'Primer number less than designated sample number {args.sample_number}:',
                    'will only generate', len(primer_text_list_list), 'pieces.'
                )
                args.sample_number = len(primer_text_list_list)

        for primer_text_list in primer_text_list_list:
            # turn primer text list into array
            primer_seq = text_list_to_array(primer_text_list, vocabs=model.vocabs)
            if model.permute_track_number:
                primer_seq = permute_track_number(primer_seq, model.vocabs.track_numbers.size)
            primer_seq = np.expand_dims(primer_seq, axis=0).astype(np.int32)
            primer_seq = torch.from_numpy(primer_seq)
            primer_seq_list.append(primer_seq)

    primer_process_time = time() - primer_process_time_begin
    print('Begin Generation')
    generation_time_begin = time()

    if args.sample_number == 1:
        gen_handler(model, primer_seq_list[0], args, args.output_file_path)
    else:
        pbar = tqdm(desc='Total samples', total=len(primer_seq_list), ncols=0)
        for i, primer_seq in enumerate(primer_seq_list):
            gen_handler(model, primer_seq, args, f'{args.output_file_path}_{i+1}')
            # print(f'generated {args.output_file_path}_{i+1}')
            pbar.update()
        pbar.close()

    generation_time = time() - generation_time_begin
    total_time = overhead_time + primer_process_time + generation_time
    print(f'overhead_time:{overhead_time:g} \
        primer_process_time:{primer_process_time:g} \
        generation_time:{generation_time:g} \
        total_time:{total_time:g}'
    )

    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
