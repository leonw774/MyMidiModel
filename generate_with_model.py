from argparse import ArgumentParser, Namespace
import os
import subprocess
import tempfile
from time import time
from traceback import format_exc

import numpy as np
import torch
from miditoolkit import MidiFile

from util.tokens import BEGIN_TOKEN_STR, END_TOKEN_STR
from util.midi import midi_to_piece, piece_to_midi, get_first_k_measures, get_first_k_nths
from util.corpus_reader import CorpusReader
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
            that would be used as the primer in conditional generation.\n\
            If the extension is "*.mid" or "*.midi", I will try to parse it as MIDI.\n\
            Otherwise, I will try to parse it as a list of paths to midi files, separated by newlines.\n\
            If this option is not set, unconditional generation will be performed.'
    )
    parser.add_argument(
        '--primer-length', '-l',
        type=int,
        default=0,
        help='How long from the start the primer music should be used. Default is %(default)s.\
            The unit of these number is controlled with "--unit" option.\n\
            When --unit is "nth", the the length is counted with the token\'s start time.'
    )
    parser.add_argument(
        '--unit',
        choices=['measure', 'nth', 'token'],
        default='measure',
        help='Specify the unit of PRIMER_LENGTH. \
            If use "measure", primer will be the first PRIMER_LENGTH measures of the piece no matter how long is actually is.\n\
            If use "nth", the length of a "nth"-note is the unit. \
            The value of "nth" is determined by the setting of the dataset on which the model trained.\n\
            If use "tokens", primer will be the first PRIMER_LENGTH tokens of the piece.'
    )
    parser.add_argument(
        '--sample-number', '-n',
        type=int,
        default=1,
        help='How many sample will be generated. Default is %(default)s\
            Set it as -1 to use all primers in the primer list if provided.'
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
        help='The maximum TOKENS in each sample would be generated. Default is models max sequence length.'
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
        '--nucleus-sampling-threshold', '--nu',
        type=float,
        nargs='+',
        default=[1.0],
        help='The probability threshold of nucleus sampling. Default is %(default)s.'
    )
    parser.add_argument(
        '--try-count-limit',
        type=int,
        default=100,
        help='The model may fail to generate next token that satisfy the rule.\
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
        help='When model fail to generate next toke that satisfy the rule. Print out the exception message.'
    )
    parser.add_argument(
        '--no-tqdm',
        action='store_true',
        help='No tqdm progress bar when generating samples.'
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
        help='The path of generated MIDI file(s).\n\
            If SAMPLE_NUMBER == 1 (default), the path will be "{OUTPUT_FILE_PATH}.mid".\n\
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
            nucleus_sampling_threshold=args.nucleus_sampling_threshold,
            print_exception=args.print_exception,
            show_tqdm=(not args.no_tqdm)
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


def primer_file_to_test_list(primer_path: str, primer_length: int, length_unit: str, vocabs, worker_number: int = 1) -> list:
    try:
        primer_piece = midi_to_piece(MidiFile(primer_path), **vocabs.paras)
    except Exception:
        return []
    primer_text_list = primer_piece.split(' ')
    if length_unit == 'measure':
        primer_text_list = get_first_k_measures(primer_text_list, primer_length)
    elif length_unit == 'nth':
        primer_text_list = get_first_k_nths(primer_text_list, vocabs.paras['nth'], primer_length)
    primer_text_list.append(END_TOKEN_STR)
    primer_piece = ' '.join(primer_text_list)

    if len(vocabs.bpe_shapes_list) > 0 and primer_length > 0:
        # model use BPE, then apply it
        with tempfile.TemporaryDirectory() as tmp_in_corpus_dir_path:
            # make tmp corpus
            with open(to_corpus_file_path(tmp_in_corpus_dir_path), 'w+', encoding='utf8') as tmp_corpus_file:
                tmp_corpus_file.write(primer_piece + '\n')
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
                    merged_piece = tmp_out_corpus.readline().strip() # get first piece
                primer_text_list = merged_piece.split(' ')
    else:
        primer_text_list = primer_piece.split(' ')

    if primer_text_list[-1] == END_TOKEN_STR:
        primer_text_list.pop()

    if length_unit == 'token':
        if len(primer_text_list) < primer_length:
            raise ValueError(f'primer_text_list has less than primer_length={primer_length} tokens.')
        primer_text_list = primer_text_list[:primer_length]

    return primer_text_list


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
            raise FileNotFoundError()

        if args.primer.endswith('.mid') or args.primer.endswith('.midi'):
            assert args.sample_number > 0
            primer_text_list = primer_file_to_test_list(args.primer, args.primer_length, args.unit, model.vocabs, args.workers)
            primer_text_list_list = [primer_text_list] * args.sample_number
        else: # we guess it is a list of paths to midi files
            primer_path_list = open(args.primer, 'r', encoding='utf8').readlines()
            if args.sample_number == -1:
                args.sample_number = len(primer_path_list)
            else:
                primer_path_list = primer_path_list[:args.sample_number]
            primer_path_list = [p.strip() for p in primer_path_list]
            assert all(p.endswith('.mid') or p.endswith('.midi') for p in primer_path_list)
            assert all(os.path.isfile(p) for p in primer_path_list)
            primer_text_list_list = [
                primer_file_to_test_list(pp, args.primer_length, args.unit, model.vocabs, args.workers)
                for pp in primer_path_list[:args.sample_number]
            ]
            primer_text_list_list = [
                text_list for text_list in primer_text_list_list
                if len(text_list) > 0
            ]
            if len(primer_text_list_list) != 1 or args.sample_number != 1:
                if len(primer_text_list_list) < args.sample_number:
                    print(
                        f'The number of primer paths in the list is {len(primer_path_list)}, \
                        less than required sample number ({args.sample_number}). \
                        Will only generate {len(primer_path_list)} pieces.'
                    )
                    args.sample_number = len(primer_path_list)
                else:
                    print(f'The number of primer paths in the list is {len(primer_path_list)}, \
                        greater than or equal to required sample number ({args.sample_number}). \
                        Will used first {args.sample_number} to generate pieces.'
                    )
                    print('Hint: Set sample number as -1 to use all primers in the primer list.')

        for primer_text_list in primer_text_list_list:
            # turn primer text list into array
            primer_seq = text_list_to_array(primer_text_list, vocabs=model.vocabs)
            if model.permute_track_number:
                primer_seq = permute_track_number(primer_seq, model.vocabs.track_numbers.size)
            primer_seq = np.expand_dims(primer_seq, axis=0).astype(np.int32)
            primer_seq = torch.from_numpy(primer_seq)
            primer_seq_list.append(primer_seq)


    primer_process_time = time() - primer_process_time_begin
    generation_time_begin = time()

    if args.sample_number == 1:
        gen_handler(model, primer_seq_list[0], args, args.output_file_path)
    else:
        for i, primer_seq in enumerate(primer_seq_list):
            gen_handler(model, primer_seq, args, f'{args.output_file_path}_{i+1}')
            print(f'generated {args.output_file_path}_{i+1}')

    generation_time = time() - generation_time_begin
    print(f'overhead_time:{overhead_time}\tprimer_process_time:{primer_process_time}\tgeneration_time:{generation_time}')

    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
