from argparse import ArgumentParser, Namespace
import os
import subprocess
import tempfile
from traceback import format_exc

import numpy as np
import torch
from miditoolkit import MidiFile

from util.tokens import BEGIN_TOKEN_STR, END_TOKEN_STR
from util.midi import midi_to_piece, piece_to_midi, get_first_k_measures, get_first_k_nths
from util.corpus_reader import CorpusReader
from util.corpus import text_list_to_array, to_corpus_file_path, to_paras_file_path, dump_corpus_paras, get_full_array_string
from util.model import MyMidiTransformer
from util.generation import generate_piece

def read_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--primer', '-p',
        type=str,
        default=None,
        help='A MIDI or a single-piece corpus file that used as the primer in conditional generation.\n\
            If the extension is not "*.mid" or "*.midi", \
                the program will try to parse it as a corpus text file containing only one piece.\n\
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
        help='How many sample will be generated. Default is %(default)s.'
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
        '--temperature', '-t',
        type=float,
        default=1.0,
        help='Control the temperature of softmax before multinomial sampling. Default is %(default)s.'
    )
    parser.add_argument(
        '--no-prob-adjustment',
        action='store_true'
    )
    parser.add_argument(
        '--nucleus-sampling-threshold', '--nu',
        type=float,
        default=1.0,
        help='The probability threshold nuclues sampling. Default is %(default)s.'
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
            temperature=args.temperature,
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


def main():
    args = read_args()

    # device
    if not args.use_device.startswith('cuda') and args.use_device != 'cpu':
        raise ValueError(f'Bad device name {args.use_device}')
    if not torch.cuda.is_available():
        print('--use-device is set to \'cuda\' but found no CUDA device. Changed to CPU.')
        args.use_device = 'cpu'

    # model
    model = torch.load(args.model_file_path, map_location=torch.device(args.use_device))
    assert isinstance(model, MyMidiTransformer)
    nth = model.vocabs.paras['nth']
    if args.max_generation_step == -1:
        args.max_generation_step = model.max_seq_length

    if args.output_file_path.endswith('.mid'):
        args.output_file_path = args.output_file_path[:-4]

    # primer
    primer_seq = None
    if args.primer is not None:
        primer_text_list = []

        if not os.path.exists(args.primer):
            raise FileNotFoundError()

        if os.path.isfile(args.primer):
            primer_piece = midi_to_piece(MidiFile(args.primer), **model.vocabs.paras)

            if len(model.vocabs.bpe_shapes_list) > 0 and args.primer_length > 0:
                # model use BPE, then apply it
                with tempfile.TemporaryDirectory() as tmp_in_corpus_dir_path:
                    # make tmp corpus
                    with open(to_corpus_file_path(tmp_in_corpus_dir_path), 'w+', encoding='utf8') as tmp_corpus_file:
                        tmp_corpus_file.write(primer_piece + '\n')
                    with open(to_paras_file_path(tmp_in_corpus_dir_path), 'w+', encoding='utf8') as tmp_paras_file:
                        tmp_paras_file.write(dump_corpus_paras(model.vocabs.paras))
                    # make tmp shape_vocab
                    tmp_shape_vocab_file_path = os.path.join(tmp_in_corpus_dir_path, 'shape_vocab')
                    with open(tmp_shape_vocab_file_path, 'w+', encoding='utf8') as shape_vocab_file:
                        shape_vocab_file.write('\n'.join(model.vocabs.bpe_shapes_list) + '\n')
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
                        subprocess.run(apply_args, check=True, stdout=subprocess.DEVNULL)
                        # get content from output
                        with open(to_corpus_file_path(tmp_out_corpus_dir_path), 'r', encoding='utf8') as tmp_out_corpus:
                            merged_piece = tmp_out_corpus.readline() # get first piece
                        primer_text_list = merged_piece.split(' ')
            else:
                primer_text_list = primer_piece.split(' ')

        else: # os.path.isdir(args.primer):
            # we expect the corpus file has same midi parameters as the model
            try:
                with CorpusReader(args.primer) as corpus_reader:
                    piece = next(iter(corpus_reader)) # get first piece
                primer_text_list = piece.split()
            except Exception as e:
                print('Failed to parse primer as corpus: following exception raised')
                raise e

        if args.unit == 'measure':
            primer_text_list = get_first_k_measures(primer_text_list, args.primer_length)

        elif args.unit == 'nth':
            primer_text_list = get_first_k_nths(primer_text_list, nth, args.primer_length)

        else: # args.unit == 'token'
            if len(primer_text_list) < args.primer_length:
                raise ValueError(f'primer_text_list has less than primer_length={args.primer_length} tokens.')
            primer_text_list = primer_text_list[:args.primer_length]

        # turn primer text list into array
        primer_seq = text_list_to_array(primer_text_list, vocabs=model.vocabs)
        primer_seq = np.expand_dims(primer_seq, axis=0).astype(np.int32)
        primer_seq = torch.from_numpy(primer_seq)

    if args.sample_number == 1:
        gen_handler(model, primer_seq, args, args.output_file_path)
    else:
        for i in range(1, args.sample_number+1):
            gen_handler(model, primer_seq, args, f'{args.output_file_path}_{i}')
            print(f'generated {args.output_file_path}_{i}')

    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
