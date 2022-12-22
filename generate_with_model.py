from argparse import ArgumentParser, Namespace

import numpy as np
import torch

from util.tokens import BEGIN_TOKEN_STR, END_TOKEN_STR
from util.midi import midi_to_text_list, piece_to_midi, get_first_k_measures, get_first_k_nths
from util.corpus import CorpusReader, text_list_to_array
from util.model import MyMidiTransformer, generate_sample

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
        default=1,
        help='How long from the start the primer music should be used. Default is 1.\
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
        help='How many generated sample will created. Default is 1.'
    )
    parser.add_argument(
        '--output-txt', '-t',
        action='store_true'
    )
    parser.add_argument(
        '--max-generation-step', '--step',
        type=int,
        default=1024,
        help='The maximum TOKENS in each sample would be generated.'
    )
    parser.add_argument(
        '--try-count-limit',
        type=int,
        default=1000,
        help='The model may fail to generate next token that satisfy the rule.\
            The generation ends when its trying times pass this limit.\
            Default is %(default)s'
    )
    parser.add_argument(
        '--use-device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cuda',
        help='What device the model would be on.'
    )
    parser.add_argument(
        '--print-exception',
        action='store_true',
        help='When model fail to generate next toke that satisfy the rule. Print out the exception message.'
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
    gen_text_list = generate_sample(
        model,
        steps=args.max_generation_step,
        start_seq=primer_seq,
        try_count_limit=args.try_count_limit,
        print_exception=args.print_exception
    )
    if gen_text_list == BEGIN_TOKEN_STR + " " + END_TOKEN_STR:
        print(f'{output_file_path}: generated empty piece. will not output file.')
    else:
        midi = piece_to_midi(' '.join(gen_text_list), model.vocabs.paras['nth'], ignore_pending_note_error=True)
        midi.dump(f'{output_file_path}.mid')
        if args.output_txt:
            with open(f'{output_file_path}.txt', 'w+', encoding='utf8') as f:
                f.write(' '.join(gen_text_list))

def main():
    args = read_args()
    # model
    model = torch.load(args.model_file_path)
    assert isinstance(model, MyMidiTransformer)
    nth = model.vocabs.paras['nth']

    # device
    if not args.use_device.startswith('cuda') and args.use_device != 'cpu':
        raise ValueError(f'Bad device name {args.use_device}')
    if not torch.cuda.is_available():
        args.use_device = 'cpu'
    if args.use_device.startswith('cuda'):
        model = model.to(args.use_device)

    # primer
    primer_seq = None
    if args.primer is not None:
        primer_text_list = []
        if args.primer.endswith('.mid') or args.primer.endswith('.midi'):
            primer_text_list = midi_to_text_list(args.primer, **model.vocabs.paras)
        else:
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

        primer_seq = text_list_to_array(primer_text_list, vocabs=model.vocabs)
        primer_seq = np.expand_dims(primer_seq, axis=0)
        primer_seq = torch.from_numpy(primer_seq)

    if args.sample_number == 1:
        gen_handler(model, primer_seq, args, args.output_file_path)
    else:
        for i in range(1, args.sample_number+1):
            gen_handler(model, primer_seq, args, f'{args.output_file_path}_{i}')

    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
