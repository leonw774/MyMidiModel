from argparse import ArgumentParser, Namespace
from traceback import format_exc

import torch

from util.tokens import b36str2int, BEGIN_TOKEN_STR, END_TOKEN_STR
from util.midi import midi_to_text_list, piece_to_midi
from util.corpus import CorpusReader, text_list_to_array
from util.model import MyMidiTransformer, generate_sample

def read_args():
    parser = ArgumentParser()
    parser.add_argument(
        'model_file_path',
        type=str
    )
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
        'output_file_path',
        type=str,
        help='The path of generated MIDI file(s).\n\
            If SAMPLE_NUMBER == 1 (default), the path will be "{OUTPUT_FILE_PATH}.mid".\n\
            If SAMPLE_NUMBER > 1, the paths will be "{OUTPUT_FILE_PATH}_{i}.mid", where i is 1 ~ SAMPLE_NUMBER.'
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
            The generation ends when its trying times pass this limit.'
    )
    parser.add_argument(
        '--print-exception',
        action='store_true',
        help='When model fail to generate next toke that satisfy the rule. Print out the exception message.'
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
        midi = piece_to_midi(' '.join(gen_text_list), model.vocabs.paras['nth'], ignore_panding_note_error=True)
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
            except:
                print('Failed to parse primer as corpus: following exception raised')
                print(format_exc())
                exit()

        if args.unit == 'measure':
            m_count = 0
            end_index = 0
            for i, text in enumerate(primer_text_list):
                if text[0] == 'M':
                    m_count += 1
                if m_count > args.primer_length:
                    end_index = i
                    break
            if end_index == 0:
                raise ValueError('Primer piece is shorter than PRIMER_LENGTH in MEASURE unit.')
            primer_text_list = primer_text_list[:end_index]

        elif args.unit == 'nth':
            cur_time = 0
            cur_measure_onset = 0
            cur_measure_length = 0
            end_index = 0
            for i, text in enumerate(primer_text_list):
                typename = text[0]
                if typename == 'M':
                    numer, denom = (b36str2int(x) for x in text[1:].split('/'))
                    cur_measure_onset += cur_measure_length
                    cur_time = cur_measure_onset
                    cur_measure_length = round(nth * numer / denom)

                elif typename == 'P':
                    position = b36str2int(text[1:])
                    cur_time = position + cur_measure_onset

                elif typename == 'T':
                    if ':' in text[1:]:
                        tempo, position = (b36str2int(x) for x in text[1:].split(':'))
                        cur_time = position + cur_measure_onset

                elif typename == 'N':
                    is_cont = (text[1] == '~')
                    if is_cont:
                        note_attr = tuple(b36str2int(x) for x in text[3:].split(':'))
                    else:
                        note_attr = tuple(b36str2int(x) for x in text[2:].split(':'))
                    if len(note_attr) == 5:
                        cur_time = note_attr[4] + cur_measure_onset

                elif typename == 'S':
                    shape_string, *other_attr = text[1:].split(':')
                    note_attr = tuple(b36str2int(x) for x in other_attr)
                    if len(note_attr) == 5:
                        cur_time = note_attr[4] + cur_measure_onset

                if cur_time > args.primer_length:
                    end_index = i
            if end_index == 0:
                raise ValueError('Primer piece is shorter than PRIMER_LENGTH in NTH unit.')
            primer_text_list = primer_text_list[:end_index]

        else: # args.unit == 'token'
            if len(primer_text_list) < args.primer_length:
                raise ValueError('Primer piece is shorter than PRIMER_LENGTH in TOKEN unit.')
            primer_text_list = primer_text_list[:args.primer_length]

        primer_seq = text_list_to_array(primer_text_list, vocabs=model.vocabs)

    if args.sample_number == 1:
        gen_handler(model, primer_seq, args, args.output_file_path)
    else:
        for i in range(1, args.sample_number+1):
            gen_handler(model, primer_seq, args, f'{args.output_file_path}_{i}')

    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
