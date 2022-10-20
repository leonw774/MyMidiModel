from argparse import ArgumentParser
from traceback import format_exc

import torch

from util import (
    b36str2int,
    midi_to_text_list,
    piece_to_midi,
    CorpusIterator,
    MidiTransformerDecoder,
    generate_sample
)
from util.corpus import Vocabs, text_list_to_array

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
        help='A MIDI or a piece text file that used as the primer in conditional generation.\n\
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
        '--max-generation-step', '--step',
        type=int,
        default=1024,
        help='The maximum TOKENS in each sample would be generated.'
    )
    parser.add_argument(
        '--output-txt', '-t',
        action='store_true'
    )
    return parser.parse_args()

def main():
    args = read_args()
    # model
    model = torch.load(args.model_file_path)
    assert isinstance(model, MidiTransformerDecoder)
    nth = model.vocabs.paras['nth']

    # primer
    primer_seq = None
    if args.primer is not None:
        primer_text_list = []
        if args.primer.endswith('.mid') or args.primer.endswith('.midi'):
            primer_text_list = midi_to_text_list(args.primer, **model.vocabs.paras)
        else:
            try:
                with CorpusIterator(args.primer) as corpus_iter:
                    piece = next(iter(corpus_iter)) # get first piece
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
                    note_attr = (b36str2int(x) for x in other_attr)
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
        gen_text_list = generate_sample(model, steps=args.max_generation_step, start_seq=primer_seq)
        midi = piece_to_midi(' '.join(gen_text_list), nth, ignore_panding_note_error=True)
        midi.dump(f'{args.output_file_path}.mid')
        if args.output_txt:
            with open(f'{args.output_file_path}.txt', 'w+', encoding='utf8') as f:
                f.write(' '.join(gen_text_list))
    else:
        for i in range(1, args.sample_number+1):
            gen_text_list = generate_sample(model, steps=args.max_generation_step, start_seq=primer_seq)
            midi = piece_to_midi(' '.join(gen_text_list), nth, ignore_panding_note_error=True)
            midi.dump(f'{args.output_file_path}_{i}.mid')
            if args.output_txt:
                with open(f'{args.output_file_path}_{i}.txt', 'w+', encoding='utf8') as f:
                    f.write(' '.join(gen_text_list))
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)