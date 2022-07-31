import io
import logging
import os
import shutil
from argparse import ArgumentParser
from time import localtime, strftime

import numpy as np

from data import *


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--max-sample-length',
        dest='max_sample_length',
        type=int,
        default=4096
    )
    parser.add_argument(
        '--debug',
        action='store_true'
    )
    parser.add_argument(
        '--bpe',
        type=int,
        default=0,
        help='The number of iteration the BPE algorithm does. Default is 0.\
            If the number is integer that greater than 0, BPE will be performed.'
    )
    parser.add_argument(
        'input_file_path',
    )
    parser.add_argument(
        'output_dir_path',
    )

    loglevel = logging.INFO
    logging.basicConfig(
        filename=strftime("logs/make_data-%Y%m%d-%H%M.log", localtime()),
        filemode='w',
        level=loglevel
    )
    console = logging.StreamHandler()
    console.setLevel(loglevel)
    logging.getLogger().addHandler(console)

    args = parser.parse_args()

    with open(args.input_file_path, 'r', encoding='utf8') as infile:
        paras, piece_iterator = file_to_paras_and_piece_iterator(infile)
        assert len(piece_iterator) > 0, f'empty corpus {args.input_file_path}'

        if not os.path.isdir(args.output_dir_path):
            os.makedirs(args.output_dir_path)

        if args.bpe > 0:
            logging.info('Use BPE')
            root_name, ext = os.path.splitext(args.input_file_path)
            bpe_input_file_path = root_name + '_bpe' + ext
            # prepare parameter yaml for bpe program to append to
            with open(bpe_input_file_path, 'w+', encoding='utf8') as bpe_input_file:
                bpe_input_file.write(make_para_yaml(paras))
            bpe_shapes_list = do_bpe(piece_iterator, paras, args.bpe, bpe_input_file_path)
            buildvocab_input_file_path = bpe_input_file_path

        else:
            bpe_shapes_list = []
            buildvocab_input_file_path = args.input_file_path

    logging.info('Begin build vocabs')
    with open(buildvocab_input_file_path, 'r', encoding='utf8') as infile:
        paras, piece_iterator = file_to_paras_and_piece_iterator(infile)
        assert len(piece_iterator) > 0, f'empty corpus: {buildvocab_input_file_path}'

        vocab_dicts, summary_string = build_vocabs(piece_iterator, paras, args.output_dir_path, args.max_sample_length, bpe_shapes_list)
        logging.info(summary_string)

        start_time = time()
        logging.info('Begin write npy')

        npy_dir_path = os.path.join(args.output_dir_path, 'arrays')
        if os.path.exists(npy_dir_path):
            shutil.rmtree(npy_dir_path)
            print(f'Find existing {npy_dir_path}, removed.')
        if os.path.exists(os.path.join(args.output_dir_path, 'arrays.zip')):
            os.remove(os.path.join(args.output_dir_path, 'arrays.zip'))
            print(f'Find existing {os.path.join(args.output_dir_path, "arrays.zip")}, removed.')
        if os.path.exists(os.path.join(args.output_dir_path, 'arrays.npz')):
            os.remove(os.path.join(args.output_dir_path, 'arrays.npz'))
            print(f'Find existing {os.path.join(args.output_dir_path, "arrays.npz")}, removed.')

        os.makedirs(npy_dir_path)
        for i, p in enumerate(piece_iterator):
            array = text_list_to_numpy(p.split(), vocab_dicts)
            np.save(os.path.join(npy_dir_path, str(i)), array)

        # zip all the npy files into one file with '.npz' extension
        shutil.make_archive(npy_dir_path, 'zip', root_dir=npy_dir_path)
        os.rename(os.path.join(args.output_dir_path, 'arrays.zip'), os.path.join(args.output_dir_path, 'arrays.npz'))
        # delete the npys
        shutil.rmtree(npy_dir_path)

        # for debugging
        if args.debug:
            p0 = next(iter(piece_iterator))
            array_data = text_list_to_numpy(p0.split(), vocab_dicts)
            array_text_byteio = io.BytesIO()
            debug_txt_path = os.path.join(args.output_dir_path, 'text_list_to_numpy_debug.txt')
            print(f'Write debug file: {debug_txt_path}')
            np.savetxt(array_text_byteio, array_data, fmt='%d')
            array_text_list = array_text_byteio.getvalue().decode().split('\n')
            original_text_list = p0.split()
            reconst_text_list = numpy_to_text_list(array_data, vocab_dicts)
            text_data_reconst_text_list = [
                (t, *d.split(), rt)
                for t, d, rt in zip(original_text_list, array_text_list, reconst_text_list)
            ]
            with open(debug_txt_path, 'w+', encoding='utf8') as f:
                f.write(
                    f"{'token_text':<50}{'evt':>5}{'pit':>5}{'dur':>5}{'vel':>5}"
                    + f"{'trn':>5}{'ins':>5}{'tmp':>5}{'pos':>5}{'mea':>5}     {'reconstructed_text'}\n")
                formated_lines = [
                    f"{line[0]:<50}{line[1]:>5}{line[2]:>5}{line[3]:>5}{line[4]:>5}{line[5]:>5}"
                    + f"{line[6]:>5}{line[7]:>5}{line[8]:>5}{line[9]:>5}     {line[10]}"
                    for line in text_data_reconst_text_list
                ]
                f.write('\n'.join(formated_lines))

    logging.info('npys write time: %.3f', time()-start_time)
    logging.info('make_data.py exited')
