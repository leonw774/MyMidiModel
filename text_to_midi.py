import sys
from data import piece_to_midi, file_to_paras_and_piece_iterator

in_filepath = sys.argv[1]
out_filename = sys.argv[2]

with open(in_filepath, 'r', encoding='utf8') as infile:
    paras, piece_iterator = file_to_paras_and_piece_iterator(infile)
    print(paras)
    # decode
    for i, piece in enumerate(piece_iterator):
        if len(piece) == 0:
            continue
        midi_obj = piece_to_midi(piece, paras['nth'])
        midi_obj.dump(f'{out_filename}_{i}.mid')
        print(f'{out_filename}_{i}.mid')
