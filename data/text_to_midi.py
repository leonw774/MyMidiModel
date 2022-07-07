import sys
from midi_util import piece_to_midi, file_to_paras_and_pieces_iterator

filename = sys.argv[1]

with open(filename, 'r', encoding='utf8') as infile:
    paras, piece_iterator = file_to_paras_and_pieces_iterator(infile)
    print(paras)
    # decode
    for i, piece in enumerate(piece_iterator):
        if len(piece) == 0:
            continue
        midi_obj = piece_to_midi(piece, paras['nth'])
        midi_obj.dump(f'{filename}_restored_{i}.mid')
        print(f'{filename}_restored_{i}.mid')
