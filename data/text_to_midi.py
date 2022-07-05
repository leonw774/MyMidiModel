import sys
from midi_util import text_to_midi, file_to_paras_and_pieces

filename = sys.argv[1]

paras, pieces = file_to_paras_and_pieces(filename)
print(paras)
# decode
for i, text in enumerate(pieces):
    if len(text) == 0:
        continue
    midi_obj = text_to_midi(text, paras['nth'])
    midi_obj.dump(f'{filename}_restored_{i}.mid')
    print(f'{filename}_restored_{i}.mid')
