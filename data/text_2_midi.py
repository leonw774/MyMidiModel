import sys
from midi_util import text_2_midi, text_2_paras_and_pieces

filename = sys.argv[1]

with open(filename, 'r', encoding='utf8') as f:
    texts = f.read()
paras, pieces = text_2_paras_and_pieces(texts)
print(paras)
# decode
for i, text in enumerate(pieces):
    if len(text) == 0:
        continue
    midi_obj = text_2_midi(text, paras['nth'])
    midi_obj.dump(f'restored_{i}.mid')
    print(f'restored_{i}.mid')
