import sys
from midi_util import text_2_midi

filename = sys.argv[1]
nth = int(sys.argv[2])

with open(filename, 'r', encoding='utf8') as f:
    texts = f.read()
for i, text in enumerate(texts.split('\n')):
    if len(text) == 0: continue
    midi_obj = text_2_midi(text, nth)
    midi_obj.dump(f'restored_{i}.mid')
