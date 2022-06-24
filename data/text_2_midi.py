import sys
from midi_util import text_2_midi, parse_para_yaml

filename = sys.argv[1]

with open(filename, 'r', encoding='utf8') as f:
    texts = f.read()
_, head, body =  texts.split('---')
paras = parse_para_yaml(head)
# decode
for i, text in enumerate(body.split('\n')):
    if len(text) == 0:
        continue
    midi_obj = text_2_midi(text, paras['nth'])
    midi_obj.dump(f'restored_{i}.mid')
    print(i)
