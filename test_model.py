import json

from torch import randint

from util.corpus import get_input_array_debug_string, to_vocabs_file_path
from util.model import MidiTransformerDecoder, get_seq_mask

vocabs_dict = json.load(
    open(to_vocabs_file_path('data/corpus/test_midis_nth96_r32_d96_v4_t24_200_1_posattribute'), 'r', encoding='utf8')
)

model = MidiTransformerDecoder(
    layers_number=2,
    attn_heads_number=4,
    d_model=32,
    vocabs=vocabs_dict
)

print(model)

test_seq = randint(0, 16, (4, 8, len(model._embeddings)))
test_mask = get_seq_mask(test_seq.shape[1])

print(test_seq.shape)
print(test_mask)

out = model(test_seq, test_mask)

print([str(o.shape) for o in out])

target = randint(0, 16, (4, 8, len(model._logits)))

print(target.shape)

loss = model.calc_loss(out, target)

print(loss)