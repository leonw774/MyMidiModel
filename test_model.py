import json
import sys
from time import time

from torch import randint

from util.corpus import to_vocabs_file_path
from util.model import MyMidiTransformer, get_seq_mask, calc_losses

if len(sys.argv) == 1:
    data_dir_path = 'data/corpus/test_midis_nth32_r32_d32_v16_t24_200_16'
elif len(sys.argv) == 2:
    data_dir_path = sys.argv[1]
else:
    raise ValueError()

vocabs_dict = json.load(
    open(to_vocabs_file_path(data_dir_path), 'r', encoding='utf8')
)

start_time = time()
model = MyMidiTransformer(
    layers_number=2,
    attn_heads_number=4,
    d_model=32,
    vocabs=vocabs_dict
)

print(model)
print('model construction time:', time()-start_time)

test_seq = randint(0, 16, (4, 8, len(model.embedding_dim)))
test_mask = get_seq_mask(test_seq.shape[1])

print(test_seq.shape)
print(test_mask)

input_seq = test_seq[:-1]
target_seq = test_seq[1:]
target_seq = model.to_output_attrs(test_seq[1:])

start_time = time()
pred = model(input_seq, test_mask)

print([str(a.shape) for a in pred])
print(target_seq.shape)

loss = calc_losses(pred, target_seq)
loss = sum(loss)
loss.backward()
print(loss)
print('model train step time:', time()-start_time)
