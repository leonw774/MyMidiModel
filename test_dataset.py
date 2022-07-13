import io

import numpy as np
from torch.utils.data import random_split, DataLoader

from model.dataset import MidiDataset, collate_mididataset
from data import numpy_to_text_list


dataset = MidiDataset(
    data_dir_path='data/data/test_midis',
    max_sample_length=24, # to be printed on screen, so small
    sample_stride=1,
    sample_in_equaltime_sequence=False,
    sample_always_include_head=True,
    permute_notes_in_position=True,
    permute_track_number=True,
    measure_number_augument_range=0
)
vocabs_dict = dataset.vocabs
print('max_sample_length', dataset.max_sample_length)
print('sample_stride', dataset.sample_stride)
print('sample_in_equaltime_sequence', dataset.sample_in_equaltime_sequence)
print('sample_always_include_head', dataset.sample_always_include_head)
print('permute_notes_in_position', dataset.permute_notes_in_position)
print('permute_track_number', dataset.permute_track_number)
print('measure_number_augument_range', dataset.measure_number_augument_range)

def print_array(sample_array, ets_sep_indices, name):
    array_text_byteio = io.BytesIO()
    np.savetxt(array_text_byteio, sample_array, fmt='%d')
    array_savetxt_list = array_text_byteio.getvalue().decode().split('\n')
    text_list = numpy_to_text_list(sample_array, vocabs=vocabs_dict)
    if ets_sep_indices is None:
        ets_sep_indices = []
    print(name)
    print(' evt  dur  vel  trn  ins  tmp  pos  mea')
    print(
        '\n'.join([
            ' '.join([f'{s:>4}' for s in a.split()])
            + f' {z:<13}' + (' ETS_SEP' if i in ets_sep_indices else '')
            for i, (a, z) in enumerate(zip(array_savetxt_list, text_list))
        ])
    )
    print('ets_sep_indices', ets_sep_indices)

# print('TEST RANDOM SPLIT')

# train_len = int(len(dataset) * 0.8)
# vel_len = len(dataset) - train_len
# train_set, val_set = random_split(dataset, [train_len, vel_len])
# print(len(dataset), len(train_set), len(val_set))

# train_sample, train_ets_sep_indices = train_set[0]
# train0 = train_sample.numpy()
# print_array(train0, train_ets_sep_indices, 'TRAIN[0]')

# val_sample, val_ets_sep_indices = val_set[0]
# val0 = val_sample.numpy()
# print_array(val0, val_ets_sep_indices, 'VAL[0]')

print('TEST DATA CORRECTNESS')

train_dataloader = DataLoader(
    dataset=dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_mididataset
)
batched_samples, batched_ets_sep_indices = next(iter(train_dataloader))
for i, (s, e) in enumerate(zip(batched_samples, batched_ets_sep_indices)):
    print_array(s.numpy(), e, f'BATCHED[{i}]')
