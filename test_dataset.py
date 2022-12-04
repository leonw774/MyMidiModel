import numpy as np
from torch.utils.data import random_split, DataLoader

from util.corpus import get_input_array_format_string
from util.dataset import MidiDataset, collate_mididataset
# from util.model import MidiTransformerDecoder


dataset = MidiDataset(
    data_dir_path='data/corpus/test_midis_nth96_r32_d96_v4_t24_200_16_posevent',
    max_seq_length=32, # to be printed on screen, so small
    use_permutable_subseq_loss=False,
    permute_mps=True,
    permute_track_number=True,
    pitch_augmentation=3,
    sample_from_start=True
)
vocabs_dict = dataset.vocabs
print('max_seq_length', dataset.max_seq_length)
print('use_permutable_subseq_loss', dataset.use_permutable_subseq_loss)
print('permute_mps', dataset.permute_mps)
print('permute_track_number', dataset.permute_track_number)

print('TEST RANDOM SPLIT')

train_len = int(len(dataset) * 0.8)
vel_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, vel_len])
# print(len(dataset), len(train_set), len(val_set))

train_sample, train_mps_sep_indices = train_set[0]
train0 = train_sample.numpy()
# print('TRAIN[0]')
# print(get_input_array_format_string(train0, train_mps_sep_indices, vocabs_dict))

val_sample, val_mps_sep_indices = val_set[0]
val0 = val_sample.numpy()
# print('VAL[0]')
# print(get_input_array_format_string(val0, val_mps_sep_indices, vocabs_dict))

print('TEST DATA CORRECTNESS')

train_dataloader = DataLoader(
    dataset=train_set,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_mididataset
)
batched_samples, batched_mps_sep_indices = next(iter(train_dataloader))
if len(batched_mps_sep_indices) != 0:
    for i, (s, e) in enumerate(zip(batched_samples, batched_mps_sep_indices)):
        print(f'BATCHED[{i}]')
        print(get_input_array_format_string(s.numpy(), e, vocabs_dict))
else:
    for i, s in enumerate(batched_samples):
        print(f'BATCHED[{i}]')
        print(get_input_array_format_string(s.numpy(), None, vocabs_dict))
