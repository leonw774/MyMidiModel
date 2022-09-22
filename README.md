# MyMidiModel

## Overall

1. Make a python 3.7 or higher virtual environment, then run `pip3 install -r requirements.txt`.
2. Make you own copy of config files (e.g.: `./config/midi/my_setting.sh`) if you want to make some changes to the settings.
3. Run `./pipeline.sh {midi_preprocess_config_filename} {bpe_config_filename} {training_config_filename}` to do everything for model training at once.
   - You need to add `--use-existed` at the end of the command to tell `pipeline.sh` not to overwrite existing data.
   - The config files are placed in `./configs/midi`, `./configs/bpe` and `./configs/train`.
4. Thid line is reserved for the method to use the trained model to generate music samples.

## Configuration Files

Details about configuration setting.

## Corpus Structure

A complete "corpus" is directory containing at least 5 files in the following list. It can contain more files/directories which would be described in later section.

- `corpus`
- `paras`
- `pathlist`
- `vocabs.json`
- `arrays.npz`

Other possible files and directories:

- `stats`
- `shape_vocab`
- `arrays`
- `text_to_array_debug.txt`

## Multinote BPE

## Model Structure

We train model with a corpus (or more). The model is stored as a "pickled" python object. A model contains:

- design not finished yet
