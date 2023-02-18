# MyMidiModel

## Overall

1. Create environment with conda: `conda env create --name {ENV_NAME} --file environment.yml`. You may need to clear cache first by `pip cache purge` and `conda clean --all`.
2. Make you own copy of config files (e.g.: `./config/midi/my_setting.sh`) if you want to make some changes to the settings.
3. Run `./pipeline.sh {midi_preprocess_config_filename} {bpe_config_filename} {training_config_filename}` to do everything from pre-processing to model training at once.
   - You need to add `--use-existed` at the end of the command to tell `pipeline.sh` not to overwrite existing data.
   - The config files are placed in `configs/midi`, `configs/bpe` and `configs/train`.
   - You can recreate our experiment by running `experiment*.sh` files.

## Configuration Files

- Config files in `configs/midi` are parameters for `midi_to_corpus.py` (`MAX_TRACK_NUMBER`, `MAX_DURATION`, etc.)
- Config files in `configs/bpe` are parameters for `bpe/learn_vocabs` (implementation of Multi-note BPE)
- Config files in `configs/train` are parameters for `train.py`

## Corpus Structure

Corpi are located at `data/corpus`. A complete "corpus" is directory containing at least 5 files in the following list. The name of the corpus directory is in the format of

```
{CORPUS_NAME}_nth{NTH}_r{MAX_TRACK_NUMBER}_d{MAX_DURATION}_v{VELOCITY_STEP}_t{TEMPO_MIN}_{TEMPO_MAX}_{TEMPO_STEP}_pos{POSITION_METHOD}_bpe{BPE_ITER}_{SCORE_FUNC}_{MERGE_CONDITION}_{SAMPLE_RATE}
```

if is processed by Multi-note BPE. Otherwise, it is

```
{CORPUS_NAME}_nth{NTH}_r{MAX_TRACK_NUMBER}_d{MAX_DURATION}_v{VELOCITY_STEP}_t{TEMPO_MIN}_{TEMPO_MAX}_{TEMPO_STEP}_pos{POSITION_METHOD}
```

- `corpus`: A text file. Each `\n`-separated line is a text representation of a midi file. This is the "main form" of the representation.
- `paras`: A yaml file that contains parameters of pre-processing used by `midi_to_corpus.py`.
- `pathlist`: A text file. Each `\n`-separated line is the path of midi file corresponding to text representation in `corpus`.
- `vocabs.json`: The vocabulary to be used by the model. The format is defined in `util/vocabs.py`.
- `arrays.npz`: A zip file of numpy array in `.npy` format. Can be accessed by `numpy.load()` and it will return an instance of`NpzFile` class. This is the "final form" of the representation (i.e. include pre-computed positional-contextual encoding) that would be used to train model.

Other possible files and directories:

- `stats`: A directoy that contains statistics about the corpus. The some output of `make_arrays.py` and all the figures by `plot_bpe_log.py` would be end up here.
- `shape_vocab`: A text file created by `bpe/learn_vocab`. It will be read by `make_arrays.py` to help create `vocabs.json`.
- `arrays`: A temporary directory for placing the `.npy` files before they are zipped.
- `make_array_debug.txt`: A text file that shows array content of the first piece in the corpus. Created by `make_arrays.py`.

## Multinote BPE

Stuffs about Multi-note BPE are all in `bpe/`.

Source codes:

- `apply_vocab.cpp`
- `classes.cpp` and `classes.hpp`: Define class of corpus, multi-note, rel-note, etc. And I/O functions.
- `learn_vocab.cpp`
- `functions.cpp` and `functions.hpp`: Other functions and algorithms.

They should compile to:

- `apply_vocab`: Apply merge operations with a known shape list to a corpus file. Output a new corpus file.
- `learn_vocab`: Do Multi-node BPE to a corpus file. Output a new corpus file and a shape list in `shape_vocab`.

## Models

- We train model with a corpus.
- A completed trained model is stored at `models/{DATE_AND_FULL_CONFIG_NAME}/best_model.pt` as a "pickled" python object that would be saved and loaded by `torch.save()` and `torch.load()`.
- Learning rate schedule is hard-coded warmup and linear decay.

## Tools and Scripts

Pythons scripts

- `extract.py`: Extract piece(s) from the given corpus directory into text representation(s), midi file(s), or piano-roll graph(s).
- `generate_with_models.py`: Use trained model to generate midi files, with or without a primer.
- `get_eval_features_of_dataset.py`: Do as per its name. It will sample midi file in a midi dataset. Output results as json file.
- `get_eval_features_of_dataset.py`: Do as per its name. It will use a model to generate midi files at `models/{DATE_AND_FULL_CONFIG_NAME}/eval_samples`. Output results as json file.
- `make_arrays.py`: Generate `vocabs.json` and `arrays.npz` from `corpus`.
- `midi_to_corpus.py`: Pre-process midi files into a "corpus". The parameter would be stored in `paras`. But it would not generate `vocabs.json` and `arrays.npz`, which are generated by `make_arrays.py`.
- `plot_bpe_log.py`: Make figures to visualize the data in the log files that contains the loggings of Multi-note BPE program.
- `train.py`: Train a model from a corpus
- `verify_corpus_equality.py`: Make sure two corpus are representing the same midi files.

Shell scripts

- `experiment_apply_learned_shapes_on_other.sh`
- `experiment_on_bpe_parameters.sh`
- `experiment_on_model_and_training.sh`
- `pipeline.sh`:
  1. Pre-process midi files into a corpus
  2. If `DO_BPE` is "true", then run `bpe/learn_vocab` to create a new merged corpus
  3. Make arrays file for the corpus
  4. Train a model on the corpus
  5. Get evaluation feature for the model and the midi files
