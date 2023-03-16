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
{CORPUS_NAME}_nth{NTH}_r{MAX_TRACK_NUMBER}_d{MAX_DURATION}_v{VELOCITY_STEP}_t{TEMPO_MIN}_{TEMPO_MAX}_{TEMPO_STEP}_bpe{BPE_ITER}_{MERGE_CONDITION}_{SAMPLE_RATE}
```

if is processed by Multi-note BPE. Otherwise, it is

```
{CORPUS_NAME}_nth{NTH}_r{MAX_TRACK_NUMBER}_d{MAX_DURATION}_v{VELOCITY_STEP}_t{TEMPO_MIN}_{TEMPO_MAX}_{TEMPO_STEP}
```

- `corpus`: A text file. Each `\n`-separated line is a text representation of a midi file. This is the "main form" of the representation. Created by `midi_to_corpus.py`.
- `paras`: A yaml file that contains parameters of pre-processing used by `midi_to_corpus.py`. Created by `midi_to_corpus.py`.
- `pathlist`: A text file. Each `\n`-separated line is the path of midi file corresponding to text representation in `corpus`. Created by `midi_to_corpus.py`.
- `vocabs.json`: The vocabulary to be used by the model. The format is defined in `util/vocabs.py`. Created by `make_arrays.py`.
- `arrays.npz`: A zip file of numpy array in `.npy` format. Can be accessed by `numpy.load()` and it will return an instance of `NpzFile` class. This is the "final form" of the representation (i.e. include pre-computed positional-contextual encoding) that would be used to train model. Created by `make_arrays.py`.

Other possible files and directories:

- `stats`: A directoy that contains statistics about the corpus. Some figures outputed by `make_arrays.py` and by `plot_bpe_log.py` would be end up here.
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
- Learning rate schedule is hard-coded warmup and linear decay.
- A completed trained model is stored at `models/{DATE_AND_FULL_CONFIG_NAME}/best_model.pt` as a "pickled" python object that would be saved and loaded by `torch.save()` and `torch.load()`.
- There are two directory under `models/{DATE_AND_FULL_CONFIG_NAME}/`, one is `ckpt` where checkpoint model and generated sample would be placed, the other is `eval_samples` where the evaluation samples generated by `generate_with_model.py` called in `pipeline.sh` would be placed.

## Codes in `util/`

- `corpus.py`
  - Define corpus directory structure and the array form of the representation.
  - Implement piece-to-array and array-to-piece functions.
- `dataset.py`
  - Define `MidiDataset` and the collate function.
- `evaluation.py`
  - Implement functions for features computation and preparing data for features computation.
  - Implement piece-to-feature and midi-to-feature functions.
- `midi.py`
  - Implement midi-to-piece and piece-to-midi functions.
- `model.py`
  - Define `MyMidiTransformer`, inherit from `torch.nn.Module`
  - Implement loss function for the model
  - Implement function for generating sample from model
- `token.py`
  - Define representation tokens and their "main form" (text form)
- `vocabs.py`
  - Define `Vocabs`, a class that record the vocabulary set, vocabulary building configurations and midi preprocessing parameters.
  - Implement vocabulary building function. Input: midi preprocessing parameters and bpe shape list.


### Some terms used in function name

- A **"midi"** expects a `miditoolkit.MidiFile` instance.
- A **"piece"** expects a string that would appears in a `\n`-separated line in `corpus`.
- A **"text list"** expects a list of strings that comes from `piece.split(' ')` or become a piece after `' '.join(text_list)`.
- An **"array"** expects a 2-d numpy array that encoded a piece with respect to a vocabulary set.

## Tools and Scripts

Pythons scripts

- `debug_dataset.py`: Print out the result of dataset related things including random split and dataloader.
- `extract.py`: Extract piece(s) from the given corpus directory into text representation(s), midi file(s), or piano-roll graph(s) in png.
- `generate_with_models.py`: Use trained model to generate midi files, with or without a primer.
- `get_eval_features_of_midis.py`: Do as per its name. It will sample midi files in a directory. Output results as a JSON file `eval_feature_stats.json` right at the directory.
- `make_arrays.py`: Generate `vocabs.json` and `arrays.npz` from `corpus` and `shape_vocab` if it exists.
- `midi_to_corpus.py`: Pre-process midi files into a "corpus". The parameter would be stored in `paras`. It creates `corpus`, `paras`, and `pathlist` in the corpus directory.
- `plot_bpe_log.py`: Make figures to visualize the data in the log files that contains the loggings of Multi-note BPE program.
- `train.py`: Train a model from a corpus
- `verify_corpus_equality.py`: Make sure two corpus are representing the same midi files.

Shell scripts

- `experiment_apply_learned_shapes_to_other.sh`
- `experiment_bpe_parameters.sh`
- `experiment_model_and_training_parameters.sh`
- `pipeline.sh`:
  1. Pre-process training midi files into a corpus with `midi_to_corpus.py`
  2. If `DO_BPE` is "true", then run `bpe/learn_vocab` to create a new merged corpus. After it is done, run `verify_corpus_equality.py` to make sure there are no errors and run `plot_bpe_log.py` to visualize the loggings.
  3. Make arrays file and vocabs file of the corpus with `make_arrays.py`
  4. Train a model on the corpus with `train.py`
  5. Get evaluation features of training midi files the model generated midi files with the combination of `generate_with_models.py` and `get_eval_features_of_midis.py`
