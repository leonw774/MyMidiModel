# MyMidiModel

## Contents

- [To run the code](#to-run-the-code)
- [Configuration Files](#configuration-files-configs)
- [Dataset](#datasets-datamidis)
- [Corpus Structure](#corpus-structure-datacorpus)
- [Multinote BPE](#multinote-bpe-bpe)
- [Model](#model-models)
- [Code](#code-util)
- [Tool Scripts](#tool-scripts)


## To run the code

### Step 1

Create environment with conda: 

``` bash
conda env create --name {ENV_NAME} --file environment.yml
```

You may need to clear cache first by `pip cache purge` and `conda clean --all`.


### Step 2 (Optional)

Make you own copy of config files (e.g.: `./config/model/my_model_setting.sh`) if you want to make some changes to the settings.

The config files are placed in `configs/corpus`, `configs/bpe` and `configs/model`.

### Step 3

Run `./pipeline.sh {corpus_config} {bpe_config} {model_config}` to do everything from pre-processing to model training at once.

You can add `--use-existed` at the end of the command to tell `pipeline.sh` to just use the existing data.

You can recreate our experiment by running the three scripts in `experiment_script`.

``` bash
./experiment_script/data_preproc_and_bpe.sh
./experiment_script/apply_learned_shape_to_other.sh
./experiment_script/full_model_and_ablation.sh snd
./experiment_script/full_model_and_ablation.sh lmd_full
```


## Configuration Files (`configs/`)

- Files in `configs/corpus` set parameters for `midi_to_corpus.py` and `make_arrays.py` (`MAX_TRACK_NUMBER`, `MAX_DURATION`, etc.)

- Files in `configs/bpe` set parameters for `bpe/learn_vocabs` (implementation of Multi-note BPE).

- Files in `configs/model` set parameters for `train.py` and `evaluate_model*`.

- Files in `configs/split` contain lists of paths, relative to each dataset root, of midi files to be used as test set and validation set of the datasets. Their path are referenced by variable `TEST_PATHS_FILE` and `VALID_PATHS_FILE` in files of `configs/corpus`.

- Files in `configs/eval_midi_to_piece_paras` store parameters for `evaluate_model.sh` and `evaluate_model_wrapper.py`. Their path are referenced by variable `EVAL_MIDI_TO_PIECE_PARAS_FILE` in files of `configs/model`. Set the variable to empty if default is to be used. If you want a different parameter you can make a new file and reference the path.


## Dataset (`data/midis/`)

The datasets we used, SymphonyNet_Dataset and lmd_full, are expected to be found under `data/midis`. However, the path `midi_to_corpus.py` would be looking is the `MIDI_DIR_PATH` variables set in the the corpus configuration file. So it could be in any place you want. Just set the path right.


## Corpus Structure (`data/corpus/`)

Corpi are located at `data/corpus/`. A complete "corpus" is directory containing at least 5 files in the following list.

- `corpus`: A text file. Each `\n`-separated line is a text representation of a midi file. This is the "main form" of the representation. Created by `midi_to_corpus.py`.

- `paras`: A yaml file that contains parameters of pre-processing used by `midi_to_corpus.py`. Created by `midi_to_corpus.py`.

- `pathlist`: A text file. Each `\n`-separated line is the path, relative to project root, of midi file corresponding to the text representation in `corpus`. Created by `midi_to_corpus.py`.
  - Note that a corpus include all processable, uncorrupted midi file, including the test and validation files. The split of test and validation happens at training and evaluating stage.

- `vocabs.json`: The vocabulary to be used by the model. The format is defined in `util/vocabs.py`. Created by `make_arrays.py`.

- `arrays.npz`: A zip file of numpy arrays in `.npy` format. Can be accessed by `numpy.load()` and it will return an instance of `NpzFile` class. This is the "final form" of the representation (i.e. include pre-computed positional encoding) that would be used to train model. Created by `make_arrays.py`.


Other possible files and directories are:

- `stats/`: A directoy that contains statistics about the corpus. Some figures outputed by `make_arrays.py` and by `plot_bpe_log.py` would be end up here.

- `shape_vocab`: A text file created by `bpe/learn_vocab`. If exist, it will be read by `make_arrays.py` to help create `vocabs.json`.

- `arrays/`: A temporary directory for placing the `.npy` files before they are zipped.

- `make_array_debug.txt`: A text file that shows array content of the first piece in the corpus. Created by `make_arrays.py`.


## Multinote BPE (`bpe/`)

Stuffs about Multi-note BPE are all in `bpe/`.

Source codes:

- `apply_vocab.cpp`

- `classes.cpp` and `classes.hpp`: Define class of corpus, multi-note, rel-note, etc. And I/O functions.

- `learn_vocab.cpp`

- `functions.cpp` and `functions.hpp`: Other functions and algorithms.


They should compile to two binaries with `make -C bpe all`:

- `apply_vocab`: Apply merge operations with a known shape list to a corpus file. Output a new corpus file.

- `learn_vocab`: Do Multi-node BPE to a corpus file. Output a new corpus file and a shape list in `shape_vocab`.


## Model (`models/`)

- Models are created by `train.py`.

- Learning rate schedule is hard-coded warmup and linear decay.

- A completed trained model is stored at `models/{DATE_AND_FULL_CONFIG_NAME}/best_model.pt` as a "pickled" python object that would be saved and loaded by `torch.save()` and `torch.load()`.

- Two directories are under `models/{DATE_AND_FULL_CONFIG_NAME}/`
  - `ckpt/` is where checkpoint model and generated sample would be placed
  - `eval_samples/` is where the evaluation samples generated by `generate_with_model.py` called in `evaluate_model.sh` would be placed.

- A file `models/{DATE_AND_FULL_CONFIG_NAME}/test_paths` containing all paths to the test files would be created when running `evaluate_model.sh`.


## Codes (`util/`)

- `corpus.py`
  - Define corpus directory structure and the array form of the representation.
  - Contain piece-to-array and array-to-piece functions.

- `corpus_reader.py`
  - Define corpus reader class

- `dataset.py`
  - Define `MidiDataset` class and the collate function.

- `evaluation.py`
  - Contain functions for features computation and preparing data for features computation.
  - Contain piece-to-feature and midi-to-feature functions.
  - Contain funtion for aggregating features from all midis.

- `generation.py`
  - Contain functions for generating using model.

- `midi.py`
  - Contain the midi-to-piece and piece-to-midi functions.

- `model.py`
  - Define `MyMidiTransformer` class, inherit from `torch.nn.Module`.
  - Define the loss functions for the model.

- `token.py`
  - Define representation tokens and their "main form" (text representation).
  - Contain some hard-coded configurations in midi preprocessing.

- `vocabs.py`
  - Define `Vocabs` class that record the vocabulary set, vocabulary building configurations and midi preprocessing parameters.
  - Contain the build-vocabulary function.


### Some terms used in function name

- A **"midi"** means a `miditoolkit.MidiFile` instance.

- A **"piece"** means a string of text representation of midi file, without tailing `\n`.

- A **"text list"** means a list of strings obtained from `piece.split(' ')` or can be turned into a "piece" after `' '.join(text_list)`.

- An **"array"** means a 2-d numpy array that encoded a piece with respect to a vocabulary set.


## Tool Scripts (`./`)

Pythons scripts

- `evaluate_model_wrapper.py`: Use python's `argparse` module to make using `evaluate_model.sh` easier.

- `extract.py`: Used for debugging. Extract piece(s) from the given corpus directory into text representation(s), midi file(s), or piano-roll graph(s) in png.

- `generate_with_models.py`: Use trained model to generate midi files, with or without a primer.

- `get_eval_features_of_midis.py`: Do as per its name. It will get midi files in a directory. Output results as a JSON file `eval_features.json` at the root of the directory.

- `make_arrays.py`: Generate `vocabs.json` and `arrays.npz` from `corpus` and `shape_vocab` if it exists.

- `midi_to_corpus.py`: Pre-process midi files into a "corpus". The parameter would be stored in `paras`. It creates `corpus`, `paras`, and `pathlist` in the corpus directory.

- `plot_bpe_log.py`: Make figures to visualize the data in the log files that contains the loggings of Multi-note BPE program.

- `print_dataset.py`: Used for debugging. Print out the results of dataset `__getitem__` and other related things.

- `train.py`: Train a model from a corpus.

- `verify_corpus_equality.py`: To make sure two corpus are representing the same list of midi files.


Shell scripts

- `evaluate_model_wrapper_wrapper.sh`
  - Read parameters from config files and used them as the arguments for `evaluate_model_wrapper.py`. This is for the convenience of the testing of evaluate/generation parameters.

- `evaluate_model.sh`
   1. Arguments are passed as environment variables.
   2. Get evaluation features of the dataset's `TEST_PATHS_FILE` files using `get_eval_features_of_midis.py`.
   3. Get evaluation features of the unconditional, instrument-informed, and prime continution generation result of the model using the combination of `generate_with_models.py` and `get_eval_features_of_midis.py`.

- `experiment_script/`: Pre-programmed experiment execution script
  - `apply_learned_shapes_to_other.sh`
  - `data_preproc_and_bpes.sh`
  - `full_model_and_ablation.sh`

- `pipeline.sh`:
  1. Pre-process midi files into a corpus with `midi_to_corpus.py`.
  2. If `DO_BPE` is "true", then run `bpe/learn_vocab` to create a new merged corpus. After it is done, run `verify_corpus_equality.py` to make sure there are no errors and run `plot_bpe_log.py` to visualize the loggings.
  3. Make arrays file and vocabs file of the corpus with `make_arrays.py`.
  4. Train a model on the corpus with `train.py`.
  5. Get evaluation features of training dataset the model generated midi files with `evaluate_model.sh`.
