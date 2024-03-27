# MyMidiModel

**Table of Contents**

- [To run the experiment](#to-run-the-experiment)
  - [Step 1](#step-1)
  - [Step 2 (Optional)](#step-2-optional)
  - [Step 3](#step-3)
- [Configuration Files (`configs/`)](#configuration-files-configs)
- [Dataset (`data/midis/`)](#dataset-datamidis)
- [Corpus Structure (`data/corpus/`)](#corpus-structure-datacorpus)
- [Multinote BPE (`bpe/`)](#multinote-bpe-bpe)
- [Model (`models/`)](#model-models)
- [Codes (`util/`)](#codes-util)
  - [Some terms used in function name](#some-terms-used-in-function-name)
- [Tool and Experiment Scripts (`./`)](#tool-scripts-)
  - [Python scripts](#pythons-scripts)
  - [Shell scripts](#shell-scripts)


## To run the experiment

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

You can recreate our experiment by running the scripts in `experiment_script`.

``` bash
./experiment_script/data_preproc_and_bpe.sh
./experiment_script/apply_learned_shape_to_other.sh
./experiment_script/full_model_and_ablation.sh snd --full --ablation
./experiment_script/full_model_and_ablation.sh lmd_full --full --ablation
```

(Optional 12-layer Linear Transformer model trained on LMD)

``` bash
ONLY_EVAL_UNCOND=true ./pipeline.sh lmd_full ours_sample1.0 linear_mid --use-existed
```

## Configuration Files (`configs/`)

- Files in `configs/corpus` set parameters for `midi_to_corpus.py` and `make_arrays.py`
  - Vocabulary parameters:
    - `TPQ`: ticks per quarter note / time units per quarter note
    - `MAX_TRACK_NUMBER`
    - `MAX_DURATION`
    - `VELOCITY_STEP`: quantize velocity value to `(VELOCITY_STEP * 1, VELOCITY_STEP * 2, ... VELOCITY_STEP * k)` where `k = 127 // VELOCITY_STEP`
    - `CONTINUING_NOTE`: use continuing note or not
    - `TEMPO_MIN`, `TEMPO_MAX`, `TEMPO_STEP`
  - Dataset processing setting
    - `MIDI_WORKER_NUMBER`
    - `MIDI_DIR_PATH`: the path to dataset directory
    - `DATA_NAME`
    - `TEST_PATHS_FILE`, `VALID_PATHS_FILE`: point to a file in `configs/split`

- Files in `configs/bpe` set parameters for `bpe/learn_vocabs` (implementation of Multi-note BPE algorithm).
  - `BPE_ITER_NUM`: number of iteration / size of vocabulary
  - `ADJACENCY`: name of the adjacency; can be "ours" or "mulpi"
  - `MIN_SCORE_LIMIT`: early-stop if the score of best shape (ie: its frequency) is less than it
  - `SAMPLE_RATE`: track sampling rate
  - `BPE_LOG`: output log or not
  - `BPE_WORKER_NUMBER`

- Files in `configs/model` set parameters for `train.py` and the config file name under `configs/eval` to be used by `evaluate_model.sh`.
  - `SEED`: random seed in training process
  - Dataset parameters
    - `MAX_SEQ_LENGTH`: max sequence length to feed into model
    - `VIRTUAL_PIECE_STEP_RATIO`: if > 0, split over-length pieces into multiple virtual pieces
    - `FLATTEN_VIRTUAL_PIECES`: if true, all virtual pieces has a index number. Otherwise, virtual pieces within same real piece shares the same index number
    - `PERMUTE_MPS`: whether or not the dataset should permute all the maximal permutable subarrays as data augmentation
    - `PERMUTE_TRACK_NUMBER`: Permute all the track numbers relative to the instruments as data augmentation
    - `PITCH_AUGMENTATION_RANGE`
  - Model parameter
    - `USE_LINEAR_ATTENTION`
    - `LAYERS_NUMBER`
    - `ATTN_HEADS_NUMBER`
    - `EMBEDDING_DIM`
    - `NOT_USE_MPS_NUMBER`
  - Training parameter
    - `BATCH_SIZE`
    - `MAX_UPDATES`: number of updates before training stop
    - `VALIDATION_INTERVAL`: number of update before each validation
    - `LOSS_PADDING`: How should loss function handle padding symbol; can be "ignore", "wildcard", and "normal"
    - `MAX_GRAD_NORM`
    - Learning rate schedule (linear warmup and then decay to end ratio) parameters
      - `LEARNING_RATE_PEAK`
      - `LEARNING_RATE_WARMUP_UPDATES`
      - `LEARNING_RATE_DECAY_END_UPDATES`
      - `LEARNING_RATE_DECAY_END_RATIO`
    - `EARLY_STOP`: number of non-improved validation before early
  - Others
    - `VALID_EVAL_SAMPLE_NUMBER`: number of samples the validated model generates to be evaluated
    - `USE_DEVICE`
    - `EVAL_CONFIG_NAME`: point to a file in `configs/eval`

- Files in `configs/split` contain lists of paths, relative to each dataset root, of midi files to be used as test set and validation set of the datasets. Their path are referenced by variable `TEST_PATHS_FILE` and `VALID_PATHS_FILE` in files of `configs/corpus`.

- Files in `configs/eval` store parameters for `evaluate_model.sh`.
  - `SEED`: random seed in evaluation/generation process
  - Generation setting
    - `EVAL_SAMPLE_NUMBER`
    - `EVAL_WORKER_NUMBER`
    - `PRIMER_MEASURE_LENGTH`: number of measure for primer-continuation generation task
    - `SAMPLE_FUNCTION`: can be "none", "top-p", "top-k"
    - `SAMPLE_THRESHOLD`
    - `SOFTMAX_TEMPERATURE`
  - Evaluation setting
    - `EVAL_MIDI_TO_PIECE_PARAS_FILE`: the vocabulary parameter to quantize MIDI files when evaluating them


## Dataset (`data/midis/`)

The datasets we used, [SymphonyNet_Dataset](https://symphonynet.github.io/) and [lmd_full](https://colinraffel.com/projects/lmd/), are expected to be found under `data/midis`. However, the path `midi_to_corpus.py` would be looking is the `MIDI_DIR_PATH` variables set in the the corpus configuration file. So it could be in any place you want. Just set the path right.


## Corpus Structure (`data/corpus/`)

Corpi are located at `data/corpus/`. A complete "corpus" is directory containing at least 5 files in the following list.

- `corpus`: A text file. Each `\n`-separated line is a text representation of a midi file. This is the "main form" of the representation. Created by `midi_to_corpus.py`.

- `paras`: A yaml file that contains parameters of pre-processing used by `midi_to_corpus.py`. Created by `midi_to_corpus.py`.

- `pathlist`: A text file. Each `\n`-separated line is the path, relative to project root, of midi file corresponding to the text representation in `corpus`. Created by `midi_to_corpus.py`.
  - Note that a corpus include all processable, uncorrupted midi file, including the test and validation files. The split of test and validation happens at training and evaluating stage.

- `vocabs.json`: The vocabulary to be used by the model. The format is defined in `util/vocabs.py`. Created by `make_arrays.py`.

- `arrays.npz`: A zip file of numpy arrays in `.npy` format. Can be accessed by `numpy.load()` and it will return an instance of `NpzFile` class. This is the "final form" of the representation (i.e. include pre-computed MPS order positio numbers) that would be used to train model. Created by `make_arrays.py`.

Other possible files and directories are:

- `stats/`: A directoy that contains statistics about the corpus. Some figures outputed by `make_arrays.py` and by `plot_bpe_log.py` would be end up here.

- `shape_vocab`: A text file created by `bpe/learn_vocab`. If exist, it will be read by `make_arrays.py` to help create `vocabs.json`.

- `arrays/`: A temporary directory for placing the `.npy` files before they are zipped.

- `make_array_debug.txt`: A text file that shows array content of the first piece in the corpus. Created by `make_arrays.py`.


## Multinote BPE (`bpe/`)

Stuffs about Multi-note BPE are all in `bpe/`.

Source codes:

- `apply_vocab.cpp` and `learn_vocab.cpp`

- `classes.cpp` and `classes.hpp`: Define class of corpus, multi-note, rel-note, etc. And I/O functions.

- `functions.cpp` and `functions.hpp`: Other functions and algorithms.

They should compile to two binaries with `make -C bpe all`:

- `apply_vocab`: Apply merge operations with a known shape list to a corpus file. Output a new corpus file.

- `learn_vocab`: Do Multi-node BPE to a corpus file. Output a new corpus file and a shape list in `shape_vocab`.


## Model (`models/`)

- Models are created by `train.py`.

- Learning rate schedule is hard-coded warmup and linear decay.

- `accelerate` from Huggingface when flag set for distributed training
  - in our config file, we use 4 devices

- A completed trained model is stored at `models/{DATE_AND_FULL_CONFIG_NAME}/best_model.pt` as a "pickled" python object that would be saved and loaded by `torch.save()` and `torch.load()`.

- Two directories are under `models/{DATE_AND_FULL_CONFIG_NAME}/`
  - `ckpt/` is where checkpoint model and generated sample would be placed
  - `eval_samples/` is where the evaluation samples generated by `generate_with_model.py` called in `evaluate_model.sh` would be placed.

- A file `models/{DATE_AND_FULL_CONFIG_NAME}/test_paths` containing all paths to the test files would be created when running `evaluate_model.sh`.


## Codes (`util/`)

### Some terms used in function name

- A **"midi"** means a `miditoolkit.MidiFile` instance.

- A **"piece"** means a string of text representation of midi file, without tailing `\n`.

- A **"text-list"** means a list of strings obtained from `piece.split(' ')` or can be turned into a "piece" after `' '.join(text_list)`.

- An **"array"** means a 2-d numpy array that encoded a piece with respect to a vocabulary set.

### Modules

- `corpus.py`
  - Define corpus directory structure.
  - Define corpus reader class

- `arrays.py`
  - Define the array form of the representation.
  - Contain text-list-to-array and array-to-text-list functions.

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


## Tool and Experiment Scripts (`./`)

### Python scripts

- `extract.py`: Used for debugging. Extract piece(s) from the given corpus directory into text representation(s), midi file(s), or piano-roll graph(s) in png.

- `generate_with_models.py`: Use model to generate midi files, with or without primer(s).

- `get_eval_features_of_midis.py`: Do as per its name. It will get midi files in a directory. Output results as a JSON file `eval_features.json` at the root of the directory.

- `make_arrays.py`: Generate `vocabs.json` and `arrays.npz` from `corpus` and `shape_vocab` if it exists.

- `midi_to_corpus.py`: Pre-process midi files into a "corpus". The parameter would be stored in `paras`. It creates `corpus`, `paras`, and `pathlist` in the corpus directory.

- `plot_bpe_log.py`: Make figures to visualize the data in the log files that contains the loggings of Multi-note BPE program.

- `print_dataset.py`: Used for debugging. Print out the results of dataset `__getitem__` and other related things.

- `train.py`: Train a model from a corpus.

- `verify_corpus_equality.py`: To make sure two corpus are representing the same list of midi files.

### Shell scripts

- `evaluate_model.sh`
  1. Read a file in `config/eval` for its arguments.
  2. Get evaluation features of the dataset's `TEST_PATHS_FILE` files using `get_eval_features_of_midis.py`.
  3. Get evaluation features of the unconditional, instrument-conditioned, and primer-continution generation result of the model using the combination of `generate_with_models.py` and `get_eval_features_of_midis.py`.

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
