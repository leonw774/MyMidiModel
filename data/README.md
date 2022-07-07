# data

This directory stores midi files and python scripts for processing midi files.

## example_midi

Example midi files for checking/debuging the result of processing.

## midi

Midi files for dataset.

## processed_midi

Text file(s) of processed midis.

## vocab

Vocabolary JSON files made from a processed midi text.

## midi_util.py

Where processing logics are implemented.

## tokens.py

Define all token related classes and functions.

## midi_2_text.py

Tool to process one or multiple midi files into a text file. 

```
usage: midi_to_text.py [-h] [--nth NTH] [--max-track-number MAX_TRACK_NUMBER] [--max-duration MAX_DURATION] [--velocity-step VELOCITY_STEP] [--tempo-quantization TEMPO_MIN TEMPO_MAX TEMPO_STEP]
                       [--tempo-method {measure_attribute,position_attribute,measure_event,position_event}] [--not-merge-drums] [--not-merge-sparse] [--debug] [--verbose] [-r] [-w MP_WORK_NUMBER] [-o OUTPUT_PATH]
                       input_path [input_path ...]

positional arguments:
  input_path            The path(s) of input files/dictionaries

optional arguments:
  -h, --help            show this help message and exit
  --nth NTH             The time unit length would be the length of a n-th note. Must be multiples of 4. Default is 96.
  --max-track-number MAX_TRACK_NUMBER
                        The maximum tracks nubmer to keep in text, if the input midi has more "instruments" than this value, some tracks would be merged or discard. Default is 24.
  --max-duration MAX_DURATION
                        Max length of duration in unit of quarter note (beat). Default is 4.
  --velocity-step VELOCITY_STEP
                        Snap the value of velocities to multiple of this number. Default is 16.
  --tempo-quantization TEMPO_MIN TEMPO_MAX TEMPO_STEP
                        Three integers: (min, max, step), where min and max are INCLUSIVE. Default is 8, 264, 4
  --tempo-method {measure_attribute,position_attribute,measure_event,position_event}
                        Could be one of four options: 'measure_attribute', 'position_attribute', 'measure_event' and 'position_event'. 'attribute' means tempo info is part of the measure or position event token 'event' means tempo will
                        be its own token, and it will be in the sequence where there tempo change occurs. The token will be placed after the measure token and before position token
  --not-merge-drums
  --not-merge-sparse
  --debug
  --verbose
  -r, --recursive       If set, the input path will have to be directory path(s). And all ".mid" files under them will be processed recursively.
  -w MP_WORK_NUMBER, --workers MP_WORK_NUMBER
                        The number of worker for multiprocessing. Default is 1. If this number is 1 or below, multiprocessing would not be used.
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        The path of the output file of the inputs files/directories. All output texts will be written into this file, seperated by breaklines. Default is "output.txt".
```

## text_2_midi.py

A simpler tool that turns a text file back to midi files. The usage is `python3 text_2_midi.py text_path nth`. The restored midi files are named in their order in the texts: "restored_1", "restored_2", ... , "restored_N". 