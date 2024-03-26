from argparse import ArgumentParser

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.figure import Figure
# from miditoolkit import MidiFile

from util.tokens import (
    b36strtoi,
    MEASURE_EVENTS_CHAR,
    POSITION_EVENTS_CHAR,
    TEMPO_EVENTS_CHAR,
    MULTI_NOTE_EVENTS_CHAR
)
from util.midi import piece_to_midi
from util.corpus import CorpusReader, get_corpus_paras
from util.argparse_helper import MyHelpFormatter


PIANOROLL_MEASURELINE_COLOR = (0.8, 0.8, 0.8, 1.0)
PIANOROLL_TRACK_COLORMAP = cm.get_cmap('rainbow')

def piece_to_roll(piece: str, tpq: int) -> Figure:
    text_list = piece.split(' ')
    midi = piece_to_midi(piece, tpq)

    track_count = len(midi.instruments)
    track_colors = [
        PIANOROLL_TRACK_COLORMAP(i / track_count, alpha=0.6)
        for i in range(track_count)
    ]
    track_number_to_index = {
        int(track.name.split('_')[1]): track_index
        for track_index, track in enumerate(midi.instruments)
    }

    last_measure_end = 0
    last_event_time = max(
        # note off events
        [note.end for track in midi.instruments for note in track.notes]
        # tempo change events
        + [tempo_change.time for tempo_change in midi.tempo_changes]
    )
    # print('last_event', last_event_time)
    last_ts_change = midi.time_signature_changes[-1]
    last_ts_length = 4 * last_ts_change.numerator // last_ts_change.denominator
    last_measure_end = (
        (last_event_time - last_ts_change.time) // last_ts_length + 1
    )
    figure_width = min(last_measure_end * 0.25, 128)
    plt.clf()
    plt.figure(figsize=(figure_width, 12.8), dpi=200)
    current_axis = plt.gca()
    # plt.subplots_adjust(left=0.025, right=0.975, top=0.975, bottom=0.05)


    for track_index, track in enumerate(midi.instruments):
        for note in track.notes:
            if track.is_drum:
                # xy of ellipse is center
                drum_note_width = note.end - note.start - 0.25
                drum_note_xy = (
                    note.start + drum_note_width / 2, note.pitch + 0.4
                )
                current_axis.add_patch(
                    Ellipse(
                        xy=drum_note_xy,
                        width=drum_note_width,
                        height=0.8,
                        edgecolor='grey',
                        linewidth=0.25,
                        facecolor=track_colors[track_index],
                        fill=True
                    )
                )
            else:
                current_axis.add_patch(
                    Rectangle(
                        xy=(note.start, note.pitch),
                        width=(note.end - note.start - 0.25),
                        height=0.8,
                        edgecolor='grey',
                        linewidth=0.25,
                        facecolor=track_colors[track_index],
                        fill=True
                    )
                )

    # draw other infos
    # - draw measure lines
    # - put bpm texts
    # - draw lines for multi-note
    cur_time = 0
    cur_measure_length = 0
    cur_measure_onset = 0
    for text in text_list[1:-1]:
        typename = text[0]
        if typename == MEASURE_EVENTS_CHAR:
            numer, denom = (b36strtoi(x) for x in text[1:].split('/'))
            cur_measure_onset += cur_measure_length
            cur_time = cur_measure_onset
            cur_measure_length = 4 * tpq * numer // denom
            # print('draw measure line', cur_measure_onset, cur_measure_length)
            plt.axvline(
                x=cur_time,
                ymin=0,
                ymax=128,
                color=PIANOROLL_MEASURELINE_COLOR,
                linewidth=0.5,
                zorder=0
            )

        elif typename == TEMPO_EVENTS_CHAR:
            tempo = b36strtoi(text[1:])
            # print('bpm', tempo, 'at', cur_time)
            plt.annotate(
                xy=(cur_time+0.05, 0.97),
                text=f'bpm\n{tempo}',
                xycoords=('data', 'axes fraction')
            )

        elif typename == POSITION_EVENTS_CHAR:
            cur_time = cur_measure_onset + b36strtoi(text[1:])

        elif typename == MULTI_NOTE_EVENTS_CHAR:
            shape_string, *other_attrs = text[1:].split(':')
            relnote_list = []
            for s in shape_string[:-1].split(';'):
                if s[-1] == '~':
                    relnote = [True] + [b36strtoi(a) for a in s[:-1].split(',')]
                else:
                    relnote = [False] + [b36strtoi(a) for a in s.split(',')]
                relnote_list.append(relnote)
            base_pitch, stretch_factor, _velocity, track_number = (
                map(b36strtoi, other_attrs)
            )
            # print(base_pitch, stretch_factor, _velocity, track_number)
            track_index = track_number_to_index[track_number]

            darker_track_colors = (
                *map(lambda x: 0.75 * x, track_colors[track_index][:3]), 0.97
            )

            # for every note and its neighbors: draw line
            # if relation is overlap -> from onset to onset
            # if relation is immed following -> from offset to onset
            for i in range(len(relnote_list)-1):
                ri = relnote_list[i]
                ri_start = cur_time + ri[1] * stretch_factor
                ri_end = cur_time + (ri[1] + ri[3]) * stretch_factor
                ri_pitch = ri[2] + base_pitch
                for j in range(i+1, len(relnote_list)):
                    rj = relnote_list[j]
                    rj_start = cur_time + rj[1] * stretch_factor
                    rj_pitch = rj[2] + base_pitch
                    immed_follow_onset = None
                    if rj_start >= ri_end:
                        if immed_follow_onset is None:
                            if rj_start - ri_end > 4 * tpq:
                                break
                            immed_follow_onset = rj_start
                        elif rj_start != immed_follow_onset:
                            break
                        # draw line for immed following
                        plt.plot(
                            [ri_end - 0.5, rj_start + 0.5],
                            [ri_pitch + 0.4, rj_pitch + 0.4],
                            color=darker_track_colors,
                            linewidth=0.5,
                            markerfacecolor=track_colors[track_index],
                            marker='>',
                            markersize=1.0
                        )
                    else:
                        # draw line for overlapping
                        plt.plot(
                            [ri_start + 0.5, rj_start + 0.5],
                            [ri_pitch + 0.4, rj_pitch + 0.4],
                            color=darker_track_colors,
                            linewidth=0.5,
                            markerfacecolor=track_colors[track_index],
                            marker='o',
                            markersize=1
                        )
                # end for relnote_list
            # end if
        # end for text_list
    plt.xlabel('Time')
    plt.ylabel('Pitch')
    plt.xlim(xmin=0)
    plt.autoscale()
    plt.margins(x=0)
    plt.tight_layout()
    return plt.gcf()


def read_args():
    parser = ArgumentParser(formatter_class=MyHelpFormatter)
    parser.add_argument(
        'corpus_dir_path',
        metavar='CORPUS_DIR_PATH',
        type=str
    )
    parser.add_argument(
        'output_path',
        metavar='OUTPUT_PATH',
        type=str,
        help='The output file path will be "{OUTPUT_PATH}_{i}.{EXT}", \
            where i is the index number of piece in corpus.'
    )
    parser.add_argument(
        '--indexing', '-i',
        type=str,
        required=True,
        help="""Required at least one indexing string.
An indexing string is in the form of "INDEX" or "BEGIN:END".\\n
Former: extract piece at INDEX.\\n
Latter: extract pieces from BEGIN (inclusive) to END (exclusive).\\n
- If any number A < 0, it will be replaced with (CORPUS_LENGTH - A).\\n
- If BEGIN is empty, 0 will be used.\\n
- If END is empty, CORPUS_LENGTH will be used.\\n
- BEGIN and END can not be empty at the same time.\\n
- Multiple indexing strings are seperated by commas.\\n
Example: --indexing ":2, 3:5, 7, -7, -5:-3, -2:"\\n
"""
    )
    parser.add_argument(
        '--extract-midi', '--midi',
        action='store_true',
        help='Output midi file(s) stored in extracted pieces \
            with path name "{OUTPUT_PATH}_{i}.mid"'
    )
    parser.add_argument(
        '--extract-txt', '--txt',
        action='store_true',
        help='Output text file(s) containing text representation of \
            extracted pieces with path "{OUTPUT_PATH}_{i}.txt"'
    )
    parser.add_argument(
        '--extract-img', '--img',
        action='store_true',
        help='Output PNG file(s) of the pianoroll representation of \
            extracted pieces with path "{OUTPUT_PATH}_{i}.png"'
    )
    args = parser.parse_args()
    return args

def parse_index_string(index_str_list: list, corpus_length: int) -> set:
    indices_to_extract = set()
    for index_str in index_str_list:
        if ':' in index_str:
            s = index_str.split(':')
            assert len(s) == 2
            b, e = s
            assert not (b == '' and e == '')
            if b == '':
                b = 0
            if e == '':
                e = corpus_length
            b = int(b)
            e = int(e)
            if b < 0:
                b = corpus_length + b
            if e < 0:
                e = corpus_length + e
            # print(b, e)
            indices_to_extract.update(list(range(b, e))) # implies b <= e
        else:
            i = int(index_str)
            if i < 0:
                i = corpus_length + i
            # print(i)
            indices_to_extract.add(i)
    return set(indices_to_extract)


def main():
    print('==== start extract.py ====')
    args = read_args()

    if not (args.extract_img or args.extract_txt or args.extract_midi):
        print('Please choose at least one format to output')
        return 0

    print('\n'.join([
        f'{k}: {v}'
        for k, v in vars(args).items()
    ]))

    with CorpusReader(args.corpus_dir_path) as corpus_reader:
        corpus_paras = get_corpus_paras(args.corpus_dir_path)
        print('Corpus parameters:')
        print(corpus_paras)

        indices_to_extract = parse_index_string(
            index_str_list=args.indexing.split(','),
            corpus_length=len(corpus_reader)
        )
        print("extracting indices:", list(indices_to_extract))

        # extract
        for i in indices_to_extract:
            piece = corpus_reader[i]
            if len(piece) == 0:
                continue

            if args.extract_midi:
                midi = piece_to_midi(
                    piece,
                    corpus_paras['tpq'],
                    ignore_pending_note_error=False
                )
                midi.dump(f'{args.output_path}_{i}.mid')

            if args.extract_txt:
                output_file_path = f'{args.output_path}_{i}.txt'
                with open(output_file_path, 'w+', encoding='utf8') as f:
                    f.write(piece+'\n')

            if args.extract_img:
                figure = piece_to_roll(piece, corpus_paras['tpq'])
                figure.savefig(f'{args.output_path}_{i}.png')

            print(f'extracted {args.output_path}_{i}')

if __name__ == '__main__':
    EXIT_CODE = main()
    exit(EXIT_CODE)
