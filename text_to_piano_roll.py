import sys

from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt

from util import CorpusIterator, get_corpus_paras, b36str2int

TRANSPARENT_GREY = (0.8, 0.8, 0.8, 0.8)

def print_help():
    print('python3 text_to_piano_roll.py corpus_dir_path out_path begin end')
    print('The output file name will be [out_path]_[i].png, where i is the index number of piece in corpus.')
    print('If [being] and [end] are specified, will only out the pieces in range [begin, end)')
    print('If [end] is unset, it will be begin + 1.')
    print('If [end] is -1, it will be set to the length of corpus.')

def text_to_piano_roll_read_args():
    if len(sys.argv) == 3:
        corpus_dir_path = sys.argv[1]
        out_path = sys.argv[2]
        begin = 0
        end = -1
    elif len(sys.argv) == 4:
        corpus_dir_path = sys.argv[1]
        out_path = sys.argv[2]
        begin = int(sys.argv[3])
        end = begin + 1
    elif len(sys.argv) == 5:
        corpus_dir_path = sys.argv[1]
        out_path = sys.argv[2]
        begin = int(sys.argv[3])
        end = int(sys.argv[4])
    else:
        print_help()
        exit()
    return corpus_dir_path, out_path, begin, end

def plot_roll(piece, nth, output_file_path):
    plt.clf()
    plt.xlabel('Time')
    plt.ylabel('Pitch')
    # plt.ylim(0, 127)
    text_list = piece.split(' ')

    cur_measure_length = 0
    cur_measure_onset = 0
    # know how long this piece is and draw measure lines
    for text in text_list[1:-1]:
        typename = text[0]
        if typename == 'M':
            numer, denom = (b36str2int(x) for x in text[1:].split('/'))
            cur_measure_onset += cur_measure_length
            cur_measure_length = round(nth * numer / denom)

    figure_width = min((cur_measure_onset+cur_measure_length)*0.5, 64)
    plt.figure(figsize=(figure_width, 12.8))
    plt.subplots_adjust(left=0.025, right=0.975, top=0.975, bottom=0.05)
    current_axis = plt.gca()

    cur_time = 0
    cur_measure_length = 0
    cur_measure_onset = 0
    # cur_time_signature = None
    for text in text_list[1:-1]:
        typename = text[0]
        if typename == 'R':
            pass

        elif typename == 'M':
            numer, denom = (b36str2int(x) for x in text[1:].split('/'))
            cur_measure_onset += cur_measure_length
            cur_time = cur_measure_onset
            cur_measure_length = round(nth * numer / denom)
            # if cur_time_signature != (numer, denom):
            #     cur_time_signature = (numer, denom)
            # draw measure line between itself and the next measure
            plt.axvline(x=cur_time+cur_measure_length, ymin=0, ymax=128, color=TRANSPARENT_GREY, linewidth=0.5, zorder=0)

        elif typename == 'P':
            position = b36str2int(text[1:])
            cur_time = position + cur_measure_onset

        elif typename == 'T':
            if ':' in text[1:]:
                tempo, position = (b36str2int(x) for x in text[1:].split(':'))
                cur_time = position + cur_measure_onset
            else:
                tempo = b36str2int(text[1:])
            plt.text(x=cur_time, y=128, s=f'Tempo={tempo}')

        elif typename == 'N':
            is_cont = (text[1] == '~')
            if is_cont:
                note_attr = tuple(b36str2int(x) for x in text[3:].split(':'))
            else:
                note_attr = tuple(b36str2int(x) for x in text[2:].split(':'))
            if len(note_attr) == 5:
                cur_time = note_attr[4] + cur_measure_onset
                note_attr = note_attr[:4]
            pitch, duration, velocity, track_number = note_attr
            current_axis.add_patch(
                Rectangle(
                    (cur_time, pitch), duration-0.1, 0.8,
                    edgecolor='grey', linewidth=0.75, facecolor='lightblue', fill=True
                )
            )

        elif typename == 'S':
            shape_string, *other_attr = text[1:].split(':')
            relnote_list = []
            for s in shape_string[:-1].split(';'):
                is_cont = (s[-1] == '~')
                if is_cont:
                    relnote = [b36str2int(a) for a in s[:-1].split(',')]
                else:
                    relnote = [b36str2int(a) for a in s.split(',')]
                relnote = [is_cont] + relnote
                relnote_list.append(relnote)
            base_pitch, time_unit, velocity, track_number, *position = (b36str2int(x) for x in other_attr)
            if len(position) == 1:
                cur_time = position[0] + cur_measure_onset
            prev_pitch = None
            prev_onset_time = None
            for is_cont, rel_onset, rel_pitch, rel_dur in relnote_list:
                pitch = base_pitch + rel_pitch
                duration = rel_dur * time_unit
                onset_time = cur_time + rel_onset * time_unit
                current_axis.add_patch(
                    Rectangle(
                        (onset_time, pitch), duration-0.1, 0.8,
                        edgecolor='grey', linewidth=0.75, facecolor='lightblue', fill=True
                    )
                )
                if prev_pitch is not None:
                    plt.plot(
                        [prev_onset_time+0.4, onset_time+0.4], [prev_pitch+0.4, pitch+0.4],
                        color=(0.9, 0.12, 0.2, 0.9), marker='o', markersize=2, linewidth=1.0
                    )
                prev_pitch = pitch
                prev_onset_time = onset_time
        else:
            raise ValueError(f'bad token string: {text}')
    plt.autoscale()
    plt.savefig(output_file_path)

def text_to_piano_roll(corpus_dir_path, out_path, begin, end):
    with CorpusIterator(corpus_dir_path) as corpus_iterator:
        corpus_paras = get_corpus_paras(corpus_dir_path)
        print(corpus_paras)
        if len(corpus_paras) == 0:
            print('Error: no piece in input file')
            exit(1)
        if end == -1:
            end = len(corpus_iterator)
        for i, piece in enumerate(corpus_iterator):
            if begin <= i < end:
                if len(piece) == 0:
                    continue
                # with open(f'{out_path}_{i}', 'w+', encoding='utf8') as tmp_file:
                #     tmp_file.write(piece)
                #     tmp_file.write('\n')
                plot_roll(piece, corpus_paras['nth'], f'{out_path}_{i}.png')
                print(f'dumped {out_path}_{i}.png')
            elif i >= end:
                break

if __name__ == '__main__':
    corpus_dir_path, out_path, begin, end = text_to_piano_roll_read_args()
    print('start text_to_piano_roll.py:', corpus_dir_path, out_path, begin, end)
    text_to_piano_roll(corpus_dir_path, out_path, begin, end)
