from collections import Counter
import json
import os
import sys

from matplotlib import pyplot as plt
from pandas import Series

from util.tokens import NOTE_EVENTS_CHAR, MULTI_NOTE_EVENTS_CHAR
from util.corpus import CorpusReader


def main(corpus_dir_path, log_file_path):
    with open(log_file_path, 'r', encoding='utf8') as logfile:
        log_texts = logfile.read()
    # raise ValueError if not found
    total_used_time_start_pos = log_texts.index('Total used time: ')
    total_used_time_end_pos = (log_texts[total_used_time_start_pos:]).index('\n') + total_used_time_start_pos
    total_used_time_text = log_texts[total_used_time_start_pos:total_used_time_end_pos]
    total_used_time = float(total_used_time_text.split(': ')[1])

    bpe_log_start_pos = log_texts.index('Reading done. There are')
    bpe_log_end_pos = log_texts.index('Writing merged corpus file')
    log_texts = log_texts[bpe_log_start_pos:bpe_log_end_pos]

    log_lines = log_texts.splitlines()
    piece_number = int(log_lines[0].split(' ')[-2])
    start_stats_texts = log_lines[1].split(', ')
    start_stats = {
        sstext.split(': ')[0] : sstext.split(': ')[1]
        for sstext in start_stats_texts
    }
    end_stats_texts = log_lines[-1].split(', ')
    end_stats = {
        estext.split(': ')[0] : estext.split(': ')[1]
        for estext in end_stats_texts
    }

    iteration_log_column_name =  log_lines[2].split(', ')[1:] # start from 1 because index 0 is iter number
    iteration_log_lines = log_lines[3:-1]
    iteration_log_lists = [[] for _ in range(len(iteration_log_column_name))]
    for iter_log_line in iteration_log_lines:
        for i, row_text in enumerate(iter_log_line.split(', ')[1:]):
            row_element = None
            if row_text.isdigit():
                row_element = int(row_text)
            else:
                try:
                    row_element = float(row_text)
                except ValueError:
                    row_element = row_text
            iteration_log_lists[i].append(row_element)
    iteration_log_dict = {
        col_name: col_list
        for col_name, col_list in zip(iteration_log_column_name, iteration_log_lists)
    }
    iteration_log_dict['Shape size'] = [
        shape_str.count(';')
        for shape_str in iteration_log_dict['Shape']
    ]


    corpus_stats_dir_path = os.path.join(corpus_dir_path, 'stats')
    if not os.path.exists(corpus_stats_dir_path):
        os.makedirs(corpus_stats_dir_path)

    # output json
    bpe_stats_dict = {
        'piece_number': piece_number,
        'start_stats': start_stats,
        'end_stats': end_stats,
        'total_used_time': total_used_time,
        'iteration_log_dict': iteration_log_dict
    }
    with open(os.path.join(corpus_stats_dir_path, 'bpe_learn_stats.json'), 'w+', encoding='utf8') as statsfile:
        json.dump(bpe_stats_dict, statsfile)

    # plot iteration_log_dict
    iter_nums = list(range(len(iteration_log_lines)))
    for col_name, col_list in iteration_log_dict.items():
        if col_name != 'Shape':
            plt.figure(figsize=(16.8, 6.4))
            plt.title(col_name)
            if col_name == 'Multinote count':
                left_texts = (f'total_piece_number\n  ={piece_number}\n'
                    + '\n'.join(f'{k}\n  ={v}' for k, v in start_stats.items()) + '\n'
                    + '\n'.join(f'{k}\n  ={v}' for k, v in end_stats.items()) + '\n'
                    + f'total_used_time\n  ={total_used_time}'
                )
            else:
                left_texts = '\n'.join(
                    [f'{k}={float(v):.5f}' for k, v in dict(Series(col_list).describe()).items()]
                )
            # fit y = a * exp(b * x) --> log(y) = log(a) + b * x
            # x = np.array(iter_nums)
            # y = np.array(col_list)
            # b, loga = np.polyfit(x, np.log(y), 1, w=np.sqrt(y))
            # left_texts += '\nexponential equation fit:\n  ' + f'y = {np.exp(loga):.5f} * e ^ ({b:.5f} * x)'

            plt.subplots_adjust(left=0.2)
            plt.text(
                x=0.01,
                y=0.2,
                s=left_texts,
                transform=plt.gcf().transFigure
            )

            plt.plot(iter_nums, col_list, label=col_name)
            plt.savefig(os.path.join(corpus_stats_dir_path, f'bpe_{col_name.replace(" ", "_")}.png'))
            plt.clf()

            # draw shape size distribution
            if col_name == 'Shape size':
                plt.figure(figsize=(16.8, 6.4))
                plt.title(col_name+' histogram')
                plt.hist(col_list, bins=list(range(max(col_list)+1)))
                plt.savefig(os.path.join(corpus_stats_dir_path, f'bpe_{col_name.replace(" ", "_")}_hist.png'))
                plt.clf()

    # plot shape distribution
    shape_counter = Counter()
    with CorpusReader(corpus_dir_path) as corpus_reader:
        for piece in corpus_reader:
            for text in piece.split():
                if text[0] == NOTE_EVENTS_CHAR:
                    if len(text) == 1:
                        shape_counter['0,0,1;'] += 1
                    else:
                        shape_counter['0,0,1~;'] += 1
                elif text[0] == MULTI_NOTE_EVENTS_CHAR:
                    shape_counter[text[1:]] += 1
    
    shape_counter = dict(shape_counter)
    sorted_shape_counter = sorted([(count, shape) for shape, count in shape_counter.items()])
    sorted_shape_freq = [c for c, _ in sorted_shape_counter]
    total_shape_num = sum(sorted_shape_freq)
    sorted_shape_freq = [c / total_shape_num for c in sorted_shape_freq]
    sorted_shape_label = [s for _, s in sorted_shape_counter]

    plt.figure(figsize=(16.8, 6.4))
    plt.title('Distribution of shapes')
    plt.bar(
        x=[c for c, _ in sorted_shape_freq],
        label=[s for _, s in sorted_shape_label]
    )
    plt.savefig(os.path.join(corpus_stats_dir_path, f'bpe_corpus_shape_distribution.png'))
    plt.clf()

    print('Write bpe_stats json and png done.')

if __name__ == '__main__':
    corpus_dir_path = sys.argv[1]
    log_file_path = sys.argv[2]
    main(corpus_dir_path, log_file_path)
