import json
import os
import sys

from matplotlib import pyplot as plt
from pandas import Series


def main(corpus_dir_path, log_file_path):
    with open(log_file_path, 'r', encoding='utf8') as logfile:
        log_texts = logfile.read()
    # raise ValueError if not found
    total_used_time_start_pos = log_texts.index('Total used time: ')
    total_used_time_end_pos = total_used_time_start_pos
    total_used_time_end_pos += (log_texts[total_used_time_start_pos:]).index('\n') # first newline
    total_used_time_text = log_texts[total_used_time_start_pos:total_used_time_end_pos]
    total_used_time = float(total_used_time_text.split(': ')[1])

    end_early = 'End iterations early' in log_texts
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

    # start from 1 because index 0 is iter number
    iteration_log_column_name =  log_lines[2].split(', ')[1:]
    iteration_log_lines = log_lines[3:-2] if end_early else log_lines[3:-1]
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
    stats_file_path = os.path.join(corpus_stats_dir_path, 'bpe_learn_stats.json')
    with open(stats_file_path, 'w+', encoding='utf8') as stats_file:
        json.dump(bpe_stats_dict, stats_file)

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

            plt.subplots_adjust(left=0.2)
            plt.text(
                x=0.01,
                y=0.2,
                s=left_texts,
                transform=plt.gcf().transFigure
            )

            plt.plot(iter_nums, col_list, label=col_name)
            output_path = os.path.join(
                corpus_stats_dir_path,
                f'bpe_{col_name.replace(" ", "_")}.png'
            )
            plt.savefig(output_path)
            plt.clf()

            # draw shape size distribution
            if col_name == 'Shape size':
                plt.figure(figsize=(16.8, 6.4))
                plt.title(col_name+' histogram')
                plt.hist(col_list, bins=list(range(max(col_list)+1)))
                output_path = os.path.join(
                    corpus_stats_dir_path,
                    f'bpe_{col_name.replace(" ", "_")}_hist.png'
                )
                plt.savefig(output_path)
                plt.clf()

    print('Write bpe_stats json and png done.')

if __name__ == '__main__':
    corpus_dir_path = sys.argv[1]
    log_file_path = sys.argv[2]
    main(corpus_dir_path, log_file_path)
