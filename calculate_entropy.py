from argparse import ArgumentParser, Namespace
from collections import Counter
import os

import numpy as np
from tqdm import tqdm

from util.corpus import to_arrays_file_path

def read_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        'corpus_dir_paths',
        type=str,
        nargs='+', # at least one
    )

    return parser.parse_args()


def main():
    args = read_args()
    for corpus_dir_path in args.corpus_dir_paths:
        entropy_counter = Counter()
        print(corpus_dir_path)
        npz_path = to_arrays_file_path(corpus_dir_path)
        if not os.path.isfile(npz_path):
            print(f'No arrays.npz file in {corpus_dir_path}')
            continue
        total_token_count = 0
        with np.load(npz_path) as npz_obj:
            for array in tqdm(npz_obj.values()):
                total_token_count += array.shape[0]
                for token in array:
                    t = tuple(map(int, token))
                    # because array is int16, we can put 3 of them into a basic python int type (int64)
                    if len(t) == 9:
                        t = ((t[0] | t[1] << 16 | t[2] << 32), (t[3] | t[4] << 16 | t[5] << 32), (t[6] | t[7] << 16 | t[8] << 32))
                    if len(t) > 6:
                        t = ((t[0] | t[1] << 16 | t[2] << 32), (t[3] | t[4] << 16 | t[5] << 32), *t[6:])
                    elif len(t) > 3:
                        t = ((t[0] | t[1] << 16 | t[2] << 32), *t[3:])
                    entropy_counter[t] += 1
        entropy = 0
        for v in entropy_counter.values():
            p = v / total_token_count
            entropy -= p * np.log2(p)
        print(corpus_dir_path, ':', entropy)


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
