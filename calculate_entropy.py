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
                    # use string instead of int tuple to save space
                    t = ':'.join(map(str, [t for t in token]))
                    entropy_counter[t] += 1
        entropy = 0
        for v in entropy_counter.values():
            p = v / total_token_count
            entropy -= p * np.log2(p)
        print(corpus_dir_path, ':', entropy)


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
