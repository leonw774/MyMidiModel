from argparse import ArgumentParser, Namespace
from collections import Counter
import os

import numpy as np

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
    entropy_counter = Counter()
    for corpus_dir_path in args.corpus_dir_paths:
        npz_path = to_arrays_file_path(corpus_dir_path)
        if not os.path.isfile(npz_path):
            continue
        total_token_count = 0
        with np.load(npz_path) as npz_obj:
            for array in npz_obj.values():
                total_token_count += array.shape[0]
                for token in array:
                    t = tuple(token)
                    entropy_counter[t] += 1
        entropy = 0
        for v in entropy_counter.values():
            p = v / total_token_count
            entropy -= p * np.log2(p)
        print(corpus_dir_path, ':', entropy)


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
