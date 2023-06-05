from argparse import ArgumentParser
import os
import subprocess
from traceback import format_exc

def parse_args():
    parser = ArgumentParser()

    # optional
    parser.add_argument(
        '--model-dir-path',
        type=str,
        default=''
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1
    )
    parser.add_argument(
        '--midi-to-piece-paras',
        type=str,
        default='',
    )
    parser.add_argument(
        '--log-path',
        type=str,
        default='/dev/null'
    )
    parser.add_argument(
        '--softmax-temperature',
        type=float,
        nargs='+',
        default=[1.0]
    )
    parser.add_argument(
        '--nucleus-sampling-threshold',
        type=float,
        nargs='+',
        default=[1.0]
    )
    parser.add_argument(
        '--seed',
        type=int,
        nargs='?',
        default='',
        const=''
    )

    # required
    parser.add_argument(
        'midi_dir_path',
        type=str,
    )
    parser.add_argument(
        'test_pathlist',
        type=int
    )
    parser.add_argument(
        'primer_measure_length',
        type=int
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args_dict = vars(args)
    # push
    for k, v in args_dict.items():
        if isinstance(v, list):
            os.environ[k] = ' '.join(map(str, v))
        else:
            os.environ[k] = str(v)
    subprocess.run(
        ['./evaluate_model.sh'],
        check=True
    )
    return 0

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
