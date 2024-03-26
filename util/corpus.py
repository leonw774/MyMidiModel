import os

from tqdm import tqdm
import yaml

from .tokens import BEGIN_TOKEN_STR

def to_paras_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'paras')

def to_corpus_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'corpus')

def to_pathlist_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'pathlist')

def to_shape_vocab_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'shape_vocab')

def to_vocabs_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'vocabs.json')

def to_arrays_file_path(corpus_dir_path: str) -> str:
    return os.path.join(corpus_dir_path, 'arrays.npz')

def dump_corpus_paras(paras_dict: dict) -> str:
    return yaml.dump(paras_dict)

def get_corpus_paras(corpus_dir_path: str) -> dict:
    paras_path = to_paras_file_path(corpus_dir_path)
    with open(paras_path, 'r', encoding='utf8') as paras_file:
        corpus_paras_dict = yaml.safe_load(paras_file)
    return corpus_paras_dict


class CorpusReader:
    """
    Lazy read corpus file (which is often quite big)

    In the first iteration call, the offsets of each line are cached.
    After that, each line is obtained by file.seek() and file.read().
    """
    def __init__(self, corpus_dir_path: str, use_tqdm=True) -> None:
        self.corpus_file_path = to_corpus_file_path(corpus_dir_path)
        self.pathlist_file_path = to_pathlist_file_path(corpus_dir_path)
        self.file = open(self.corpus_file_path, 'r', encoding='utf8')
        self.line_cache = []
        self.length = None
        self.use_tqdm = use_tqdm

    def prepare(self) -> None:
        for _ in self:
            pass

    # context manager
    def __enter__(self):
        return self

    def __exit__(self, _value, _type, _trackback):
        self.file.close()

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        with open(self.pathlist_file_path, 'r', encoding='utf8') as f:
            self.length = sum(1 for _ in f)
        return self.length

    def __iter__(self):
        offset = 0
        self.file.seek(0)
        tqdm_file_iter = tqdm(
            self.file,
            desc=f'CorpusReader iter: ({self.corpus_file_path})',
            ncols=0,
            disable=(not self.use_tqdm)
        )
        for line_index, line in enumerate(tqdm_file_iter):
            # not just newline
            if len(line) > 1:
                # is piece
                if line.startswith(BEGIN_TOKEN_STR):
                    # is new piece
                    if line_index > len(self.line_cache):
                        self.line_cache.append((offset, len(line)))
                    yield line[:-1] # remove \n at the end
            offset += len(line)


    def __getitem__(self, index: int) -> str:
        if index >= len(self.line_cache):
            if len(self.line_cache) > 0:
                # move file pointer to the last known line end
                last_start, last_len = self.line_cache[-1]
                start_index =  len(self.line_cache)
            else:
                last_start, last_len = 0, 0
                start_index = 0
            offset = last_start + last_len
            self.file.seek(offset)
            tqdm_file_iter = tqdm(
                self.file,
                desc=f'CorpusReader: [{index}] ({self.corpus_file_path})',
                ncols=0,
                disable=(not self.use_tqdm)
            )
            for line_index, line in enumerate(tqdm_file_iter):
                # not just newline
                if len(line) > 1:
                    # is piece
                    if line.startswith(BEGIN_TOKEN_STR):
                        # is always new piece
                        self.line_cache.append((offset, len(line)))
                        if line_index + start_index == index:
                            return line[:-1] # remove \n at the end
                offset += len(line)

        # cannot return --> out of bound
        if index >= len(self.line_cache):
            raise IndexError(
                f'line index out of range: {index} > {len(self.line_cache)}'
            )

        self.file.seek(self.line_cache[index][0])
        # minus one to remove \n at the end
        result = self.file.read(self.line_cache[index][1] - 1)
        return result
