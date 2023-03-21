import os

from tqdm import tqdm

from .tokens import BEGIN_TOKEN_STR

# to handle large file
class CorpusReader:
    def __init__(self, corpus_dir_path: str) -> None:
        self.file_path = os.path.join(corpus_dir_path, 'corpus')
        self.line_pos = []
        self.length = -1
        self.file = open(self.file_path, 'r', encoding='utf8')

    # context manager
    def __enter__(self):
        return self

    def __exit__(self, _value, _type, _trackback):
        self.file.close()

    def __len__(self) -> int:
        if self.length == -1:
            for _ in self: # iter itself to get cache
                pass
        return self.length

    def __iter__(self):
        if len(self.line_pos) == 0:
            self.file.seek(0)
            tmp_line_pos = [0]
            tmp_length = 0
            offset = 0
            for line in tqdm(self.file, desc=f'CorpusReader reading {self.file_path} '):
                offset += len(line)
                if len(self.line_pos) == 0:
                    tmp_line_pos.append(offset)
                if len(line) > 1: # not just newline
                    tmp_length += 1 if line.startswith(BEGIN_TOKEN_STR) else 0
                    yield line[:-1] # remove \n at the end
            self.line_pos = tmp_line_pos
            self.length = tmp_length
        else:
            for i in range(self.length):
                yield self[i]

    def __getitem__(self, index: int) -> str:
        if index >= self.length:
            raise IndexError(f'line index out of range: {index} > {self.length}')
        if len(self.line_pos) == 0:
            result = ''
            for i, line in enumerate(self): # iter itself to get cache
                if i == index:
                    result = line
            return result
        else:
            self.file.seek(self.line_pos[index])
            # minus one to remove \n at the end
            result = self.file.read(self.line_pos[index+1]-self.line_pos[index]-1)
            return result
