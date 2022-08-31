import torch
from torch import nn, Tensor

from util import SPECIAL_TOKENS

class MidiTransformerDecoder(nn.Module):
    def __init__(self,
            layers_number: int,
            attn_heads_number: int,
            d_model: int,
            vocabs: dict,
            ) -> None:
        super().__init__()

        self._pad_id = vocabs['events']['text2id'][SPECIAL_TOKENS[0]]
        # self._pad_id = 0
        assert self._pad_id == 0

        self._d_model = d_model

        # Input
        # event, pitch, duration, velocity, track_number, instrument, tempo, position, measure_number
        embedding_vocab_size = [
            vocabs['events']['size'],
            vocabs['pitchs']['size'],
            vocabs['durations']['size'],
            vocabs['velocities']['size'],
            vocabs['track_numbers']['size'],
            vocabs['instruments']['size'],
            vocabs['tempos']['size'],
            vocabs['positions']['size'],
            vocabs['max_seq_length']
        ]
        self._embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=vsize,
                embedding_dim=d_model,
                padding_idx=self._pad_id # [...] the embedding vector at padding_idx will default to all zeros [...]
            )
            for vsize in embedding_vocab_size
        ])
        # Output
        # event, pitch, duration, velocity, track_number, (instrument), (position)
        logit_vocab_size = [
            vocabs['events']['size'],
            vocabs['pitchs']['size'],
            vocabs['durations']['size'],
            vocabs['velocities']['size'],
            vocabs['track_numbers']['size'],
            vocabs['instruments']['size'], # NOTE: should we predict instrument?
            vocabs['positions']['size'],
        ]
        if vocabs['position_method'] == 'event':
            logit_vocab_size.pop()
        self._logit = nn.ModuleList([
            nn.Linear(
                in_features=d_model,
                out_features=vsize
            )
            for vsize in logit_vocab_size[:-1]
        ])

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=attn_heads_number,
            batch_first=True
        )
        self._transfromer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=layers_number
        )

    def forward(self, x, mask):
        # x.shape = (batch, sequence, feature)
        x = sum(
            emb(x[:,:,i]) for i, emb in self._embeddings
        )
        x = self._transfromer_decoder(
            tgt=x,
            memory=None,
            tgt_mask=mask
        )
        out = (
            logit(x) for logit in self._logit
        )
        return out

    def get_mask(size: int) -> Tensor:
        return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)