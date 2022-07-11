import torch
from torch import nn, Tensor
from fast_transformers import TransformerDecoderBuilder


class LinearTransformerDecoder(nn.Module):
    
    def __init__(self,
            n_layers: int,
            n_attn_heads: int,
            embed_dim: int,
            dropout_rate: float,
            vocabs: dict,
            max_sample_length: int,
            ) -> None:
        super().__init__()

        self._pad_id = vocabs['events']['text2id']['PAD']

        self._embed_dim = embed_dim

        # event, duration, velocity, track_number, instrument, tempo, position, measure_number
        # [...] the embedding vector at padding_idx will default to all zeros [...]
        self._evt_emb = nn.Embedding(
            num_embeddings=vocabs['events']['size'],
            embedding_dim=embed_dim,
            padding_idx=self._pad_id
        )
        self._dur_emb = nn.Embedding(
            num_embeddings=vocabs['durations']['size'],
            embedding_dim=embed_dim,
            padding_idx=self._pad_id
        )
        self._vel_emb = nn.Embedding(
            num_embeddings=vocabs['velocities']['size'],
            embedding_dim=embed_dim,
            padding_idx=self._pad_id
            
        )
        self._trn_emb = nn.Embedding(
            num_embeddings=vocabs['track_numbers']['size'],
            embedding_dim=embed_dim,
            padding_idx=self._pad_id
        )
        self._ins_emb = nn.Embedding(
            num_embeddings=vocabs['instruments']['size'],
            embedding_dim=embed_dim,
            padding_idx=self._pad_id
        )
        self._tmp_emb = nn.Embedding(
            num_embeddings=vocabs['tempos']['size'],
            embedding_dim=embed_dim,
            padding_idx=self._pad_id
        )

        # these two don't have padding
        # NOTE: should we use signoid encoding for "positional" and measure embedding?
        self._pos_emb = nn.Embedding(
            num_embeddings=vocabs['positions']['size'],
            embedding_dim=embed_dim
        )
        self._mea_emb = nn.Embedding(
            num_embeddings=vocabs['max_measure_number'],
            embedding_dim=embed_dim
        )

        self._linear_transformer = TransformerDecoderBuilder.from_kwargs(
            n_layers=n_layers,
            n_heads=n_attn_heads,
            query_dimensions=embed_dim//n_attn_heads,
            value_dimensions=embed_dim//n_attn_heads,
            feed_forward_dimensions=4*embed_dim,
            activation='gelu',
            dropout=dropout_rate,
            attention_type="causal-linear", # linear attention with low-triangular masking
        ).get()
    
    def forward():
        pass
