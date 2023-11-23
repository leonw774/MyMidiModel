from functools import reduce
from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import x_transformers

try:
    from fast_transformers.builders import TransformerEncoderBuilder, RecurrentEncoderBuilder
    from fast_transformers.masking import FullMask
    from fast_transformers.feature_maps import ActivationFunctionFeatureMap
    from fast_transformers.utils import make_mirror

    def my_elu_function(x):
        return F.elu(x) + 1

    # becasuse the elu feature map implemented in fast_transformers use lambda function
    # have to rewrite it to be pickle-able
    MY_ELU_FEATURE_MAP = ActivationFunctionFeatureMap.factory(my_elu_function)

except ImportError:
    pass

from . import tokens
from .vocabs import Vocabs
from .corpus import ATTR_NAME_INDEX, ALL_ATTR_NAMES, OUTPUT_ATTR_NAMES


class MyMidiTransformer(nn.Module):
    def __init__(self,
            vocabs: Vocabs,
            use_linear_attn: bool,
            max_seq_length: int,
            permute_mps: bool,
            permute_track_number: bool,
            layers_number: int,
            attn_heads_number: int,
            embedding_dim: int,
            not_use_mps_number: bool,
            dropout_rate: float = 0.1
            ) -> None:
        super().__init__()

        assert vocabs.events.text2id[tokens.PADDING_TOKEN_STR] == 0

        self.vocabs: Vocabs = vocabs
        self.max_seq_length = max_seq_length
        self.permute_mps = permute_mps
        self.permute_track_number = permute_track_number

        self.layers_number = layers_number
        self.attn_heads_number = attn_heads_number
        self.embedding_dim = embedding_dim
        self.not_use_mps_number = not_use_mps_number

        # set the to True ONLY when generating sample
        self.inferencing = False

        ######## Inputs

        self.input_attr_names = list(ALL_ATTR_NAMES) # make copy
        if not_use_mps_number:
            self.input_attr_names.remove('mps_numbers')
        self.input_attrs_indices = [ATTR_NAME_INDEX[fname] for fname in self.input_attr_names]

        self.embedding_vocabs_size = [
            min(self.max_seq_length, self.vocabs.max_mps_number) + 1 # plus one for padding
            if ALL_ATTR_NAMES[idx] == 'mps_numbers' else
            getattr(self.vocabs, ALL_ATTR_NAMES[idx]).size
            for idx in self.input_attrs_indices
        ]

        self.embedding_layers = nn.ModuleList([
            nn.Embedding(
                num_embeddings=vsize,
                embedding_dim=embedding_dim,
                # padding_idx=self.vocabs.events.text2id[tokens.PADDING_TOKEN_STR]
                padding_idx=0 # should be zero
                # [...] the embedding vector of padding_idx will default to all zeros [...]
            )
            for vsize in self.embedding_vocabs_size
        ])
        if not_use_mps_number:
            self.positional_embedding_layer = nn.Embedding(
                num_embeddings=self.max_seq_length,
                embedding_dim=embedding_dim
            )
        else:
            self.positional_embedding_layer = None
        self.embedding_dropout = nn.Dropout(dropout_rate)

        ######## Outputs

        self.output_attr_names = list(OUTPUT_ATTR_NAMES) # make copy
        self.output_attrs_indices = [
            ATTR_NAME_INDEX[fname]
            for fname in self.output_attr_names
        ]

        self.logit_vocabs_size = [
            getattr(self.vocabs, ALL_ATTR_NAMES[i]).size
            for i in self.output_attrs_indices
        ]
        self.to_logit_linears = nn.ModuleList([
            nn.Linear(
                in_features=embedding_dim,
                out_features=vsize,
                bias=False # no bias need because they are after a layer normalization
            )
            for vsize in self.logit_vocabs_size
        ])

        self.use_linear_attn = use_linear_attn

        ######## Attention layer

        # False is masked, True is keep
        if use_linear_attn:
            # this causal (lower trianglur) mask is passed but not actually used,
            # because the "causal-ness" is already implemented in fast_transformers
            self.causal_mask = torch.tril(
                torch.ones(max_seq_length, max_seq_length),
                diagonal=0
            ).bool()
            self.causal_mask = FullMask(self.causal_mask)

        if use_linear_attn:
            # https://linear-transformers.com/
            params = {
                'activation': 'gelu', # same as SymphonyNet
                'n_layers': layers_number,
                'n_heads': attn_heads_number,
                'query_dimensions': embedding_dim // attn_heads_number,
                'value_dimensions': embedding_dim // attn_heads_number,
                'feed_forward_dimensions': 4 * embedding_dim, # same as SymphonyNet
                'attention_type': 'causal-linear',
                'dropout': dropout_rate,
                'feature_map': MY_ELU_FEATURE_MAP  # make the model pickle-able
            }
            self.transformer_decoder = TransformerEncoderBuilder.from_dictionary(params).get()
            self.transformer_decoder_inference = RecurrentEncoderBuilder.from_dictionary(params).get()
            make_mirror(self.transformer_decoder, self.transformer_decoder_inference)
        else:
            self.transformer_decoder = x_transformers.Decoder(
                depth=layers_number,
                dim=embedding_dim,
                heads=attn_heads_number,
                attn_dim_head=(embedding_dim // attn_heads_number),
                attn_dropout=dropout_rate,
                ff_dropout=dropout_rate
            )
            # layer = nn.TransformerEncoderLayer(
            #     d_model=embedding_dim,
            #     nhead=attn_heads_number,
            #     dim_feedforward=4*embedding_dim,
            #     dropout=dropout_rate,
            #     batch_first=True
            # )
            # self.transformer_decoder = nn.TransformerEncoder(
            #     encoder_layer=layer,
            #     num_layers=layers_number
            # )

        # set padding vectors to zeros because the _init_weights set it to something else
        with torch.no_grad():
            for emb_layer in self.embedding_layers:
                emb_layer.weight.data[0].zero_()


    def to_input_attrs(self, batch_input_seqs: Tensor) -> Tensor:
        # expect batch_input_seqs has shape: (batch_size, seq_size, all_attr_num)
        return batch_input_seqs[..., self.input_attrs_indices]

    def to_output_attrs(self, batch_input_seqs: Tensor) -> Tensor:
        # expect batch_input_seqs has shape: (batch_size, seq_size, all_attr_num)
        return batch_input_seqs[..., self.output_attrs_indices]

    def forward(self, x, memory=None):
        # x has shape: (batch_size, seq_size, in_attr_number)
        if self.use_linear_attn and self.inferencing:
            # The recurrent linear transformer only need to receive the last element
            # with shape of (batch_size, embed_size)
            x = x[:, -1:] # become (batch_size, 1, embed_size)
        embs = [emb(x[..., i]) for i, emb in enumerate(self.embedding_layers)]
        # emb_sum = sum(embs)
        emb_sum = reduce(torch.add, embs) # cool trick

        position_memory = None
        if self.not_use_mps_number:
            if memory is None:
                bs, ss, _ = x.size
                potision_number = torch.arange(ss).repeat((bs, 1)).to(x.device)
                position_memory = 0
            else:
                potision_number = torch.tensor([memory[1]]).repeat((bs, 1)).to(x.device)
                position_memory = memory[1] + 1
            # potision_number has shape (batch_size, seq_size)
            pos_emb = self.positional_embedding_layer(potision_number)
            emb_sum = emb_sum + pos_emb

        emb_sum_dropout = self.embedding_dropout(emb_sum)

        if self.use_linear_attn:
            if self.inferencing:
                # no mask is needed when using recurrent for inference
                emb_sum_dropout = emb_sum_dropout[:, 0] # become (batch_size, embed_size)
                linear_tf_memory = None if memory is None else memory[0]
                transformer_output, linear_tf_memory = self.transformer_decoder_inference(
                    emb_sum_dropout,
                    linear_tf_memory
                )
            else:
                # in fast_transformer's FullMask class, 0 is masked, 1 is keep
                causal_mask = self.causal_mask
                length_mask = x[..., ATTR_NAME_INDEX['evt']].ne(0).bool().to(x.device)
                length_mask = FullMask(mask=length_mask)
                transformer_output = self.transformer_decoder(
                    emb_sum_dropout,
                    causal_mask,
                    length_mask
                )
        else:
            # Casual mask is not needed because x_transformer.Decoder has causal=True on default
            # False is masked, True is keep
            length_mask = x[..., ATTR_NAME_INDEX['evt']].ne(0).to(x.device)
            # print(length_mask)
            transformer_output = self.transformer_decoder(emb_sum_dropout, mask=length_mask)

        logits_tuple = tuple(
            to_logit(transformer_output) for to_logit in self.to_logit_linears
        )

        if self.use_linear_attn and self.inferencing:
            return logits_tuple, (linear_tf_memory, position_memory)
        else:
            return logits_tuple

    # Override the eval() and train() method to integrate the self.inferencing flag
    # Also added self.inference()
    def eval(self) -> None:
        super().eval()
        self.inferencing = False

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        self.inferencing = False

    def inference(self) -> None:
        """
        Equivalent to
        ```
        model.eval()
        model.inferencing = True
            ```
        """
        super().eval()
        self.inferencing = True

# end class MyMidiTransformer

LOSS_PADDING_ARG_CHOICES = ['ignore', 'wildcard', 'normal']
"""
- ignore:   PADDING = IGNORE, no exception
- wildcard: its loss is ignored, but its presence is not ignored
- normal:   the model have to correctly predict it to be padding
"""
LOSS_PADDING_ARG_CHOICES_DEFAULT = 'ignore'

def compute_losses(
        pred_logits: List[Tensor],
        target_labels: Tensor,
        padding: str = LOSS_PADDING_ARG_CHOICES_DEFAULT) -> Tuple[Tensor, List[Tensor]]:
    """
    Parameters:
    - `pred_logits` is a list:
        - Length: out_attr_number
        - Elements are tenors.
          - Shape is (batch_size, seq_size, attr_vocab_size)
    - `target_labels` has shape (batch_size, seq_size, out_attr_number)
    - `padding` decide how to handle padding symbol and how to reduce
      the loss vector. Can be one in `LOSS_PADDING_ARG_CHOICE`.

    Return final loss and a list of losses of each head.
    """
    if padding not in LOSS_PADDING_ARG_CHOICES:
        raise ValueError(
            f'`padding` argument in compute_losses should be in {LOSS_PADDING_ARG_CHOICES}.'
        )

    # target_labels have to be long int
    target_labels = target_labels.long()
    head_losses = [
        F.cross_entropy(
            input=logits.transpose(1, 2),
            # transpose(1, 2) to become (batch_size, attr_vocab_size, seq_size)
            # because input shape should be (batch, category, dimensions... )
            target=target_labels[..., k], # (batch_size, seq_size)
            ignore_index=(0 if padding != 'normal' else -100), # padding is index 0
            reduction=(
                'none' # return tensor size (batch_size, seq_size)
                if padding != 'ignore' else
                'mean' # return scalar tensor
            )
        )
        for k, logits in enumerate(pred_logits)
    ]

    if padding != 'ignore':
        length_mask = target_labels[..., ATTR_NAME_INDEX['evt']].ne(0)
        head_losses = [
            (head_loss * length_mask).sum() / length_mask.sum()
            for head_loss in head_losses
        ]
    loss = torch.stack(head_losses).mean()

    return loss, head_losses
