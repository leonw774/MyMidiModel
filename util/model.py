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
        self.project_linears = nn.ModuleList([
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
            # because the "causal-ness" is already implemented in the calculation of linear attention
            self.causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length), diagonal=0).bool()
            self.causal_mask = FullMask(self.causal_mask)

        # name's encoder, used as decoder
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
        # self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-6)

        self.apply(self._init_weights)
        # set padding vectors to zeros because the _init_weights set it to something else
        with torch.no_grad():
            for emb_layer in self.embedding_layers:
                emb_layer.weight.data[0].zero_()

    # copied from SymphonyNet
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.embedding_dim ** -0.5)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def to_input_attrs(self, batch_input_seqs: Tensor) -> Tensor:
        # expect batch_input_seqs has shape: (batch_size, seq_size, all_attr_num)
        return batch_input_seqs[..., self.input_attrs_indices]

    def to_output_attrs(self, batch_input_seqs: Tensor) -> Tensor:
        # expect batch_input_seqs has shape: (batch_size, seq_size, all_attr_num)
        return batch_input_seqs[..., self.output_attrs_indices]

    def forward(self, x, memory=None):
        # x has shape: (batch_size, seq_size, in_attr_number)
        if self.use_linear_attn and self.inferencing:
            # The recurrent model only need to receive the last element with size of (batch_size, embed_size)
            x = x[:, -1:] # become (batch_size, 1, embed_size)
        embs = [emb(x[..., i]) for i, emb in enumerate(self.embedding_layers)]
        # for i, emb in enumerate(embs):
        #     mask = x[..., i].ne(0).float()[..., None].to(x.device)
        #     assert (emb == emb*mask).all(), f'{i}\n{x}\n{emb}\n{mask}\n{emb*mask}'
        # emb_sum = sum(embs)
        emb_sum = reduce(torch.add, embs) # cool trick

        position_memory = None
        if self.not_use_mps_number:
            if memory is None:
                potision_number = torch.arange(x.size(1)).repeat((x.size(0), 1)).to(x.device)
                position_memory = 0
            else:
                position_memory = memory[1]
                potision_number = torch.tensor([position_memory]).repeat((x.size(0), 1)).to(x.device)
                position_memory += 1
            # potision_number has shape (batch_size, seq_size)
            pos_emb = self.positional_embedding_layer(potision_number)
            emb_sum = emb_sum + pos_emb

        emb_sum_dropout = self.embedding_dropout(emb_sum)
        if self.use_linear_attn:
            if self.inferencing:
                # no mask is needed when using recurrent for inference
                emb_sum_dropout = emb_sum_dropout[:, 0] # become (batch_size, embed_size)
                linear_tf_memory = None if memory is None else memory[0]
                transformer_output, linear_tf_memory = self.transformer_decoder_inference(emb_sum_dropout, linear_tf_memory)
            else:
                # in fast_transformer's FullMask class, 0 is masked, 1 is keep
                causal_mask = self.causal_mask
                length_mask = FullMask(mask=x[..., 0].ne(0).bool().to(x.device))
                transformer_output = self.transformer_decoder(emb_sum_dropout, causal_mask, length_mask)
        else:
            # Casual mask is not needed because x_transformer.Decoder is has causal=True on default
            # causal_mask = self.causal_mask[:x.size(1), :x.size(1)].repeat(x.size(0), 1, 1).to(x.device)
            # False is masked, True is keep
            length_mask = x[..., 0].ne(0).to(x.device)
            # print(length_mask)
            transformer_output = self.transformer_decoder(emb_sum_dropout, mask=length_mask)
        # transformer_output = self.layer_norm(transformer_output)
        logits_tuple = tuple(
            proj(transformer_output) for proj in self.project_linears
        )
        # assert all(not torch.isnan(lg).any() for lg in logits), [torch.isnan(lg).nonzero(as_tuple=True) for lg in logits]
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
        # changing the order of these two expression will cause self.inferencing = False, weird
        super().eval()
        self.inferencing = True

# end class MyMidiTransformer

LOSS_PADDING_ARG_CHOICES = ['ignore', 'wildcard', 'normal']
LOSS_PADDING_ARG_CHOICES_DEFAULT = 'ignore'

def compute_losses(
        pred_logits: List[Tensor],
        target_labels: Tensor,
        padding: str = LOSS_PADDING_ARG_CHOICES_DEFAULT) -> Tuple[Tensor, List[Tensor]]:
    """
        Parameters:
        - `pred_logits` is a list:
          - length: out_attr_number
          - elements are tenors with shape: (batch_size, seq_size, attr_vocab_size)

        - `target_labels` has shape: (batch_size, seq_size, out_attr_number)

        - `padding` decide how to reduce the loss vector. Can be one in `LOSS_PADDING_ARG_CHOICE`.

        Return final loss and a list of losses of each head.
    """
    # target_labels have to be long int
    target_labels = target_labels.long()
    head_losses_list = [
        F.cross_entropy(
            input=logits.transpose(1, 2),
            # transpose(1, 2) to become (batch_size, attr_vocab_size, seq_size)
            # basically treat seq_size as one of the K dimensions and K = 1
            target=target_labels[..., k], # (batch_size, seq_size)
            ignore_index=(0 if padding != 'normal' else -100), # padding is index 0
            reduction='none' # return tensor size (batch_size, seq_size)
        )
        for k, logits in enumerate(pred_logits)
    ]
    # (batch_size, seq_size, out_attr_number) <- [(batch_size, seq_size) ... (batch_size, seq_size)]

    if padding == 'ignore':
        # average each attribute losses by number of its non-padding labels
        # as if we have k sequences with ignore paddings == 'all'
        head_losses = [
            head_losses.sum() / torch.count_nonzero(head_losses)
            for head_losses in head_losses_list
        ]
        loss = torch.stack(head_losses).mean()

    elif padding == 'wildcard':
        # only ignore the padding labels at the end
        # the padding labels in the non-padding-event tokens are seen as wildcards that ANY predictions is correct
        num_token = torch.count_nonzero(target_labels[..., ATTR_NAME_INDEX['evt']])
        head_losses = [
            head_losses.sum() / num_token
            for head_losses in head_losses_list
        ]
        loss = torch.stack(head_losses).mean()

    elif padding == 'normal':
        # only ignore the padding labels at the end
        # the padding labels in the non-padding-event tokens are seen as normal labels and IS the correct answer
        num_token = torch.count_nonzero(target_labels[..., ATTR_NAME_INDEX['evt']])
        head_losses = [
            head_losses.sum() / num_token
            for head_losses in head_losses_list
        ]
        loss = torch.stack(head_losses).mean()

    else:
        raise ValueError(f'`padding` argument in compute_losses can should be in {LOSS_PADDING_ARG_CHOICES}.')

    return loss, head_losses
