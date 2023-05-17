from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
# from torch.profiler import record_function

try:
    import mps_loss
except ImportError:
    pass

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
from .corpus import ATTR_NAME_INDEX, ALL_ATTR_NAMES, OUTPUTABLE_ATTR_NAMES


class MyMidiTransformer(nn.Module):
    def __init__(self,
            vocabs: Vocabs,
            use_linear_attn: bool,
            max_seq_length: int,
            permute_mps: bool,
            layers_number: int,
            attn_heads_number: int,
            embedding_dim: int,
            input_context: bool,
            input_instruments: bool,
            output_instruments: bool,
            embedding_dropout_rate: float = 0.1
            ) -> None:
        super().__init__()

        assert vocabs.events.text2id[tokens.PADDING_TOKEN_STR] == 0

        self.vocabs: Vocabs = vocabs
        self.max_seq_length = max_seq_length
        self.permute_mps = permute_mps
        self.layers_number = layers_number
        self.attn_heads_number = attn_heads_number
        self.embedding_dim = embedding_dim
        self.input_context = input_context
        self.input_instruments = input_instruments
        self.output_instruments = output_instruments

        # set the to True ONLY when generating sample
        self.inferencing = False

        ######## Inputs

        self.input_attr_names = list(ALL_ATTR_NAMES) # make copy
        if input_context:
            if input_instruments:
                pass
            else:
                self.input_attr_names.remove('instruments')
        else:
            if input_instruments:
                self.input_attr_names.remove('time_signatures')
            else:
                self.input_attr_names.remove('instruments')
                self.input_attr_names.remove('time_signatures')

        self.input_attrs_indices = [ATTR_NAME_INDEX[fname] for fname in self.input_attr_names]

        self.embedding_vocabs_size = [
            min(self.max_seq_length, self.vocabs.max_mps_number) + 1 # plus one for padding
            if ALL_ATTR_NAMES[idx] == 'mps_numbers' else
            getattr(self.vocabs, ALL_ATTR_NAMES[idx]).size
            for idx in self.input_attrs_indices
        ]

        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=vsize,
                embedding_dim=embedding_dim,
                # padding_idx=self.vocabs.events.text2id[tokens.PADDING_TOKEN_STR]
                padding_idx=0 # should be zero
                # [...] the embedding vector of padding_idx will default to all zeros [...]
            )
            for vsize in self.embedding_vocabs_size
        ])
        self.embedding_dropout = nn.Dropout(embedding_dropout_rate)

        ######## Outputs

        self.output_attr_names = list(OUTPUTABLE_ATTR_NAMES) # make copy
        if not output_instruments:
            self.output_attr_names.remove('instruments')

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
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-6)

        self.use_linear_attn = use_linear_attn

        ######## Attention layer

        if use_linear_attn:
            # in fast_transformer's FullMask class, 0 is masked, 1 is keep
            # but this causal (lower trianglur) mask is not actually used,
            # because the "causal-ness" is already implemented in the calculation of linear attention
            self.causal_mask = FullMask(torch.tril(torch.ones(max_seq_length, max_seq_length), diagonal=0).bool())
        else:
            # in pytorch's official implementation, True is masked, False is keep
            self.causal_mask = torch.triu(torch.ones(max_seq_length, max_seq_length), diagonal=1).bool()

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
                'dropout': 0.1, # same as SymphonyNet
                'feature_map': MY_ELU_FEATURE_MAP  # make the model pickle-able
            }
            self.transformer_decoder = TransformerEncoderBuilder.from_dictionary(params).get()
            self.transformer_decoder_inference = RecurrentEncoderBuilder.from_dictionary(params).get()
            make_mirror(self.transformer_decoder, self.transformer_decoder_inference)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=attn_heads_number,
                batch_first=True
            )
            self.transformer_decoder = nn.TransformerEncoder(
                encoder_layer=layer,
                num_layers=layers_number
            )
        self.apply(self._init_weights)
        # set padding vectors to zeros because the _init_weights set it to something else
        with torch.no_grad():
            for emb_layer in self.embeddings:
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
        embs = [emb(x[..., i]) for i, emb in enumerate(self.embeddings)]
        # for i, emb in enumerate(embs):
        #     mask = x[..., i].ne(0).float()[..., None].to(x.device)
        #     assert (emb == emb*mask).all(), f'{i}\n{x}\n{emb}\n{mask}\n{emb*mask}'
        emb_sum = sum(embs)
        emb_sum_dropout = self.embedding_dropout(emb_sum)
        if self.use_linear_attn:
            if self.inferencing:
                # no mask is needed when using recurrent for inference
                emb_sum_dropout = emb_sum_dropout[:, 0] # become (batch_size, embed_size)
                transformer_output, memory = self.transformer_decoder_inference(emb_sum_dropout, memory)
            else:
                # in fast_transformer's FullMask class, 0 is masked, 1 is keep
                causal_mask = self.causal_mask
                length_mask = FullMask(mask=x[..., 0].ne(0).bool().to(x.device))
                transformer_output = self.transformer_decoder(
                    emb_sum_dropout,
                    causal_mask,
                    length_mask
                )
        else:
            # in pytorch's official implementation, True is masked, False is keep
            causal_mask = self.causal_mask[:x.shape[1], :x.shape[1]].to(x.device)
            length_mask = x[..., 0].eq(0).to(x.device)
            transformer_output = self.transformer_decoder(
                emb_sum_dropout,
                causal_mask,
                length_mask
            )
        transformer_output = self.layer_norm(transformer_output)
        logits_tuple = tuple(
            proj(transformer_output) for proj in self.project_linears
        )
        # assert all(not torch.isnan(lg).any() for lg in logits), [torch.isnan(lg).nonzero(as_tuple=True) for lg in logits]
        if self.use_linear_attn and self.inferencing:
            return logits_tuple, memory
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
        '''
            Equivalent to
            ```
            model.eval()
            model.inferencing = True
            ```
        '''
        # changing the order of these two expression will cause self.inferencing = False, weird
        super().eval()
        self.inferencing = True

# end class MyMidiTransformer


def compute_losses(
        pred_logits: List[Tensor],
        target_labels: Tensor) -> Tuple[Tensor, List[Tensor]]:
    """
        pred_logits is a list
        - length: out_attr_number
        - elements are tenors with shape: (batch_size, seq_size, attr_vocab_size)
        target_labels has shape: (batch_size, seq_size, out_attr_number)
        return a list of losses of each head
    """
    # basically treat seq_size as one of the K dimensions and K = 1
    # target_labels have to be long int
    target_labels = target_labels.long()
    head_losses = [
        F.cross_entropy(
            input=logits.transpose(1, 2),
            # transpose(1, 2) to become (batch_size, attr_vocab_size, seq_size)
            target=target_labels[..., k], # (batch_size, seq_size)
            ignore_index=0, # padding is index 0
            reduction='none' # return tensor size (batch_size, seq_size)
        )
        for k, logits in enumerate(pred_logits)
    ]
    # average losses by number of labels
    # number_of_labels = target_labels.numel()
    # head_losses = [head_l.sum() / number_of_labels for head_l in head_losses]
    # loss = sum(head_losses)

    # average losses by number of non-padding labels by each sequence
    number_of_nonpadding_each_token = torch.count_nonzero(target_labels, dim=2) # (batch_size, seq_size)
    number_of_nonpadding_each_seq = number_of_nonpadding_each_token.sum(dim=1) # (batch_size,)
    loss = torch.stack(head_losses, dim=2) # (batch_size, seq_size, out_attr_number)
    loss = loss.sum(dim=2).sum(dim=1) # (batch_size,)
    loss = torch.div(loss, number_of_nonpadding_each_seq) # element-wise division, average of sequences
    loss = loss.mean() # average of batches
    head_losses = [head_l.sum() / (head_l != 0.0).sum() for head_l in head_losses]
    return loss, head_losses


def compute_permutable_subseq_losses(
        pred_logits: List[Tensor],
        target_labels: Tensor,
        batched_mps_numbers: Tensor) -> Tuple[Tensor, List[Tensor]]:
    """
        pred_logits is a list
        - length: out_attr_num
        - elements are tenors with shape: (batch_size, seq_size, attr_vocabs_size)
        target_labels has shape: (batch_size, seq_size, out_attr_num)
        return a list of losses of each head
    """

    # find mps separators
    batched_mps_indices_as_list = [
        (
            torch.where(mps_number[:-1] < mps_number[1:])[0].tolist() # find where the number increases
            + torch.where(mps_number == 0)[0].tolist() # include all index where mps is zero (head and EOS)
        )
        for mps_number in batched_mps_numbers
    ]

    with torch.no_grad():
        # start_time = time()
        # modified_target_labels = find_min_loss_target(pred_logits, target_labels, batched_mps_indices_as_list)
        # print('Python function used time:', time()-start_time)
        # print('--------')

        # start_time = time()
        detached_pred_logits = [
            pred_attr_logit.clone().detach()
            # pred_attr_logit.clone().detach().cpu()
            for pred_attr_logit in pred_logits
        ]
        detached_target_labels = target_labels.clone().detach().long()
        # detached_target_labels = target_labels.clone().detach().cpu().long()
        cpp_modified_target_labels: Tensor = mps_loss.find_min_loss_target(
            detached_pred_logits,
            detached_target_labels,
            batched_mps_indices_as_list
        )

        # print('C++ extension function used time:', time()-start_time)
        # print('========')
        # assert (modified_target_labels == cpp_modified_target_labels).all().item()

    loss, head_losses = compute_losses(
        pred_logits,
        cpp_modified_target_labels.to(target_labels.device)
    )
    return loss, head_losses


# NOTE: this function is VERY SLOW since it has three level of for-loops and does cross_entropy with O(N^2) data worst case
# How can I make it faster?
def find_min_loss_target(
        pred_logits: List[Tensor],
        target_labels: Tensor,
        batched_mps_indices_as_list: List[List[int]]) -> Tensor:
    use_device = target_labels.device
    # detach the tensors because we don't need their gradients when finding best permutation
    detached_pred_logits = [
        pred_attr_logit.clone().detach().cpu()
        for pred_attr_logit in pred_logits
    ]
    detached_target_labels = target_labels.clone().detach().cpu()
    out_attr_number = len(pred_logits)
    for batch_number, mps_indices in enumerate(batched_mps_indices_as_list):
        # print('batch_number', batch_number)
        # print('mps_indices len', len(mps_indices))
        indices_replacments = []
        flatten_mps_target_list = []
        # will become a tensor of shape (seq_mps_flatten_size, out_attr_number)
        flatten_mps_pred_logits_list = [[] for _ in range(out_attr_number)]
        # will have out_attr_number tensors, each has shape (seq_mps_flatten_size, attr_vocabs_size)
        triu_indices_of_mps = dict()
    # with record_function('flatten_mps'):
        for i in range(len(mps_indices) - 1):
            begin_index = mps_indices[i]
            if begin_index < 0:
                continue
            end_index = mps_indices[i+1]
            mps_size = end_index - begin_index
            # assert mps_size >= 0, f'{begin_index}~{end_index}\n{mps_indices}'
            if mps_size == 2:
                flatten_mps_target_list.append(
                    detached_target_labels[batch_number, begin_index:end_index]
                )
                # view (mps_size, out_attr_number)
                for k, pred_attr_logit in enumerate(detached_pred_logits):
                    flatten_mps_pred_logits_list[k].append(
                        pred_attr_logit[batch_number, begin_index].expand((2, -1))
                    )
                    # view (attr_vocabs_size, ) and then expand into (mps_size, attr_vocabs_size)
            elif mps_size > 2:
                triu_indices = torch.triu_indices(mps_size, mps_size)
                triu_indices = triu_indices[:, :-1] # drop last
                triu_indices_of_mps[i] = triu_indices
                full_mps_target = detached_target_labels[batch_number, begin_index:end_index]
                # view (mps_size, out_attr_number)
                full_mps_target = full_mps_target.unsqueeze(0).expand((mps_size, -1, -1))
                # exapnd dim to (1, mps_size, out_attr_number) then repeat to (mps_size, mps_size, out_attr_number)
                # t1, t2, t3 ... -> t1, t2, t3 ...
                #                   t1, t2, t3 ...
                #                   :
                #                   t1, t2, t3 ...
                flatten_full_mps_target = full_mps_target[triu_indices[0], triu_indices[1]].flatten(end_dim=-2)
                # print(flatten_full_mps_target.shape)
                # t1, t2, t3 ... tn, t2, t3, t4 ... tn, ... , tn-1, tn
                flatten_mps_target_list.append(flatten_full_mps_target)

                for k, pred_attr_logit in enumerate(detached_pred_logits):
                    full_mps_attr_logit = pred_attr_logit[batch_number, begin_index:end_index]
                    # view (mps_size, attr_vocabs_size)
                    full_mps_attr_logit = full_mps_attr_logit.unsqueeze(1).expand((-1, mps_size, -1))
                    # exapnd dim to (mps_size, 1, out_attr_number) then repeat to (mps_size, mps_size, attr_vocabs_size)
                    # l1        l1, l1, l1 ...
                    # l2  -->   l2, l2, l2 ...
                    # l3        l3, l3, l3 ...
                    # :         :
                    flatten_full_mps_attr_logit = full_mps_attr_logit[triu_indices[0], triu_indices[1]].flatten(end_dim=-2)
                    # l1, l1, l1, ... , l1 (n times), l2, l2, l2, ... , l2 (n-1 times), ... , ln-1, ln-1 (2 times)
                    flatten_mps_pred_logits_list[k].append(flatten_full_mps_attr_logit)
    # with record_function('cat_and_cross'):
        # print('flatten list len:', len(flatten_mps_target_list))
        # print([t.shape[0] for t in flatten_mps_target_list])
        if len(flatten_mps_target_list) == 0:
            continue
        # F.cross_entropy only accept long
        flatten_mps_targets = torch.cat(flatten_mps_target_list, dim=0).long().to(use_device)
        # cat into (seq_mps_flatten_size, out_attr_number)
        # print('concated flatten mps target len:', flatten_mps_targets.shape[0])
        # print('concated flatten mps target:\n', flatten_mps_targets)
        flatten_mps_pred_logitss = [
            torch.cat(flatten_mps_pred_logits_list[k], dim=0).to(use_device)
            # cat into (seq_mps_flatten_size, attr_vocabs_size)
            for k in range(out_attr_number)
        ]
        flatten_mps_head_losses = [
            F.cross_entropy(
                input=flatten_mps_pred_logitss[k], # shape = (seq_mps_flatten_size, attr_vocabs_size)
                target=flatten_mps_targets[:, k], # shape = (seq_mps_flatten_size, )
                ignore_index=0, # padding is index 0
                reduction='none' # return shape (seq_mps_flatten_size, )
            )
            for k in range(out_attr_number)
        ]
        flatten_mps_head_losses = torch.stack(flatten_mps_head_losses) # (out_attr_number, seq_mps_flatten_size)
        # print(flatten_mps_head_losses)
        # because when an attribute is labeled as padding, its loss would be zero
        # we want to ignore the zero values
        flatten_mps_loss_sum = torch.sum(flatten_mps_head_losses, dim=0) # (seq_mps_flatten_size, )
        nonzero_elem_count = torch.sum((flatten_mps_head_losses != 0), dim=0) # (seq_mps_flatten_size, )
        # if divided by zero error happen at this point, the dataset must have problem
        flatten_mps_loss_mean = flatten_mps_loss_sum / nonzero_elem_count
    # with record_function('find_argmins'):
        cur_flatten_mps_index = 0
        for i in range(len(mps_indices) - 1):
            begin_index = mps_indices[i]
            if begin_index < 0:
                continue
            end_index = mps_indices[i+1]
            mps_size = end_index - begin_index
            if mps_size == 2:
                if flatten_mps_loss_mean[cur_flatten_mps_index] > flatten_mps_loss_mean[cur_flatten_mps_index+1]:
                    indices_replacments.append((begin_index, begin_index + 1))
                cur_flatten_mps_index += 2
            elif mps_size > 2:
                flatten_length = mps_size * (mps_size + 1) // 2 - 1
                cur_mps_flatten_loss_mean = flatten_mps_loss_mean[cur_flatten_mps_index:cur_flatten_mps_index+flatten_length]
                # print(cur_mps_flatten_loss_mean)
                cur_flatten_mps_index += flatten_length
                triu_indices = triu_indices_of_mps[i]
                loss_stacked = torch.full((mps_size-1, mps_size), float('inf'), device=use_device)
                loss_stacked[triu_indices[0], triu_indices[1]] = cur_mps_flatten_loss_mean
                # print(loss_stacked)
                argmin_loss_stacked = torch.argmin(loss_stacked, dim=1)
                for i, min_loss in enumerate(argmin_loss_stacked):
                    min_loss_int = min_loss.item()
                    if i != min_loss_int:
                        indices_replacments.append((begin_index + i, begin_index + min_loss_int))

        # print('indices_replacments len:', len(indices_replacments))
        # print(indices_replacments)
    # with record_function('mod_target'):
        # modify target such that the target at cur_index is now the target at min_loss_sum_index
        for cur_index, min_loss_index in indices_replacments:
            detached_target_labels[batch_number, cur_index] = detached_target_labels[batch_number, min_loss_index]

    return detached_target_labels
