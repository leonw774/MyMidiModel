from typing import List
# from time import time

import torch
from torch import nn, Tensor
import torch.nn.functional as F
# from torch.profiler import record_function

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import FullMask
from fast_transformers.feature_maps import ActivationFunctionFeatureMap

from .vocabs import Vocabs
from .corpus import TOKEN_ATTR_INDEX, COMPLETE_ATTR_NAME, OUTPUT_ATTR_NAME, array_to_text_list, text_list_to_array
from .midi import piece_to_midi
from .tokens import PADDING_TOKEN_STR, BEGIN_TOKEN_STR, END_TOKEN_STR

# becasuse the elu feature map implemented in fast_transformers use lambda function
# have to rewrite it to pickle-able
def my_elu_function(x):
    return F.elu(x) + 1

MY_ELU_FEATURE_MAP = ActivationFunctionFeatureMap.factory(my_elu_function)

class MyMidiTransformer(nn.Module):
    def __init__(self,
            vocabs: Vocabs,
            use_linear_attn: bool,
            max_seq_length: int,
            layers_number: int,
            attn_heads_number: int,
            embedding_dim: int,
            input_no_tempo: bool,
            input_no_time_signature: bool,
            embedding_dropout_rate=0.2
            ) -> None:
        super().__init__()

        assert vocabs.events.text2id[PADDING_TOKEN_STR] == 0

        self.vocabs = vocabs
        self.max_seq_length = max_seq_length
        self.layers_number = layers_number
        self.attn_heads_number = attn_heads_number
        self.embedding_dim = embedding_dim
        self.input_no_tempo = input_no_tempo
        self.input_no_time_signature = input_no_time_signature

        # Input attributess

        # the indices of input attrs in complete attribute list
        self.input_attrs_indices = [
            TOKEN_ATTR_INDEX[fname]
            for fname in COMPLETE_ATTR_NAME
            if not (input_no_tempo and fname=='tempos') or not (input_no_time_signature and fname=='time_signatures')
        ]
        self.embedding_vocabs_size = [
            ( getattr(vocabs, fname).size if fname != 'measure_numbers' else max_seq_length )
            for fname in COMPLETE_ATTR_NAME
            if not (input_no_tempo and fname=='tempos') or not (input_no_time_signature and fname=='time_signatures')
        ]
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=vsize,
                embedding_dim=embedding_dim,
                padding_idx=vocabs.events.text2id[PADDING_TOKEN_STR]
                # [...] the embedding vector at padding_idx will default to all zeros [...]
            )
            for vsize in self.embedding_vocabs_size
        ])
        self.embedding_dropout = nn.Dropout(embedding_dropout_rate)

        # Output attributes
        self.output_attrs_indices = [
            TOKEN_ATTR_INDEX[fname]
            for fname in OUTPUT_ATTR_NAME
        ]
        # because position is the last attribute in OUTPUT_ATTR_NAME
        if vocabs.paras['position_method'] == 'event':
            self.output_attrs_indices.pop()
        self.logit_vocabs_size = [
            getattr(vocabs, OUTPUT_ATTR_NAME[i]).size
            for i in self.output_attrs_indices
        ]
        self.logits = nn.ModuleList([
            nn.Linear(
                in_features=embedding_dim,
                out_features=vsize
            )
            for vsize in self.logit_vocabs_size
        ])

        self.use_linear_attn = use_linear_attn

        if use_linear_attn:
            # in fast_transformer's FullMask class, 0 is masked, 1 is keep
            # but this causal (lower trianglur) mask is not actually used,
            # because the "causal-ness" is already implemented in the calculation of linear attention
            self.causal_mask = FullMask(torch.tril(torch.ones(max_seq_length, max_seq_length), diagonal=0).bool())
        else:
            # in pytorch's official implementation, True is masked, False is keep
            self.causal_mask = torch.triu(torch.ones(max_seq_length, max_seq_length), diagonal=1).bool()

        if use_linear_attn:
            enc_builder = TransformerEncoderBuilder.from_kwargs(
                activation='relu',
                n_layers=layers_number,
                n_heads=attn_heads_number,
                feed_forward_dimensions=2048,   # same as torch's default
                query_dimensions=embedding_dim // attn_heads_number,
                value_dimensions=embedding_dim // attn_heads_number,
                dropout=0.1,                    # same as torch's default
                attention_dropout=0.1,          # same as torch's default
                attention_type="causal-linear",
                final_normalization=True,       # same as torch's default
                feature_map=MY_ELU_FEATURE_MAP  # make the model pickle-able
            )
            self.transformer_encoder = enc_builder.get()
        else:
            layer = nn.TransformerEncoderLayer( # name's encoder, used as decoder.
                d_model=embedding_dim,
                nhead=attn_heads_number,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=layer,
                num_layers=layers_number
            )

    # batched_seq_complete_attrs has shape: (batch_size, seq_size, complete_attr_num)
    def to_input_attrs(self, batched_seq_complete_attrs):
        return batched_seq_complete_attrs[..., self.input_attrs_indices]

    def to_output_attrs(self, batched_seq_complete_attrs):
        return batched_seq_complete_attrs[..., self.output_attrs_indices]

    def forward(self, x):
        # x has shape: (batch_size, seq_size, in_attr_number)
        emb_sum = sum(
            emb(x[..., i]) for i, emb in enumerate(self.embeddings)
        )
        # emb_sum = None
        # for i, emb in enumerate(self.embeddings):
        #     emb_i = emb(x[..., i])
        #     # to deal with sparisity of non-padding tokens in some attribute
        #     emb_mask = emb_i.ne(0).float().unsqueeze(dim=-1).to(x.device)
        #     if emb_sum is not None:
        #         emb_sum += emb_mask * emb_i
        #     else:
        #         emb_sum = emb_mask * emb_i
        emb_sum_dropout = self.embedding_dropout(emb_sum)
        if self.use_linear_attn:
            # in fast_transformer's FullMask class, 0 is masked, 1 is keep
            causal_mask = self.causal_mask
            length_mask = FullMask(mask=x[..., 0].ne(0).bool().to(x.device))
        else:
            # in pytorch's official implementation, True is masked, False is keep
            causal_mask = self.causal_mask[:x.shape[1], :x.shape[1]].to(x.device)
            length_mask = x[..., 0].eq(0).to(x.device)

        transformer_output = self.transformer_encoder(
            emb_sum_dropout,
            causal_mask,
            length_mask
        )
        logits = tuple(
            logit(transformer_output) for logit in self.logits
        )
        # assert all(not torch.isnan(lg).any() for lg in logits), [torch.isnan(lg).nonzero(as_tuple=True) for lg in logits]
        return logits

def generate_sample(model: MyMidiTransformer, steps: int, start_seq = None, temperature=1.0, try_count_limit=1000, print_exception=False) -> list:
    """
        Expect start_seq to be Tensor with shape: (1, seq_size, complete_attr_number) or None
        - if start_seq is None, will use `text_list_to_array([BEGIN_TOKEN_STR])` as start_seq
        return the text list of the generated piece
    """
    if start_seq is not None:
        exception_msg = f'start_seq\'s shape have to be (1, seq_length, complete_attr_number), get {start_seq.shape}'
        if len(start_seq.shape) != 3:
            raise AssertionError(exception_msg)
        elif start_seq.shape[0] != 1 or start_seq.shape[2] != len(COMPLETE_ATTR_NAME):
            raise AssertionError(exception_msg)
    else:
        start_seq = torch.from_numpy(text_list_to_array([BEGIN_TOKEN_STR], model.vocabs)).unsqueeze(0).int()

    # get model device
    model_device = next(model.parameters()).device
    training_state = model.training
    model.eval()

    text_list = array_to_text_list(start_seq[0].cpu().numpy(), vocabs=model.vocabs, is_output=False)

    input_seq = start_seq
    output_seq = model.to_output_attrs(start_seq)
    end_with_end_token = False
    # print(seq.shape)
    # for _ in tqdm(range(steps)):
    for _ in range(steps):
        clipped_seq = input_seq[:, -model.max_seq_length:]
        logits = model(clipped_seq.to(model_device))
        last_logits = [
            l[0, -1, :].to('cpu') # back to cpu, if was cuda (likely)
            # l has shape (1, sequence_length, attr_vocab_size)
            for l in logits
        ]
        # print('\n'.join([repr(logit) for logit in last_logits]))
        try_count = 0
        while try_count < try_count_limit:
            sampled_attrs = [
                torch.multinomial(F.softmax(l / temperature, dim=0), 1)
                for l in last_logits # l has shape (1, attr_vocab_size)
            ]
            # print(sampled_attrs)
            try_token = torch.stack(sampled_attrs, dim=1) # shape = (1, attr_num)
            # print([ model.vocabs.to_dict()[OUTPUT_ATTR_NAME[i]]['id2text'][int(f)] for i, f in enumerate(try_token[0]) ])
            try_token = torch.unsqueeze(try_token, dim=0) # shape = (1, 1, attr_num)
            try_seq = torch.cat((output_seq, try_token), dim=1)
            try_text_list = array_to_text_list(try_seq[0].cpu().numpy(), vocabs=model.vocabs, is_output=True)
            # print(try_text_list)
            if try_text_list[-1] == END_TOKEN_STR:
                text_list = try_text_list
                # if sampled EOS, then dont check. just end
                break

            try:
                # format checking
                # have to append EOS at the end to not raise error
                piece_to_midi(' '.join(try_text_list + [END_TOKEN_STR]), model.vocabs.paras['nth'])
                # format checking and position, tempo, measure_number calculation
                input_seq = torch.from_numpy(
                    text_list_to_array(try_text_list, vocabs=model.vocabs)
                ).unsqueeze(0).int()
                output_seq = model.to_output_attrs(input_seq)
                text_list = try_text_list
                break
            except (AssertionError, ValueError) as e:
                if print_exception:
                    print(repr(e))
                try_count += 1
                continue # keep sampling until no error

        # end while
        # print(text_list)
        if try_count == try_count_limit:
            break

        if text_list[-1] == END_TOKEN_STR:
            end_with_end_token = True
            break
    # end for each step

    if not end_with_end_token:
        text_list.append(END_TOKEN_STR)

    model.train(training_state)
    return text_list


def calc_losses(pred_logit: List[Tensor], target_logit: Tensor) -> List[Tensor]:
    """
        pred_logit is a list
        - length: out_attr_number
        - elements are tenors with shape: (batch_size, seq_size, attr_vocab_size)
        target_logit has shape: (batch_size, seq_size, out_attr_number)
        return a list of losses of each head
    """
    # basically treat seq_size as one of the K dimensions and K = 1
    # target_logit have to be long int
    target_logit = target_logit.long()
    head_losses = [
        F.cross_entropy(
            input=pred_attr_logit.transpose(1, 2), # (batch_size, attr_vocab_size, seq_size)
            target=target_logit[..., k] # (batch_size, seq_size)
        )
        for k, pred_attr_logit in enumerate(pred_logit)
    ]
    return head_losses
    # loss = sum(head_losses)
    # return loss


def calc_permutable_subseq_losses(pred_logit: List[Tensor], target_logit: Tensor, batched_mps_indices: List):
    """
        pred_logit is a list
        - length: out_attr_num
        - elements are tenors with shape: (batch_size, seq_size, attr_vocabs_size)
        target_logit has shape: (batch_size, seq_size, out_attr_num)
        mps_indices is a numpy object array of numpy int16 arrays in varying lengths
        return a list of losses of each head
    """
    use_device = target_logit.device
    cpu_pred_logit = [
        pred_attr_logit.detach().cpu()
        for pred_attr_logit in pred_logit
    ]
    cpu_target_logit = target_logit.detach().cpu()
    out_attr_number = len(pred_logit)
    for batch_number, mps_indices in enumerate(batched_mps_indices):
        mps_indices_with_begin_and_end = [0] + [m for m in mps_indices] + [cpu_target_logit.shape[1]]
        indices_replacments = []
        seq_flatten_target_logits_list = []
        # will become a tensor of shape (seq_mps_flatten_size, out_attr_number)
        seq_flatten_pred_logits_list = [[] for _ in range(out_attr_number)]
        # will have out_attr_number tensors, each has shape (seq_mps_flatten_size, attr_vocabs_size)
        triu_indices_of_mps = dict()
    # with record_function('flatten_mps'):
        for i in range(len(mps_indices_with_begin_and_end) - 1):
            begin_index = mps_indices_with_begin_and_end[i]
            end_index = mps_indices_with_begin_and_end[i+1]
            mps_size = end_index - begin_index
            # assert mps_size >= 0, f'{begin_index}~{end_index}\n{mps_indices_with_begin_and_end}'
            if mps_size == 2:
                seq_flatten_target_logits_list.append(
                    cpu_target_logit[batch_number, begin_index:end_index]
                )
                # view (mps_size, out_attr_number)
                for k, pred_attr_logit in enumerate(cpu_pred_logit):
                    seq_flatten_pred_logits_list[k].append(
                        pred_attr_logit[batch_number, begin_index].expand((2, -1))
                    )
                    # view (attr_vocabs_size, ) and then expand into (mps_size, attr_vocabs_size)
            elif mps_size > 2:
                triu_indices = torch.triu_indices(mps_size, mps_size)
                triu_indices = triu_indices[:, :-1] # drop last
                triu_indices_of_mps[i] = triu_indices
                full_mps_target = cpu_target_logit[batch_number, begin_index:end_index]
                # view (mps_size, out_attr_number)
                full_mps_target = full_mps_target.unsqueeze(0).expand((mps_size, -1, -1))
                # exapnd dim to (1, mps_size, out_attr_number) then repeat to (mps_size, mps_size, out_attr_number)
                # t1, t2, t3 ... -> t1, t2, t3 ...
                #                   t1, t2, t3 ...
                #                   :
                #                   t1, t2, t3 ...
                flatten_full_mps_target = full_mps_target[triu_indices[0], triu_indices[1]].flatten(end_dim=-2)
                # print(flatten_full_mps_target.shape)
                # t1, t2, t3 ... tn, t2, t3, t4 ... tn, tn-1, tn
                seq_flatten_target_logits_list.append(flatten_full_mps_target)

                for k, pred_attr_logit in enumerate(cpu_pred_logit):
                    full_mps_attr_logit = pred_attr_logit[batch_number, begin_index:end_index]
                    # view (mps_size, attr_vocabs_size)
                    full_mps_attr_logit = full_mps_attr_logit.unsqueeze(1).expand((-1, mps_size, -1))
                    # exapnd dim to (mps_size, 1, out_attr_number) then repeat to (mps_size, mps_size, attr_vocabs_size)
                    # l1        l1, l1, l1 ...
                    # l2  -->   l2, l2, l2 ...
                    # l3        l3, l3, l3 ...
                    # :         :
                    flatten_full_mps_attr_logit = full_mps_attr_logit[triu_indices[0], triu_indices[1]].flatten(end_dim=-2)
                    # l1, l1, l1, ... l2, l2, l2, ... ln-1, ln, ln
                    seq_flatten_pred_logits_list[k].append(flatten_full_mps_attr_logit)
    # with record_function('cat_and_cross'):
        if len(seq_flatten_target_logits_list) == 0:
            continue
        # F.cross_entropy only accept long
        seq_flatten_target_logits = torch.cat(seq_flatten_target_logits_list, dim=0).long().to(use_device)
        # cat into (attr_vocabs_size, out_attr_number)
        seq_flatten_pred_logits = [
            torch.cat(seq_flatten_pred_logits_list[k], dim=0).to(use_device)
            # cat into (attr_vocabs_size, attr_vocabs_size)
            for k in range(out_attr_number)
        ]
        seq_flatten_head_losses = [
            F.cross_entropy(
                input=seq_flatten_pred_logits[k], # shape = (seq_mps_flatten_size, attr_vocabs_size)
                target=seq_flatten_target_logits[:, k], # shape = (seq_mps_flatten_size, )
                reduction='none' # return shape (seq_mps_flatten_size, )
            )
            for k in range(out_attr_number)
        ]
        seq_flatten_head_losses = torch.stack(seq_flatten_head_losses) # (out_attr_number, seq_mps_flatten_size)
        seq_flatten_loss_sum = torch.sum(seq_flatten_head_losses, dim=0) # (seq_mps_flatten_size, )
        # print(
        #     'mps_indices len:', len(mps_indices),
        #     'flatten len:', seq_flatten_loss_sum.shape[0]
        # )
    # with record_function('find_argmins'):
        prev_seq_flatten_index = 0
        for i in range(len(mps_indices_with_begin_and_end) - 1):
            begin_index = mps_indices_with_begin_and_end[i]
            end_index = mps_indices_with_begin_and_end[i+1]
            mps_size = end_index - begin_index
            if mps_size == 2:
                if seq_flatten_loss_sum[prev_seq_flatten_index] < seq_flatten_loss_sum[prev_seq_flatten_index+1]:
                    argmin_loss_sum = 0
                else:
                    argmin_loss_sum = 1
                prev_seq_flatten_index += 2
                indices_replacments.append((begin_index, begin_index + argmin_loss_sum))
            elif mps_size > 2:
                flatten_length = mps_size * (mps_size + 1) // 2 - 1
                cur_mps_flatten_loss_sum = seq_flatten_loss_sum[prev_seq_flatten_index:prev_seq_flatten_index+flatten_length]
                prev_seq_flatten_index += flatten_length
                triu_indices = triu_indices_of_mps[i]
                loss_stacked = torch.full((mps_size-1, mps_size), float('inf'), device=use_device)
                loss_stacked[triu_indices[0], triu_indices[1]] = cur_mps_flatten_loss_sum
                argmin_loss_stacked = torch.argmin(loss_stacked, dim=1)
                for i, m in enumerate(argmin_loss_stacked):
                    indices_replacments.append((begin_index + i, begin_index + m))

    # with record_function('mod_target'):
        # modify target logit such that the target at cur_index is now the target at min_loss_sum_index
        for cur_index, min_loss_sum_index in indices_replacments:
            cpu_target_logit[batch_number, cur_index] = cpu_target_logit[batch_number, min_loss_sum_index]

    head_losses = calc_losses(pred_logit, cpu_target_logit.to(target_logit.device))
    return head_losses

# MidiTransformerDecoder = MyMidiTransformer # old name
