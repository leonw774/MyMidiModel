from typing import List, Tuple, Union
from time import time

import torch
from torch import nn, Tensor
import torch.nn.functional as F
# from torch.profiler import record_function
from tqdm import tqdm

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
from .tokens import b36str2int
from .midi import piece_to_midi
from .vocabs import Vocabs
from .corpus import ATTR_NAME_INDEX, COMPLETE_ATTR_NAME, OUTPUT_ATTR_NAME, array_to_text_list, text_list_to_array


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

        self.input_attrs_indices = [ATTR_NAME_INDEX[fname] for fname in COMPLETE_ATTR_NAME]
        if input_context:
            if input_instruments:
                pass
            else:
                self.input_attrs_indices.remove(ATTR_NAME_INDEX['instruments'])
        else:
            if input_instruments:
                self.input_attrs_indices.remove(ATTR_NAME_INDEX['tempos'])
                self.input_attrs_indices.remove(ATTR_NAME_INDEX['time_signatures'])
            else:
                self.input_attrs_indices.remove(ATTR_NAME_INDEX['instruments'])
                self.input_attrs_indices.remove(ATTR_NAME_INDEX['tempos'])
                self.input_attrs_indices.remove(ATTR_NAME_INDEX['time_signatures'])

        self.embedding_vocabs_size = [
            self.vocabs.max_measure_number + 1 # plus one for padding
            if COMPLETE_ATTR_NAME[idx] == 'measure_numbers' else
            (
                self.vocabs.max_mps_number + 1
                if COMPLETE_ATTR_NAME[idx] == 'mps_numbers' else
                getattr(self.vocabs, COMPLETE_ATTR_NAME[idx]).size
            )
            for idx in self.input_attrs_indices
        ]

        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=vsize,
                embedding_dim=embedding_dim,
                padding_idx=self.vocabs.events.text2id[tokens.PADDING_TOKEN_STR]
                # [...] the embedding vector of padding_idx will default to all zeros [...]
            )
            for vsize in self.embedding_vocabs_size
        ])
        self.embedding_dropout = nn.Dropout(embedding_dropout_rate)

        ######## Outputs

        self.output_attrs_indices = [
            ATTR_NAME_INDEX[fname]
            for fname in OUTPUT_ATTR_NAME
        ]
        if not output_instruments:
            self.output_attrs_indices.remove(ATTR_NAME_INDEX['instruments'])

        self.logit_vocabs_size = [
            getattr(self.vocabs, OUTPUT_ATTR_NAME[i]).size
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
        # expect batch_input_seqs has shape: (batch_size, seq_size, complete_attr_num)
        return batch_input_seqs[..., self.input_attrs_indices]

    def to_output_attrs(self, batch_input_seqs: Tensor) -> Tensor:
        # expect batch_input_seqs has shape: (batch_size, seq_size, complete_attr_num)
        return batch_input_seqs[..., self.output_attrs_indices]

    def forward(self, x, memory=None):
        # x has shape: (batch_size, seq_size, in_attr_number)
        if self.use_linear_attn and self.inferencing:
            # The recurrent model only need to receive the last element with size of (batch_size, embed_size)
            x = x[:, -1:] # become (batch_size, 1, embed_size)
        emb_sum = sum(
            emb(x[..., i]) for i, emb in enumerate(self.embeddings)
        )
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
            reduction='sum'
            # some attributes have more non-padding occurence than others
            # we reflect this by them having different "weight" of loss by using 'sum'
        )
        for k, logits in enumerate(pred_logits)
    ]

    # "normalize" the losses by number of non-padding targets
    number_of_nonpadding = torch.count_nonzero(target_labels)
    head_losses = [head_l / number_of_nonpadding for head_l in head_losses]
    loss = sum(head_losses)

    return loss, head_losses


def adjust_probs_with_context(
        probs: List[Tensor],
        context_text_list: List[str],
        vocabs: Vocabs,
        event_family_indices: Tuple[set],
        permute_mps: bool = True):
    """
        Set to zero to the probs of the attribute values that would definitily raise exception
        if the next token in context_text_list has them and force the position tokens to be ordered increasingly in time
    """
    assert len(context_text_list) > 0, 'Empty context_text_list'
    assert len(probs) == len(OUTPUT_ATTR_NAME) or len(probs) == len(OUTPUT_ATTR_NAME) - 1, 'Bad probs'

    bos_index = vocabs.events.text2id[tokens.BEGIN_TOKEN_STR]
    sep_index = vocabs.events.text2id[tokens.SEP_TOKEN_STR]
    (
        multinote_indices,
        position_indices,
        measure_indices,
        track_instrument_indices,
        tempo_indices
    ) = event_family_indices

    # BOS token is redundent, will not be used in generation
    probs[ATTR_NAME_INDEX['evt']][bos_index] = 0
    # predict PAD is not allowed in generation time
    for attr_probs in probs:
        attr_probs[0] = 0

    is_head = context_text_list[-1] == tokens.BEGIN_TOKEN_STR or context_text_list[-1][0] == tokens.TRACK_EVENTS_CHAR
    is_sep = context_text_list[-1] == tokens.SEP_TOKEN_STR
    if is_head:
        # if the section is head, then only track-instrument and separater allowed
        for i in range(vocabs.events.size):
            if i not in track_instrument_indices and (i != sep_index or len(context_text_list) == 1):
                probs[ATTR_NAME_INDEX['evt']][i] = 0
        # prevent generating repeating track number
        used_track_number =  [
            b36str2int(context_text_list[i].split(':')[1]) + 1 # the id of track number is 0:padding, 1:0, 2:1, ...
            for i in range(1, len(context_text_list))
        ]
        probs[ATTR_NAME_INDEX['trn']][used_track_number] = 0
        if len(used_track_number) == vocabs.track_numbers.size - 1:
            probs[ATTR_NAME_INDEX['trn']][0] = 1

    elif is_sep:
        # if is separater, then only measure tokens are allowed
        for i in range(vocabs.events.size):
            if i not in measure_indices:
                probs[ATTR_NAME_INDEX['evt']][i] = 0
    else: # if the section is body
        # the next token's event can not be track-instrument or sep
        probs[ATTR_NAME_INDEX['evt']][list(track_instrument_indices)] = 0
        probs[ATTR_NAME_INDEX['evt']][sep_index] = 0

        # if the last token in context_text_list token is measure
        # then the next token's event can not be multi-note or tempo
        if context_text_list[-1][0] == tokens.MEASURE_EVENTS_CHAR:
            multinote_and_tempo_indices = multinote_indices.union(tempo_indices)
            for i in range(vocabs.events.size):
                if i not in multinote_and_tempo_indices:
                    probs[ATTR_NAME_INDEX['evt']][i] = 0

        # this prohibits the position tokens that is "behind" the current position
        k = len(context_text_list) - 1
        while context_text_list[k][0] not in (tokens.POSITION_EVENTS_CHAR, tokens.MEASURE_EVENTS_CHAR) and k > 0:
            k -= 1
        if context_text_list[k][0] == tokens.POSITION_EVENTS_CHAR:
            cur_position = b36str2int(context_text_list[k][1:])
            for i in position_indices:
                if b36str2int(vocabs.events.id2text[i][1:]) <= cur_position:
                    probs[ATTR_NAME_INDEX['evt']][i] = 0

        # this prohibit the tempo tokens if there is already a tempo token in the current position
        k = len(context_text_list) - 1
        while context_text_list[k][0] not in (tokens.POSITION_EVENTS_CHAR, tokens.TEMPO_EVENTS_CHAR) and k > 0:
            k -= 1
        if context_text_list[k][0] == tokens.TEMPO_EVENTS_CHAR:
            probs[ATTR_NAME_INDEX['evt']][list(tempo_indices)] = 0

        # this prohibits the multi-note token to have the track number smaller than the previous multi-note token
        # if not permute_mps:
        #     prev_token_type = context_text_list[-1][0]
        #     if prev_token_type == tokens.MULTI_NOTE_EVENTS_CHAR or prev_token_type == tokens.NOTE_EVENTS_CHAR:
        #         prev_token_track_number = b36str2int(context_text_list[-1].split(':')[-1])
        #         # prohibits track number from '0' to 'prev_token_track_number - 1'
        #         # id of track number '0' is 1
        #         # id of track number 'prev_token_track_number - 1' is prev_token_track_number
        #         probs[ATTR_NAME_INDEX['trn']][1:prev_token_track_number+1] = 0

        # only allow the defined track number
        sep_index_in_text_list = context_text_list.index(tokens.SEP_TOKEN_STR)
        try:
            used_track_number = [
                b36str2int(t.split(':')[1]) + 1
                for t in context_text_list[1:sep_index_in_text_list]
                if t[0] == tokens.TRACK_EVENTS_CHAR
            ]
        except Exception as e:
            print(context_text_list[1:sep_index_in_text_list])
            raise e
        unused_track_number = list(set(range(vocabs.track_numbers.size)).difference(used_track_number))
        probs[ATTR_NAME_INDEX['trn']][unused_track_number] = 0

    # re-normalized it
    normed_probs = [p / p.sum() for p in probs]
    # if in some attribute all events are fill with zero, something is wrong
    assert not any([torch.isnan(p).any() for p in normed_probs]), \
        f'Text List: {context_text_list}\nAdjusted: {probs}\nRe-normalized: {normed_probs}'
    return normed_probs


def nucleus_sampling(probs, threshold):
    if threshold == 1.0:
        return torch.multinomial(probs, 1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus_number = sum(cumsum_probs < threshold) + 1
    if nucleus_number == 1:
        return torch.unsqueeze(sorted_indices[0], dim=0) # returned dimension have to be 1
    nucleus_probs = sorted_probs[:nucleus_number]
    sampled_nuclei = torch.multinomial(nucleus_probs, 1)
    sampled_index = sorted_indices[sampled_nuclei]
    return sampled_index


def generate_sample(
        model: MyMidiTransformer,
        steps: int,
        start_seq: Union[Tensor, None] = None,
        temperature: float = 1.0,
        try_count_limit: int = 1000,
        use_prob_adjustment: bool = True,
        nucleus_sampling_threshold: float = 1.0,
        print_exception: bool = False,
        show_tqdm: bool = False,
        ignore_pending_note_error: bool = True) -> list:
    """
        Expect start_seq to be Tensor with shape: (1, seq_size, complete_attr_number) or None
        If start_seq is None, will use `text_list_to_array([BEGIN_TOKEN_STR])` as start_seq
        ignore_pending_note_error is default to be True, otherwise model cannot generate continuing note
        return the text list of the generated piece
    """
    if start_seq is not None:
        exception_msg = f'start_seq\'s shape have to be (1, seq_length, complete_attr_number), get {start_seq.shape}'
        if len(start_seq.shape) != 3:
            raise AssertionError(exception_msg)
        elif start_seq.shape[0] != 1 or start_seq.shape[1] < 1 or start_seq.shape[2] != len(COMPLETE_ATTR_NAME):
            raise AssertionError(exception_msg)
        assert start_seq.dtype == torch.int32
    else:
        start_seq = torch.from_numpy(
            text_list_to_array([tokens.BEGIN_TOKEN_STR], model.vocabs).astype('int32') # was uint16, which torch don't suuport
        ).unsqueeze(0).int()

    # get model device
    model_device = next(model.parameters()).device
    prev_training_state = model.training
    model.inference()

    # prepare famlity indices for probs adjustment
    if use_prob_adjustment:
        multinote_indices = set()
        position_indices = set()
        measure_indices = set()
        track_instrument_indices = set()
        tempo_indices = set()
        for t, i in model.vocabs.events.text2id.items():
            if t[0] == tokens.NOTE_EVENTS_CHAR or t[0] == tokens.MULTI_NOTE_EVENTS_CHAR:
                multinote_indices.add(i)
            elif t[0] == tokens.POSITION_EVENTS_CHAR:
                position_indices.add(i)
            elif t[0] == tokens.MEASURE_EVENTS_CHAR:
                measure_indices.add(i)
            elif t[0] == tokens.TEMPO_EVENTS_CHAR:
                tempo_indices.add(i)
            elif t[0] == tokens.TRACK_EVENTS_CHAR:
                track_instrument_indices.add(i)
        event_family_indices = (
            multinote_indices,
            position_indices,
            measure_indices,
            track_instrument_indices,
            tempo_indices
        )

    text_list = array_to_text_list(start_seq[0].cpu().numpy(), vocabs=model.vocabs, is_output=False)
    # is_head = (tokens.SEP_TOKEN_STR not in text_list)

    input_seq = start_seq
    primer_length = input_seq.shape[1]
    max_gen_step = min(model.max_seq_length, steps+primer_length) - primer_length
    end_with_end_token = False
    recurrent_memory = None
    _, array_to_text_list_memory = text_list_to_array(
        text_list,
        model.vocabs,
        input_memory=None,
        output_memory=True
    )

    # build memory for primer first if use linear transformer
    if model.use_linear_attn:
        for i in range(1, primer_length):
            _, recurrent_memory = model(model.to_input_attrs(input_seq[:, :i]).to(model_device), recurrent_memory)

    # get_logit_time = 0
    # get_sample_time = 0
    # check_format_time = 0
    # print(seq.shape)
    with torch.no_grad():
        for _ in tqdm(range(max_gen_step), disable=not show_tqdm):
            # return batched_last_logits because inferencing is True
            # bt = time()
            input_seq_in_attr_device = model.to_input_attrs(input_seq).to(model_device)
            if model.use_linear_attn:
                batched_last_logits, recurrent_memory = model(input_seq_in_attr_device, recurrent_memory)
                last_logits = [
                    l[0].to('cpu') # to cpu, if was cuda (likely)
                    # l has shape (1, attr_vocab_size)
                    for l in batched_last_logits
                ]
            else:
                batched_logits = model(input_seq_in_attr_device)
                last_logits = [
                    l[0, -1].to('cpu')
                    # l has shape (1, seq_length, attr_vocab_size)
                    for l in batched_logits
                ]

            last_probs = [
                F.softmax(l / temperature, dim=0)
                for l in last_logits # l has shape (attr_vocab_size,)
            ]
            if use_prob_adjustment:
                # prevent many bad format in advance
                last_probs = adjust_probs_with_context(
                    last_probs,
                    text_list,
                    model.vocabs,
                    event_family_indices,
                    model.permute_mps
                )
            # get_logit_time += time() - bt
            # print('\n'.join([repr(p) for p in last_probs]))
            try_count = 0
            while try_count < try_count_limit:
                # bt = time()
                sampled_attrs = [
                    nucleus_sampling(p, nucleus_sampling_threshold)
                    for p in last_probs
                ]

                # print(sampled_attrs)
                try_token = torch.stack(sampled_attrs, dim=1) # shape = (1, out_attr_num)
                # print([ model.vocabs.to_dict()[OUTPUT_ATTR_NAME[i]]['id2text'][int(f)] for i, f in enumerate(try_token[0]) ])
                try_token_text = array_to_text_list(try_token.cpu().numpy(), vocabs=model.vocabs, is_output=True)[0]
                # print(try_text_list)
                if try_token_text == tokens.END_TOKEN_STR:
                    text_list = text_list + [try_token_text]
                    # if sampled EOS, then dont check. just end
                    break
                # get_sample_time = time() - bt
                # bt = time()
                try:
                    try_text_list = text_list + [try_token_text]
                    # format checking
                    if not use_prob_adjustment:
                        # have to append EOS at the end to not raise error
                        piece_to_midi(
                            piece=' '.join(try_text_list + [tokens.END_TOKEN_STR]),
                            nth=model.vocabs.paras['nth'],
                            ignore_pending_note_error=ignore_pending_note_error
                        )
                    # compute complete attribute array and second format checking
                    try_text_list_array, try_array_to_text_list_memory = text_list_to_array(
                        [try_token_text],
                        vocabs=model.vocabs,
                        input_memory=array_to_text_list_memory,
                        output_memory=True
                    )
                    try_text_list_tensor = torch.from_numpy(
                        try_text_list_array.astype('int32')
                    ).unsqueeze(0).int() # shape = (1, 1, complete_attr_num)
                    # check_format_time += time() - bt
                except (AssertionError, ValueError) as e:
                    if print_exception:
                        print(repr(e))
                    try_count += 1
                    # check_format_time += time() - bt
                    continue # keep sampling until no error
                # no error
                input_seq = try_text_list_tensor
                array_to_text_list_memory = try_array_to_text_list_memory
                text_list = try_text_list
                break
            # end while try
            # print(text_list)
            if try_count == try_count_limit:
                if print_exception:
                    print('Exceeded try count limit:', try_count_limit)
                break

            if text_list[-1] == tokens.END_TOKEN_STR:
                end_with_end_token = True
                break
        # end for each step
    # end with torch.no_grad
    # print('get_logit_time', get_logit_time)
    # print('get_sample_time', get_sample_time)
    # print('check_format_time', check_format_time)

    if not end_with_end_token:
        text_list.append(tokens.END_TOKEN_STR)

    model.train(prev_training_state)
    return text_list


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
