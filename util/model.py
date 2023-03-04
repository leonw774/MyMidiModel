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
from .corpus import TOKEN_ATTR_INDEX, COMPLETE_ATTR_NAME, OUTPUT_ATTR_NAME, array_to_text_list, text_list_to_array


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
            embedding_dropout_rate=0.1
            ) -> None:
        super().__init__()

        assert vocabs.events.text2id[tokens.PADDING_TOKEN_STR] == 0

        self.vocabs = vocabs
        self.max_seq_length = max_seq_length
        self.layers_number = layers_number
        self.attn_heads_number = attn_heads_number
        self.embedding_dim = embedding_dim
        self.input_no_tempo = input_no_tempo
        self.input_no_time_signature = input_no_time_signature

        # set the to True ONLY when generating sample
        self.inferencing = False

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
                padding_idx=vocabs.events.text2id[tokens.PADDING_TOKEN_STR]
                # [...] the embedding vector of padding_idx will default to all zeros [...]
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
        self.project_linear = nn.ModuleList([
            nn.Linear(
                in_features=embedding_dim,
                out_features=vsize
            )
            for vsize in self.logit_vocabs_size
        ])
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-6)

        self.use_linear_attn = use_linear_attn

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
                # 'final_normalization'=True,
                'feature_map': MY_ELU_FEATURE_MAP  # make the model pickle-able
            }
            self.transformer_encoder = TransformerEncoderBuilder.from_dictionary(params).get()
            self.transformer_encoder_inference = RecurrentEncoderBuilder.from_dictionary(params).get()
            make_mirror(self.transformer_encoder, self.transformer_encoder_inference)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=attn_heads_number,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=layer,
                num_layers=layers_number
            )

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
                transformer_output, memory = self.transformer_encoder_inference(emb_sum_dropout, memory)
            else:
                # in fast_transformer's FullMask class, 0 is masked, 1 is keep
                causal_mask = self.causal_mask
                length_mask = FullMask(mask=x[..., 0].ne(0).bool().to(x.device))
                transformer_output = self.transformer_encoder(
                    emb_sum_dropout,
                    causal_mask,
                    length_mask
                )
        else:
            # in pytorch's official implementation, True is masked, False is keep
            causal_mask = self.causal_mask[:x.shape[1], :x.shape[1]].to(x.device)
            length_mask = x[..., 0].eq(0).to(x.device)
            transformer_output = self.transformer_encoder(
                emb_sum_dropout,
                causal_mask,
                length_mask
            )
        transformer_output = self.layer_norm(transformer_output)
        logits = tuple(
            proj(transformer_output) for proj in self.project_linear
        )
        # assert all(not torch.isnan(lg).any() for lg in logits), [torch.isnan(lg).nonzero(as_tuple=True) for lg in logits]
        if self.use_linear_attn and self.inferencing:
            return logits, memory
        else:
            return logits

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


LARGE_NEG_VALUE = -1e8


def adjust_logits_with_context(
        logits: List[Tensor],
        context_text_list: List[str],
        vocabs: Vocabs,
        event_family_indices: Tuple[set]):
    """
        Set to large negative numeric value the logits of the attribute values that would definitily raise exception
        if the next token in context_text_list has them
        and force the position tokens to be ordered increasingly in time
    """
    assert len(context_text_list) > 0, 'Empty context_text_list'
    assert len(logits) == len(OUTPUT_ATTR_NAME) or len(logits) == len(OUTPUT_ATTR_NAME) - 1, 'Bad logits'
    position_method_is_event = len(logits) == len(OUTPUT_ATTR_NAME) - 1

    bos_index = vocabs.events.text2id[tokens.BEGIN_TOKEN_STR]
    sep_index = vocabs.events.text2id[tokens.SEP_TOKEN_STR]
    (
        multinote_indices,
        position_indices,
        measure_indices,
        track_instrument_indices,
        tempo_indices
    ) = event_family_indices

    # adjust event attribute logit

    # BOS token is redundent, will not be used in generation
    logits[TOKEN_ATTR_INDEX['evt']][bos_index] = LARGE_NEG_VALUE
    # predict PAD is not allowed in generation time
    for attr_logits in logits:
        attr_logits[0] = LARGE_NEG_VALUE

    is_head = context_text_list[-1] == tokens.BEGIN_TOKEN_STR or context_text_list[-1][0] == tokens.TRACK_EVENTS_CHAR
    is_sep = context_text_list[-1] == tokens.SEP_TOKEN_STR
    if is_head:
        # if the section is head, then only track-instrument and separater allowed
        # but if it is only BOS, then only track-instrument allowed
        for i in range(vocabs.events.size):
            if i not in track_instrument_indices and (i != sep_index or len(context_text_list) == 0):
                logits[TOKEN_ATTR_INDEX['evt']][i] = LARGE_NEG_VALUE
    elif is_sep:
        # if is separater, then only measure tokens are allowed
        for i in range(vocabs.events.size):
            if i not in measure_indices:
                logits[TOKEN_ATTR_INDEX['evt']][i] = LARGE_NEG_VALUE
    else: # if the section is body
        # the next token's event can not be track-instrument
        for i in track_instrument_indices:
            logits[TOKEN_ATTR_INDEX['evt']][i] = LARGE_NEG_VALUE

        # if the last token in context_text_list token is measure and position method is event
        # then the next token's event can not be multi-note or tempo
        if context_text_list[-1][0] == tokens.MEASURE_EVENTS_CHAR and position_method_is_event:
            for i in multinote_indices.union(tempo_indices):
                logits[TOKEN_ATTR_INDEX['evt']][i] = LARGE_NEG_VALUE

        # this part prohibits the position tokens that is "behind" the current position
        k = len(context_text_list) - 1
        while context_text_list[k][0] not in (tokens.POSITION_EVENTS_CHAR, tokens.MEASURE_EVENTS_CHAR) and k > 0:
            k -= 1
        if context_text_list[k][0] == tokens.POSITION_EVENTS_CHAR:
            cur_position = b36str2int(context_text_list[k][1:])
            for i in position_indices:
                if b36str2int(vocabs.events.id2text[i][1:]) <= cur_position:
                    logits[TOKEN_ATTR_INDEX['evt']][i] = LARGE_NEG_VALUE

        # this part prohibit the tempo tokens if there is already a tempo token in the current position
        k = len(context_text_list) - 1
        while context_text_list[k][0] not in (tokens.POSITION_EVENTS_CHAR, tokens.TEMPO_EVENTS_CHAR) and k > 0:
            k -= 1
        if context_text_list[k][0] == tokens.TEMPO_EVENTS_CHAR:
            for i in tempo_indices:
                logits[TOKEN_ATTR_INDEX['evt']][i] = LARGE_NEG_VALUE

    # adjust track attribute logit
    if not is_head and not is_sep:
        # if the section is body, then only allow the defined track
        sep_index_in_text_list = context_text_list.index(tokens.SEP_TOKEN_STR)
        try:
            max_track_number = max([
                b36str2int(t.split(':')[1])
                for t in context_text_list[1:sep_index_in_text_list]
                if t[0] == tokens.TRACK_EVENTS_CHAR
            ])
        except Exception as e:
            print(context_text_list[1:sep_index_in_text_list])
            raise e
        # start from max_track_number+1+1
        # first +1 is because index 0 is padding
        # second +1 is because we reset any index that greater than max_track_number+1
        for i in range(max_track_number+1+1, vocabs.track_numbers.size):
            logits[TOKEN_ATTR_INDEX['trn']][i] = LARGE_NEG_VALUE

    return logits


def nucleus_sample(probs, threshold):
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus_number = sum(cumsum_probs < threshold) + 1
    if nucleus_number == 1:
        return torch.unsqueeze(sorted_indices[0], dim=0) # returned dimension have to be 1
    else:
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
        use_logit_adjustment: bool = True,
        nucleus_sampling_threshold: float = 1.0,
        print_exception: bool = False,
        show_tqdm: bool = False,
        ignore_pending_note_error: bool = True
        ) -> list:
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
    else:
        start_seq = torch.from_numpy(text_list_to_array([tokens.BEGIN_TOKEN_STR], model.vocabs)).unsqueeze(0).int()

    # get model device
    model_device = next(model.parameters()).device
    prev_training_state = model.training
    model.inference()

    # prepare vocab for logit adjustment
    if use_logit_adjustment:
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
    is_head = tokens.SEP_TOKEN_STR not in text_list

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

            if use_logit_adjustment:
                # prevent many bad format in advance, but not restricting the order of track number in head section
                last_logits = adjust_logits_with_context(last_logits, text_list, model.vocabs, event_family_indices)
            # get_logit_time += time() - bt
            # print('\n'.join([repr(logit) for logit in last_logits]))
            try_count = 0
            while try_count < try_count_limit:
                # bt = time()
                softmaxed_logits = [
                    F.softmax(l / temperature, dim=0)
                    for l in last_logits # l has shape (attr_vocab_size,)
                ]
                if nucleus_sampling_threshold != 1.0:
                    sampled_attrs = [
                        nucleus_sample(p, nucleus_sampling_threshold)
                        for p in softmaxed_logits
                    ]
                else:
                    sampled_attrs = [
                        torch.multinomial(p, 1)
                        for p in softmaxed_logits # l has shape (attr_vocab_size,)
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
                    if not use_logit_adjustment or is_head:
                        # have to append EOS at the end to not raise error
                        piece_to_midi(
                            piece=' '.join(try_text_list + [tokens.END_TOKEN_STR]),
                            nth=model.vocabs.paras['nth'],
                            ignore_pending_note_error=ignore_pending_note_error
                        )
                    # compute complete attribute array and second format checking
                    try_text_list_array, array_to_text_list_memory = text_list_to_array(
                        [try_token_text],
                        vocabs=model.vocabs,
                        input_memory=array_to_text_list_memory,
                        output_memory=True
                    )
                    try_text_list_tensor = torch.from_numpy(
                        try_text_list_array
                    ).unsqueeze(0).int() # shape = (1, 1, complete_attr_num)
                    # check_format_time += time() - bt
                except (AssertionError, ValueError) as e:
                    if print_exception:
                        print(repr(e))
                    try_count += 1
                    # check_format_time += time() - bt
                    continue # keep sampling until no error
                # no error
                if try_token_text == tokens.SEP_TOKEN_STR:
                    is_head = False
                input_seq = try_text_list_tensor
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


def calc_losses(
        pred_logits: List[Tensor],
        target_labels: Tensor,
        loss_function=str,
        weighted_by_nonpadding_number=False) -> List[Tensor]:
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
    if loss_function == 'cross_entropy':
        func = F.cross_entropy
        input_tensors = [logits.transpose(1, 2) for logits in pred_logits]
    elif loss_function == 'nll':
        func = F.nll_loss
        input_tensors = [F.softmax(logits, dim=2).transpose(1, 2) for logits in pred_logits]
    else:
        raise ValueError('argument loss_function should either "cross_entropy" or "nll"')

    head_losses = [
        func(
            input=input_tensor,
            # transpose(1, 2) to become (batch_size, attr_vocab_size, seq_size)
            target=target_labels[..., k], # (batch_size, seq_size)
            ignore_index=0, # assume padding is index 0
            reduction=('sum' if weighted_by_nonpadding_number else 'mean')
            # because some attributes have more non-padding occurence than others
            # we can reflect this by them having different "weight" of loss using "sum"
        )
        for k, input_tensor in enumerate(input_tensors)
    ]
    if weighted_by_nonpadding_number:
        token_number = target_labels.shape[0] * target_labels.shape[1]
        head_losses = [hl / token_number for hl in head_losses]
    return head_losses


def calc_permutable_subseq_losses(
        pred_logits: List[Tensor],
        target_labels: Tensor,
        batched_mps_indices: List,
        loss_function: str,
        weighted_by_nonpadding_number=False):
    """
        pred_logits is a list
        - length: out_attr_num
        - elements are tenors with shape: (batch_size, seq_size, attr_vocabs_size)
        target_labels has shape: (batch_size, seq_size, out_attr_num)
        mps_indices is a numpy object array of numpy int16 arrays in varying lengths
        return a list of losses of each head
    """

    # because the target is prediction shifted left by 1 index
    # the first (0th) prediction in a mps can have (mps_size-1) possible labels
    # the last (mps_size-th) prediction in a mps can have (next_mps_size) possible labels
    # when we we are finding permutation of target, all index need to be subtracted by 1
    # so that their corrresponding target is the beginning of mps
    batched_mps_indices_as_list = [
        [m-1 for m in mps_indices] + [target_labels.shape[1]-1]
        for mps_indices in batched_mps_indices
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

    head_losses = calc_losses(
        pred_logits,
        cpp_modified_target_labels.to(target_labels.device),
        loss_function,
        weighted_by_nonpadding_number
    )
    return head_losses


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
