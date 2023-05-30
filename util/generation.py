from typing import List, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from . import tokens
from .tokens import b36str2int
from .midi import piece_to_midi
from .vocabs import Vocabs
from .corpus import (ATTR_NAME_INDEX, ALL_ATTR_NAMES, ESSENTAIL_ATTR_NAMES, OUTPUTABLE_ATTR_NAMES,
                     array_to_text_list, text_list_to_array)
from .model import MyMidiTransformer


# the same code from `dataset.py`
def permute_track_number(array: np.ndarray, track_number_size: int) -> np.ndarray:
    max_track_number = track_number_size - 1
    # add one because there is a padding token at the beginning of the vocab
    perm = np.random.permutation(track_number_size-1) + 1
    # view track number col
    new_array = array.copy()
    piece_trn_column = new_array[:, ATTR_NAME_INDEX['trn']]
    piece_trn_column_expand = np.asarray([
        (piece_trn_column == i + 1) for i in range(max_track_number)
    ])
    # permute track number
    for i in range(max_track_number):
        piece_trn_column[piece_trn_column_expand[i]] = perm[i]
    return new_array


def adjust_probs_with_context(
        probs_list: List[torch.Tensor],
        context_text_list: List[str],
        vocabs: Vocabs,
        event_family_indices: Tuple[set]):
    """
        Set to zero to the prob of the attribute values that would definitily raise exception
        if the next token in context_text_list has them and force the position tokens to be ordered increasingly in time
    """
    assert len(context_text_list) > 0, 'Empty context_text_list'
    assert len(probs_list) == len(OUTPUTABLE_ATTR_NAMES) or len(probs_list) == len(OUTPUTABLE_ATTR_NAMES) - 1, 'Bad probs_list'

    bos_index = vocabs.events.text2id[tokens.BEGIN_TOKEN_STR]
    sep_index = vocabs.events.text2id[tokens.SEP_TOKEN_STR]
    multinote_indices, position_indices, measure_indices, track_indices, tempo_indices = event_family_indices

    # BOS token is redundent, will not be used in generation
    probs_list[ATTR_NAME_INDEX['evt']][bos_index] = 0
    # predict PAD is not allowed in generation time
    for attr_probs in probs_list:
        attr_probs[0] = 0

    is_head = context_text_list[-1] == tokens.BEGIN_TOKEN_STR or context_text_list[-1][0] == tokens.TRACK_EVENTS_CHAR
    is_sep = context_text_list[-1] == tokens.SEP_TOKEN_STR
    if is_head: # if the section is head and not sep
        if len(context_text_list) == 1:
            # only track-instrument allowed
            not_track_indices = [idx for idx in range(vocabs.events.size) if idx not in track_indices]
            probs_list[ATTR_NAME_INDEX['evt']][not_track_indices] = 0
        elif 1 < len(context_text_list) < vocabs.track_numbers.size:
            # only track-instrument and separater allowed
            not_track_or_sep_indices = [
                idx
                for idx in range(vocabs.events.size)
                if idx not in track_indices and idx != sep_index
            ]
            probs_list[ATTR_NAME_INDEX['evt']][not_track_or_sep_indices] = 0
            # prevent generating repeating track number
            used_track_number =  [
                b36str2int(context_text_list[i].split(':')[1]) + 1 # the id:value of track number is 0:padding, 1:0, 2:1, ...
                for i in range(1, len(context_text_list))
            ]
            # assert all(t < vocabs.track_numbers.size for t in used_track_number), \
            #     f'context_text_list: {context_text_list}\nused_track_number: {used_track_number}'
            probs_list[ATTR_NAME_INDEX['trn']][used_track_number] = 0
        else:
            # only separater allowed
            probs_list[ATTR_NAME_INDEX['evt']][:sep_index] = 0
            probs_list[ATTR_NAME_INDEX['evt']][sep_index+1:] = 0

    elif is_sep: # if is separater, then only measure tokens are allowed
        for i in range(vocabs.events.size):
            if i not in measure_indices:
                probs_list[ATTR_NAME_INDEX['evt']][i] = 0

    else: # if the section is body
        # the next token's event can not be track-instrument or sep
        probs_list[ATTR_NAME_INDEX['evt']][list(track_indices)] = 0
        probs_list[ATTR_NAME_INDEX['evt']][sep_index] = 0

        # if the last token in context_text_list is measure
        # then the next token's event can not be multi-note or tempo
        if context_text_list[-1][0] == tokens.MEASURE_EVENTS_CHAR:
            multinote_and_tempo_indices = multinote_indices.union(tempo_indices)
            probs_list[ATTR_NAME_INDEX['evt']][list(multinote_and_tempo_indices)] = 0

        # if the last token in context_text_list is position
        # then the next token's event can not be measure
        elif context_text_list[-1][0] == tokens.POSITION_EVENTS_CHAR:
            probs_list[ATTR_NAME_INDEX['evt']][list(position_indices)] = 0

        # if the last token in context_text_list is note or multinote
        # the next token's event can not be tempo
        elif context_text_list[-1][0] in (tokens.NOTE_EVENTS_CHAR, tokens.MULTI_NOTE_EVENTS_CHAR):
            probs_list[ATTR_NAME_INDEX['evt']][list(tempo_indices)] = 0

        # this prohibits the position tokens that is "behind" the current position
        k = len(context_text_list) - 1
        while context_text_list[k][0] not in (tokens.POSITION_EVENTS_CHAR, tokens.MEASURE_EVENTS_CHAR) and k > 0:
            k -= 1
        if context_text_list[k][0] == tokens.POSITION_EVENTS_CHAR:
            cur_position = b36str2int(context_text_list[k][1:])
            for i in position_indices:
                if b36str2int(vocabs.events.id2text[i][1:]) <= cur_position:
                    probs_list[ATTR_NAME_INDEX['evt']][i] = 0

        # this prohibit the tempo tokens if there is already a tempo token in the current position
        k = len(context_text_list) - 1
        while context_text_list[k][0] not in (tokens.POSITION_EVENTS_CHAR, tokens.TEMPO_EVENTS_CHAR) and k > 0:
            k -= 1
        if context_text_list[k][0] == tokens.TEMPO_EVENTS_CHAR:
            probs_list[ATTR_NAME_INDEX['evt']][list(tempo_indices)] = 0

        # only allow the defined track number
        try:
            sep_index_in_text_list = context_text_list.index(tokens.SEP_TOKEN_STR)
            used_track_number_indices = {
                b36str2int(t.split(':')[1]) + 1
                for t in context_text_list[1:sep_index_in_text_list]
                if t[0] == tokens.TRACK_EVENTS_CHAR
            }
        except Exception as e:
            print(context_text_list)
            raise e
        unused_track_number_indices = [idx for idx in range(vocabs.track_numbers.size) if idx not in used_track_number_indices]
        probs_list[ATTR_NAME_INDEX['trn']][unused_track_number_indices] = 0

    # re-normalized probs
    normed_probs_list = [p / p.sum() for p in probs_list]
    # if in some attribute all events are fill with zero, something is wrong
    assert not any([torch.isnan(p).any() for p in normed_probs_list]), \
        f'Text List: {context_text_list}\nAdjusted: {probs_list}\nRe-normalized: {normed_probs_list}'
    return normed_probs_list


def nucleus_sampling(probs: torch.Tensor, threshold: float):
    assert 0 < threshold <= 1.0
    if threshold == 1.0:
        return torch.multinomial(probs, 1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus_number = torch.sum(cumsum_probs < threshold)
    if nucleus_number == 0:
        return torch.unsqueeze(sorted_indices[0], dim=0) # returned dimension have to be 1
    nucleus_probs = sorted_probs[:nucleus_number]
    sampled_nuclei = torch.multinomial(nucleus_probs, 1)
    sampled_index = sorted_indices[sampled_nuclei]
    return sampled_index


def generate_piece(
        model: MyMidiTransformer,
        steps: int,
        start_seq: Union[torch.Tensor, None] = None,
        softmax_temperature: Union[List[float], None] = None,
        try_count_limit: int = 1000,
        use_prob_adjustment: bool = True,
        nucleus_sampling_threshold:  Union[List[float], None] = None,
        print_exception: bool = False,
        show_tqdm: bool = False,
        ignore_pending_note_error: bool = True) -> List[str]:
    """
        Expect start_seq to be Tensor with shape: (1, seq_size, all_attr_number) or None
        If start_seq is None, will use `text_list_to_array([BEGIN_TOKEN_STR])` as start_seq
        ignore_pending_note_error is default to be True, otherwise model cannot generate continuing note
        return the text list of the generated piece
    """
    if start_seq is not None:
        exception_msg = f'start_seq\'s shape have to be (1, seq_length, all_attr_number), get {start_seq.shape}'
        if len(start_seq.shape) != 3:
            raise AssertionError(exception_msg)
        elif start_seq.shape[0] != 1 or start_seq.shape[1] < 1 or start_seq.shape[2] != len(ALL_ATTR_NAMES):
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
        # event_family_indices = (multinote_indices, position_indices, measure_indices, track_indices, tempo_indices)
        event_family_indices = (set(), set(), set(), set(), set())
        for t, i in model.vocabs.events.text2id.items():
            if t[0] == tokens.NOTE_EVENTS_CHAR or t[0] == tokens.MULTI_NOTE_EVENTS_CHAR:
                event_family_indices[0].add(i)
            elif t[0] == tokens.POSITION_EVENTS_CHAR:
                event_family_indices[1].add(i)
            elif t[0] == tokens.MEASURE_EVENTS_CHAR:
                event_family_indices[2].add(i)
            elif t[0] == tokens.TRACK_EVENTS_CHAR:
                event_family_indices[3].add(i)
            elif t[0] == tokens.TEMPO_EVENTS_CHAR:
                event_family_indices[4].add(i)

    if softmax_temperature is None:
        softmax_temperature = [1.0] * len(model.output_attrs_indices)
    elif isinstance(softmax_temperature, list):
        if len(softmax_temperature) == 1:
            softmax_temperature = softmax_temperature * len(model.output_attrs_indices)
        elif len(softmax_temperature) != len(model.output_attrs_indices):
            raise ValueError('length of softmax_temperature can only be 1 or out_attr_num. Get ' + str(softmax_temperature))

    if nucleus_sampling_threshold is None:
        nucleus_sampling_threshold = [1.0] * len(model.output_attrs_indices)
    elif isinstance(nucleus_sampling_threshold, list):
        if len(nucleus_sampling_threshold) == 1:
            nucleus_sampling_threshold = nucleus_sampling_threshold * len(model.output_attrs_indices)
        elif len(nucleus_sampling_threshold) != len(model.output_attrs_indices):
            raise ValueError('length of nucleus_sampling_threshold can only be 1 or out_attr_num. Get ' + str(nucleus_sampling_threshold))

    text_list = array_to_text_list(start_seq[0].cpu().numpy(), vocabs=model.vocabs)

    input_seq = start_seq
    primer_length = input_seq.shape[1]
    max_gen_step = min(model.max_seq_length, steps+primer_length) - primer_length
    if max_gen_step <= 0:
        print('generate_piece: start_seq is longer than model\'s max_seq_length. No generation is performed.')
        if text_list[-1] != tokens.END_TOKEN_STR:
            text_list.append(tokens.END_TOKEN_STR)
        return text_list
    end_with_end_token = False
    recurrent_memory = None
    _, array_memory = text_list_to_array(text_list, model.vocabs, input_memory=None, output_memory=True)

    # build memory for primer if use linear transformer
    if model.use_linear_attn:
        for i in range(1, primer_length):
            _, recurrent_memory = model(model.to_input_attrs(input_seq[:, :i]).to(model_device), recurrent_memory)

    with torch.no_grad():
        for _ in tqdm(range(max_gen_step), disable=not show_tqdm):
            input_seq_in_attr_device = model.to_input_attrs(input_seq).to(model_device)
            if model.use_linear_attn:
                # return batched_last_logits because inferencing is True
                batched_last_logits, recurrent_memory = model(input_seq_in_attr_device, recurrent_memory)
                last_logits = [
                    l[0].to('cpu') # to cpu, if was cuda (likely)
                    for l in batched_last_logits # l has shape (1, attr_vocab_size)
                ]
            else:
                batched_logits = model(input_seq_in_attr_device)
                last_logits = [
                    l[0, -1].to('cpu')
                    for l in batched_logits # l has shape (1, seq_length, attr_vocab_size)
                ]

            # re-arrange output order to array order
            array_order_logits = [0] * len(ESSENTAIL_ATTR_NAMES)
            for attr_name in ESSENTAIL_ATTR_NAMES:
                model_output_index = model.output_attr_names.index(attr_name)
                array_order_logits[ATTR_NAME_INDEX[attr_name]] = last_logits[model_output_index]

            probs_list = [
                F.softmax(l / t, dim=0)
                for l, t in zip(array_order_logits, softmax_temperature) # l has shape (attr_vocab_size,)
            ]
            if use_prob_adjustment:
                # prevent many bad format in advance
                probs_list = adjust_probs_with_context(probs_list, text_list, model.vocabs, event_family_indices)
            # print('\n'.join([repr(p) for p in probs]))

            try_count = 0
            while try_count < try_count_limit:
                sampled_attrs = [
                    nucleus_sampling(probs, threshold)
                    for probs, threshold in zip(probs_list, nucleus_sampling_threshold)
                ]
                # print(sampled_attrs)
                try_token = torch.stack(sampled_attrs, dim=1) # shape = (1, out_attr_num)
                try_token_text = array_to_text_list(try_token.cpu().numpy(), vocabs=model.vocabs)[0]
                # print(try_text_list)
                if try_token_text == tokens.END_TOKEN_STR:
                    text_list = text_list + [try_token_text]
                    # if sampled EOS, then dont check. just end
                    break

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
                    # compute full attribute array and second format checking
                    try_array, try_array_memory = text_list_to_array(
                        [try_token_text],
                        vocabs=model.vocabs,
                        input_memory=array_memory,
                        output_memory=True
                    )
                    try_tensor = torch.from_numpy(try_array.astype('int32')).unsqueeze(0).int()
                    # shape = (1, seq_length, all_attr_num)
                except (AssertionError, ValueError) as e:
                    if print_exception:
                        print(repr(e))
                    try_count += 1
                    continue # keep sampling until no error

                text_list = try_text_list
                array_memory = try_array_memory
                input_seq = try_tensor
                break
            # end while sample and try

            if try_count == try_count_limit:
                if print_exception:
                    print('Exceeded try count limit:', try_count_limit)
                break

            if try_token_text == tokens.END_TOKEN_STR:
                end_with_end_token = True
                break

        # end for each step
    # end with torch.no_grad

    if not end_with_end_token:
        text_list.append(tokens.END_TOKEN_STR)

    model.train(prev_training_state)
    return text_list
