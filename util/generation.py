from typing import List, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .tokens import (
    b36strtoi,
    BEGIN_TOKEN_STR, END_TOKEN_STR, SEP_TOKEN_STR,
    TRACK_EVENTS_CHAR, MEASURE_EVENTS_CHAR, TEMPO_EVENTS_CHAR,
    POSITION_EVENTS_CHAR, MULTI_NOTE_EVENTS_CHAR, NOTE_EVENTS_CHAR
)
from .midi import piece_to_midi
from .vocabs import Vocabs
from .corpus import (
    ATTR_NAME_INDEX, ALL_ATTR_NAMES, OUTPUT_ATTR_NAMES,
    array_to_text_list, text_list_to_array
)
from .model import MyMidiTransformer


# the same code from `dataset.py`
def permute_track_number(array: NDArray, track_number_size: int) -> NDArray:
    new_array = array.copy()
    max_track_number = track_number_size - 1
    # add one because there is a padding token at the beginning of the vocab
    perm = np.concatenate(([0], np.random.permutation(max_track_number) + 1))
    trn_column = new_array[:, ATTR_NAME_INDEX['trn']]
    for i, trn in enumerate(trn_column):
        trn_column[i] = perm[trn]
    return new_array


def adjust_logits(
        logits: List[torch.Tensor],
        text_list: List[str],
        vocabs: Vocabs,
        event_family_ids: Tuple[set]) -> List[torch.Tensor]:
    """
    Set to -INF to the logits of the attribute values that would
    definitily raise exception if the next token use them.

    Also force the position tokens to be ordered increasingly.
    """
    assert len(text_list) > 0, 'Empty text_list'
    assert len(logits) == len(OUTPUT_ATTR_NAMES), 'Bad probs'

    neg_inf = float('-inf')

    bos_index = vocabs.events.text2id[BEGIN_TOKEN_STR]
    sep_index = vocabs.events.text2id[SEP_TOKEN_STR]
    (multinote_ids,
     position_ids,
     measure_ids,
     track_ids,
     tempo_ids
    ) = event_family_ids

    # BOS token is redundent, will not be used in generation
    logits[ATTR_NAME_INDEX['evt']][bos_index] = neg_inf
    # predict PAD is not allowed in generation time
    for attr_logits in logits:
        attr_logits[0] = neg_inf

    is_head = (
        text_list[-1] == BEGIN_TOKEN_STR
        or text_list[-1][0] == TRACK_EVENTS_CHAR
    )
    is_sep = text_list[-1] == SEP_TOKEN_STR
    if is_head: # if the section is head and not sep
        if len(text_list) == 1:
            # only track-instrument allowed
            not_track_ids = [
                index
                for index in range(vocabs.events.size)
                if index not in track_ids
            ]
            logits[ATTR_NAME_INDEX['evt']][not_track_ids] = neg_inf
        elif 1 < len(text_list) < vocabs.track_numbers.size:
            # only track-instrument and separater allowed
            not_track_or_sep_ids = [
                index
                for index in range(vocabs.events.size)
                if index not in track_ids and index != sep_index
            ]
            logits[ATTR_NAME_INDEX['evt']][not_track_or_sep_ids] = neg_inf
            # prevent generating repeating track number
            used_track_number =  [
                # the index:value of track number is 0:padding, 1:0, 2:1, ...
                b36strtoi(text_list[i].split(':')[1]) + 1
                for i in range(1, len(text_list))
            ]
            # assert all(
            #     t < vocabs.track_numbers.size
            #     for t in used_track_number
            # ), (
            #     f'text_list: {text_list}\n'
            #     f'used_track_number: {used_track_number}'
            # )
            logits[ATTR_NAME_INDEX['trn']][used_track_number] = neg_inf
        else:
            # only separater allowed
            logits[ATTR_NAME_INDEX['evt']][:sep_index] = neg_inf
            logits[ATTR_NAME_INDEX['evt']][sep_index+1:] = neg_inf

    elif is_sep: # if is separater, then only measure tokens are allowed
        for i in range(vocabs.events.size):
            if i not in measure_ids:
                logits[ATTR_NAME_INDEX['evt']][i] = neg_inf

    else: # if the section is body
        # the next token's event can not be track-instrument or sep
        logits[ATTR_NAME_INDEX['evt']][list(track_ids)] = neg_inf
        logits[ATTR_NAME_INDEX['evt']][sep_index] = neg_inf

        # if the last token in text_list is measure
        # then the next token's event can not be multi-note or tempo
        if text_list[-1][0] == MEASURE_EVENTS_CHAR:
            mn_and_tempo_ids = multinote_ids.union(tempo_ids)
            logits[ATTR_NAME_INDEX['evt']][list(mn_and_tempo_ids)] = neg_inf

        # if the last token in text_list is position
        # then the next token's event can not be measure
        elif text_list[-1][0] == POSITION_EVENTS_CHAR:
            logits[ATTR_NAME_INDEX['evt']][list(position_ids)] = neg_inf

        # if the last token in text_list is note or multinote
        # the next token's event can not be tempo
        elif text_list[-1][0] in (NOTE_EVENTS_CHAR, MULTI_NOTE_EVENTS_CHAR):
            logits[ATTR_NAME_INDEX['evt']][list(tempo_ids)] = neg_inf

        # prohibits position tokens that is "behind" the current position
        k = len(text_list) - 1
        before_position_chars = (POSITION_EVENTS_CHAR, MEASURE_EVENTS_CHAR)
        while (text_list[k][0] not in before_position_chars) and (k > 0):
            k -= 1
        if text_list[k][0] == POSITION_EVENTS_CHAR:
            cur_position = b36strtoi(text_list[k][1:])
            for i in position_ids:
                if b36strtoi(vocabs.events.id2text[i][1:]) <= cur_position:
                    logits[ATTR_NAME_INDEX['evt']][i] = neg_inf

        # prohibit tempo tokens if there is already a tempo token
        k = len(text_list) - 1
        before_tempo_chars = (POSITION_EVENTS_CHAR, TEMPO_EVENTS_CHAR)
        while (text_list[k][0] not in before_tempo_chars) and (k > 0):
            k -= 1
        if text_list[k][0] == TEMPO_EVENTS_CHAR:
            logits[ATTR_NAME_INDEX['evt']][list(tempo_ids)] = neg_inf

        # only allow defined track number
        sep_position_in_text_list = text_list.index(SEP_TOKEN_STR)
        used_track_number_ids = {
            b36strtoi(t.split(':')[1]) + 1
            for t in text_list[1:sep_position_in_text_list]
            if t[0] == TRACK_EVENTS_CHAR
        }
        unused_track_number_ids = [
            index
            for index in range(vocabs.track_numbers.size)
            if index not in used_track_number_ids
        ]
        logits[ATTR_NAME_INDEX['trn']][unused_track_number_ids] = neg_inf

    # # re-normalized probs
    # normed_probs = [p / p.sum() for p in probs]
    # # if in some attribute all events are fill with zero, something is wrong
    # assert not any([torch.isnan(p).any() for p in normed_probs]), (
    #     f'Text List: {text_list}\n'
    #     f'Adjusted: {probs}\n'
    #     f'Re-normalized: {normed_probs}'
    # )
    return logits


SAMPLE_FUNCTION_ARGUMENT_CHOICES = ('none', 'top-k', 'top-p', 'nucleus')

def default_sampling(probs: torch.Tensor, _threshold) -> torch.Tensor:
    return torch.multinomial(probs, 1)

def top_k_sampling(probs: torch.Tensor, threshold: float) -> torch.Tensor:
    assert 0 < threshold <= 1.0
    k = max(1, int(probs.shape[0] * threshold))
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    topk_probs = sorted_probs[:k]
    sampled_topk = torch.multinomial(topk_probs, 1)
    sampled_index = sorted_indices[sampled_topk]
    return sampled_index

def nucleus_sampling(probs: torch.Tensor, threshold: float) -> torch.Tensor:
    assert 0 < threshold <= 1.0
    if threshold == 1.0:
        return torch.multinomial(probs, 1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus_number = torch.sum(cumsum_probs < threshold)
    if nucleus_number == 0:
        # returned dimension have to be 1
        return torch.unsqueeze(sorted_indices[0], dim=0)
    nucleus_probs = sorted_probs[:nucleus_number]
    sampled_nuclei = torch.multinomial(nucleus_probs, 1)
    sampled_index = sorted_indices[sampled_nuclei]
    return sampled_index


def check_next_token(
        text_list: List[str],
        next_token_str: str,
        array_memory: dict,
        vocabs: Vocabs,
        use_adjust_logit: bool,
        ignore_pending_note_error: bool,
        print_exception: bool) -> Union[Tuple[NDArray[np.uint16], dict], None]:
    """If `next_token` is ok to be put after `text_list`,
    return the new array and array memory. Otherwise, return None.
    """
    try:
        # format checking
        if not use_adjust_logit:
            # have to add EOS at the end to not raise error
            piece_to_midi(
                piece=' '.join(text_list + [next_token_str, END_TOKEN_STR]),
                tpq=vocabs.paras['tpq'],
                ignore_pending_note_error=ignore_pending_note_error
            )
        # compute full attribute array and second format checking
        next_token_array, new_array_memory = text_list_to_array(
            [next_token_str],
            vocabs=vocabs,
            input_memory=array_memory,
            output_memory=True
        )
        return next_token_array, new_array_memory
    except (ValueError, AssertionError) as e:
        if print_exception:
            print(repr(e))
        return None


@torch.no_grad()
def generate(
        model: MyMidiTransformer,
        max_generation_step: int,
        primer_seq: Union[torch.Tensor, None] = None,
        softmax_temperature: Union[List[float], None] = None,
        try_count_limit: int = 1000,
        use_adjust_logit: bool = True,
        sample_function: Union[str, None] = 'none',
        sample_threshold: Union[List[float], None] = None,
        print_exception: bool = False,
        show_tqdm: bool = False,
        ignore_pending_note_error: bool = True) -> List[str]:
    """
    Expect `primer_seq` to be Tensor with shape:
    `(1, seq_size, all_attr_number)` or `None`.

    If `primer_seq` is None, `text_list_to_array([BEGIN_TOKEN_STR])`
    will be used.

    `ignore_pending_note_error` is defaulted True, otherwise it is
    hard for model to generate continuing note.

    Return generated result as text list.
    """
    # check primer_seq
    if primer_seq is not None:
        exception_msg = (
            f'Expect primer_seq to have shape '
            f'(1, seq_length, all_attr_number). '
            f'Get {primer_seq.shape}'
        )
        assert len(primer_seq.shape) == 3, exception_msg
        assert (primer_seq.shape[0] == 1
            and primer_seq.shape[1] >= 1
            and primer_seq.shape[2] == len(ALL_ATTR_NAMES)
        ), exception_msg
        assert primer_seq.dtype == torch.int32
    else:
        primer_seq = torch.from_numpy(
            text_list_to_array([BEGIN_TOKEN_STR], model.vocabs).astype('int32')
            # was uint16 to save space torch want int32
        ).unsqueeze(0)

    primer_length = primer_seq.size(1)
    max_generation_step = min(
        model.max_seq_length, max_generation_step + primer_length
    ) - primer_length
    if max_generation_step <= 0:
        raise ValueError(
            f'primer_seq is longer than model\'s max_seq_length. '
            f'({primer_length} > {model.max_seq_length}) '
            f'No generation is performed.'
        )

    # check sampling method
    output_attrs_number = len(model.output_attrs_indices)
    if softmax_temperature is None:
        softmax_temperature = [1.0] * output_attrs_number
    elif isinstance(softmax_temperature, list):
        assert all(t >= 0 for t in softmax_temperature), \
            'Temperatures should not be less than zero'
        softmax_temperature = [
            t if t != 0 else 1e-6
            for t in softmax_temperature
        ]
        if len(softmax_temperature) == 1:
            softmax_temperature = softmax_temperature * output_attrs_number
        elif len(softmax_temperature) != output_attrs_number:
            raise ValueError(
                f'len(softmax_temperature) can only be 1 or out_attr_num. '
                f'Get {softmax_temperature}'
            )

    sample = None
    if sample_function is None:
        sample_threshold = 'none'
    assert sample_function in SAMPLE_FUNCTION_ARGUMENT_CHOICES, (
        f'Value of sample_function should be one of '
        f'{SAMPLE_FUNCTION_ARGUMENT_CHOICES}. '
        f'Get {sample_function}'
    )
    if sample_function == 'none':
        sample = default_sampling
    elif sample_function == 'top-k':
        sample = top_k_sampling
    elif sample_function == 'top-p' or sample_function == 'nucleus':
        sample = nucleus_sampling

    if sample_threshold is None:
        sample_threshold = [1.0] * output_attrs_number
    elif isinstance(sample_threshold, list):
        if len(sample_threshold) == 1:
            sample_threshold = sample_threshold * output_attrs_number
        elif len(sample_threshold) != output_attrs_number:
            raise ValueError(
                f'Length of sample_threshold can only be 1 or out_attr_num.'
                f'Get {sample_threshold}'
            )

    # get model device
    model_device = next(model.parameters()).device
    prev_training_state = model.training
    model.inference()

    # prepare famlity indices for probs adjustment
    if use_adjust_logit:
        # event_family_ids:
        #   multinote_ids, position_ids, measure_ids, track_ids, tempo_ids
        event_family_ids = (set(), set(), set(), set(), set())
        for t, i in model.vocabs.events.text2id.items():
            if t[0] == NOTE_EVENTS_CHAR or t[0] == MULTI_NOTE_EVENTS_CHAR:
                event_family_ids[0].add(i)
            elif t[0] == POSITION_EVENTS_CHAR:
                event_family_ids[1].add(i)
            elif t[0] == MEASURE_EVENTS_CHAR:
                event_family_ids[2].add(i)
            elif t[0] == TRACK_EVENTS_CHAR:
                event_family_ids[3].add(i)
            elif t[0] == TEMPO_EVENTS_CHAR:
                event_family_ids[4].add(i)

    text_list = array_to_text_list(
        array=primer_seq[0].cpu().numpy(),
        vocabs=model.vocabs
    )
    primer_length = primer_seq.shape[1]

    recurrent_memory = None
    _, array_memory = text_list_to_array(
        text_list,
        model.vocabs,
        input_memory=None,
        output_memory=True
    )

    # build memory for primer if use linear transformer
    if model.use_linear_attn:
        for i in range(primer_length):
            input_primer_seq = model.to_input_attrs(primer_seq[:, :i+1])
            input_primer_seq = input_primer_seq.to(model_device)
            _, recurrent_memory = model(input_primer_seq, recurrent_memory)

    generate_range_tqdm = tqdm(
        range(max_generation_step),
        disable=not show_tqdm,
        ncols=80,
        leave=None
    )
    for _ in generate_range_tqdm:
        input_primer_seq = model.to_input_attrs(primer_seq).to(model_device)
        if model.use_linear_attn:
            # return batched_last_logits because inferencing is True
            batched_last_logits, recurrent_memory = (
                model(input_primer_seq, recurrent_memory)
            )
            last_logits = [
                l[0].to('cpu')
                # to cpu, if was cuda (likely)
                for l in batched_last_logits
                # l has shape (1, attr_vocab_size)
            ]
        else:
            batched_logits = model(input_primer_seq)
            last_logits = [
                l[0, -1].to('cpu')
                for l in batched_logits
                # l has shape (1, seq_length, attr_vocab_size)
            ]

        if use_adjust_logit:
            # prevent many bad format in advance
            last_logits = adjust_logits(
                logits=last_logits,
                text_list=text_list,
                vocabs=model.vocabs,
                event_family_ids=event_family_ids
            )

        probs = [
            F.softmax(l / t, dim=0)
            for l, t in zip(last_logits, softmax_temperature)
            # l has shape (attr_vocab_size,)
        ]

        try_count = 0
        next_token_str = ""
        while try_count < try_count_limit:
            sampled_attrs = [
                sample(probs, threshold)
                for probs, threshold in zip(probs, sample_threshold)
            ]
            # next_token_array shape = (1, out_attr_num)
            next_token_array = torch.stack(sampled_attrs, dim=1).cpu().numpy()
            next_token_str = array_to_text_list(
                next_token_array,
                vocabs=model.vocabs
            )[0]
            if next_token_str == END_TOKEN_STR:
                text_list = text_list + [next_token_str]
                # if sampled EOS, then dont check. just end
                break

            check_result = check_next_token(
                text_list=text_list,
                next_token_str=next_token_str,
                array_memory=array_memory,
                vocabs=model.vocabs,
                use_adjust_logit=use_adjust_logit,
                ignore_pending_note_error=ignore_pending_note_error,
                print_exception=print_exception
            )
            if check_result is None:
                try_count += 1
                continue
            else:
                text_list = text_list + [next_token_str]
                next_array, array_memory = check_result
                primer_seq = torch.from_numpy(
                    next_array.astype('int32')
                ).unsqueeze(0)
                break
        # end while sample and check

        if try_count == try_count_limit:
            if print_exception:
                print('Exceeded try count limit:', try_count_limit)
            break

        if next_token_str == END_TOKEN_STR:
            break
    # end for each step

    if text_list[-1] != END_TOKEN_STR:
        text_list.append(END_TOKEN_STR)

    model.train(prev_training_state)
    return text_list
