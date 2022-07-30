import random
import re
from time import time

from miditoolkit.midi.parser import MidiFile
from miditoolkit.midi.containers import Note, Instrument, TimeSignature, TempoChange

from .tokens import *


def merge_drums(midi: MidiFile) -> None:
    """
        merge all drums into one instruments
    """
    merged_perc_notes = []
    for instrument in midi.instruments:
        if len(instrument.notes) > 0 and instrument.is_drum:
            merged_perc_notes.extend(instrument.notes)
    if len(merged_perc_notes) > 0:
        # remove duplicate
        merged_perc_notes = list(set(merged_perc_notes))
        merged_perc_notes.sort(key=lambda x: x.start)
        merged_perc_inst = Instrument(program=128, is_drum=True, name="merged_drums")
        merged_perc_inst.notes = merged_perc_notes
        new_instruments = [inst for inst in midi.instruments if not inst.is_drum] + [merged_perc_inst]
        midi.instruments = new_instruments


def merge_sparse_track(
        midi: MidiFile,
        sparse_threshold_ratio: float = 0.1,
        sparse_threshold_max: int = 50,
        discard_threshold_ratio: float = 0.01,
        discard_threshold_max: int = 5) -> None:
    good_insts_mapping = dict()
    bad_insts_mapping = dict()

    avg_note_num = sum([len(inst.notes) for inst in midi.instruments]) / len(midi.instruments)
    sparse_threshold = min(sparse_threshold_max, avg_note_num * sparse_threshold_ratio)
    discard_threshold = min(discard_threshold_max, avg_note_num * discard_threshold_ratio)

    for inst in midi.instruments:
        inst_attr = (inst.program, inst.is_drum)
        if len(inst.notes) < sparse_threshold:
            if inst_attr in bad_insts_mapping:
                bad_insts_mapping[inst_attr].append(inst)
            else:
                bad_insts_mapping[inst_attr] = [inst]
        else:
            if inst_attr in bad_insts_mapping:
                bad_insts_mapping[inst_attr].append(inst)
            else:
                bad_insts_mapping[inst_attr] = [inst]

    for bad_inst_attr, bad_inst_list in bad_insts_mapping.items():
        if bad_inst_attr in good_insts_mapping:
            for bad_inst in bad_insts_mapping:
                # find one track to merge
                rand_good_inst = random.choice(good_insts_mapping[bad_inst_attr])
                rand_good_inst.notes.extend(bad_inst.notes)
        # no track to merge
        else:
            for bad_inst in bad_inst_list:
                if len(bad_inst.notes) > discard_threshold:
                    if bad_inst_attr in good_insts_mapping:
                        good_insts_mapping[bad_inst_attr].append(bad_inst)
                    else:
                        good_insts_mapping[bad_inst_attr] = [bad_inst]

    midi.instruments = [inst for inst_list in good_insts_mapping.values() for inst in inst_list]


def merge_tracks(
        midi: MidiFile,
        max_track_number: int,
        use_merge_drums: bool,
        use_merge_sparse: bool) -> None:

    if use_merge_drums:
        merge_drums(midi)

    if use_merge_sparse:
        merge_sparse_track(midi)

    if len(midi.instruments) > max_track_number:
        insts = list(midi.instruments) # shallow copy
        # place drum track or the most note track at first
        insts.sort(key=lambda x: (not x.is_drum, -len(x.notes)))
        good_insts_mapping = dict()
        bad_insts_mapping = dict()

        for inst in insts[:max_track_number]:
            inst_attr = (inst.program, inst.is_drum)
            if inst_attr in good_insts_mapping:
                good_insts_mapping[inst_attr].append(inst)
            else:
                good_insts_mapping[inst_attr] = [inst]
        for inst in insts[max_track_number:]:
            inst_attr = (inst.program, inst.is_drum)
            if inst_attr in bad_insts_mapping:
                bad_insts_mapping[inst_attr].append(inst)
            else:
                bad_insts_mapping[inst_attr] = [inst]

        # print(midi_file_path)
        for bad_inst_attr, bad_inst_list in bad_insts_mapping.items():
            if bad_inst_attr in good_insts_mapping:
                for bad_inst in bad_inst_list:
                    # find one track to merge
                    rand_good_inst = random.choice(good_insts_mapping[bad_inst_attr])
                    rand_good_inst.notes.extend(bad_inst.notes)
            else:
                # discard all tracks in bad_insts_mapping
                pass

        good_insts_mapping = [inst for inst_list in good_insts_mapping.values() for inst in inst_list]
        assert len(good_insts_mapping) <= max_track_number, len(good_insts_mapping)
        midi.instruments = good_insts_mapping


def quatize_to_nth(time_in_tick: int, ticks_per_nth: float) -> int:
    q = round(time_in_tick / ticks_per_nth)
    return q if q != 0 else 1


def quantize_tempo(tempo, tempo_quantization):
    # snap
    t = round(tempo_quantization[0] + tempo_quantization[2] * round((tempo - tempo_quantization[0]) / tempo_quantization[2]))
    # clamp
    t = min(tempo_quantization[1], max(tempo_quantization[0], t))
    return t


def get_note_tokens(midi: MidiFile, nth: int, max_duration: int, velocity_step: int) -> list:
    """
        This return all note token in the midi as a list sorted in the ascending
        orders of onset time, track number, pitch, duration, velocity and instrument.
        The time unit used is tick
    """
    half_vel_step = velocity_step // 2
    nth_per_beat = nth // 4
    ticks_per_nth = midi.ticks_per_beat / nth_per_beat

    # because sometimes mido give value more than 32 bits to note.start and note.end
    # and those extra bits makes the while loop hangs forever
    for inst in midi.instruments:
        for note in inst.notes:
            if note.start > (1 << 32):
                note.start = note.start & 0xFFFFFFFF
            if note.end > (1 << 32):
                note.end = note.end & 0xFFFFFFFF
            assert note.end > note.start, "non-positive note duration"

    note_token_list = [
        NoteToken(
            onset=quatize_to_nth(note.start, ticks_per_nth),
            track_number=track_number,
            pitch=note.pitch,
            duration=quatize_to_nth(note.end-note.start, ticks_per_nth),
            velocity=(note.velocity//velocity_step)*velocity_step+half_vel_step
        )
        for track_number, inst in enumerate(midi.instruments)
        for note in inst.notes
    ]

    # handle too long duration
    # max_duration is in unit of quarter note (beat)
    max_duration_in_nth = max_duration * nth_per_beat
    # continuing means this note this going to connect to a note after max_duration
    # it is represented as negtive to seperate from notes that's really max_duration long
    continuing_duration = -max_duration_in_nth
    note_list_length = len(note_token_list)
    for i in range(note_list_length):
        note_token = note_token_list[i]
        if note_token.duration > max_duration_in_nth:
            cur_dur = note_token.duration
            cur_onset = note_token.onset
            while cur_dur > 0:
                if cur_dur > max_duration_in_nth:
                    note_token_list.append(
                        note_token._replace(
                            onset=cur_onset,
                            duration=continuing_duration
                        )
                    )
                    cur_onset += max_duration_in_nth
                    cur_dur -= max_duration_in_nth
                else:
                    note_token_list.append(
                        note_token._replace(
                            onset=cur_onset,
                            duration=cur_dur
                        )
                    )
                    cur_dur = 0
    note_token_list = [
        n for n in note_token_list
        if n.duration <= max_duration_in_nth
    ]
    note_token_list = list(set(note_token_list))
    note_token_list.sort()
    return note_token_list


def get_time_structure_tokens(
        midi: MidiFile,
        note_token_list: list,
        nth: int,
        tempo_quantization: list,
        tempo_method: str) -> list:
    """
        This return measure and tempo  The first measure have to contain the
        first note and the last note must end at the last measure
    """
    nth_per_beat = nth // 4
    ticks_per_nth = midi.ticks_per_beat / nth_per_beat
    time_signature_changes = midi.time_signature_changes
    # note_token_list is sorted
    first_note_onset = note_token_list[0].onset
    last_note_end = note_token_list[-1].onset + note_token_list[-1].duration
    # print('first, last:', first_note_onset, last_note_end)

    if len(time_signature_changes) == 0:
        time_sig_tuple_list = [(4, 4, 0)]
    else:
        # remove duplicated time_signature_changes
        time_sig_tuple_list = []
        for i, time_sig in enumerate(time_signature_changes):
            time_sig_tuple = (time_sig.numerator, time_sig.denominator, quatize_to_nth(time_sig.time, ticks_per_nth))
            assert time_sig.numerator != 0 and time_sig.denominator != 0, 'bad time signature'
            if i == 0:
                time_sig_tuple_list.append(time_sig_tuple)
            elif (time_sig_tuple[0] != time_signature_changes[i-1].numerator or \
                    time_sig_tuple[1] != time_signature_changes[i-1].denominator):
                time_sig_tuple_list.append(time_sig_tuple)
        if time_sig_tuple_list[0][2] > first_note_onset:
            time_sig_tuple_list = [(4, 4, 0)] + time_sig_tuple_list
    # print(time_sig_tuple_list)

    measure_token_list = []
    for i, (n, d, cur_timesig_start_time) in enumerate(time_sig_tuple_list):
        nth_per_measure = round(nth_per_beat * n * (4 / d))
        # find start
        # find end
        if i < len(time_sig_tuple_list) - 1:
            next_timesig_start_time = time_sig_tuple_list[i+1][2]
        else:
            # measure should end when no notes are ending any more
            time_to_last_note_end = last_note_end - cur_timesig_start_time
            num_of_measures = (time_to_last_note_end // nth_per_measure) + 1
            next_timesig_start_time = cur_timesig_start_time + num_of_measures * nth_per_measure
        # if note already end
        if cur_timesig_start_time > last_note_end:
            break
        # if note not started yet
        if next_timesig_start_time <= first_note_onset:
            continue

        # find the first measure that contain the first note
        while cur_timesig_start_time + nth_per_measure <= first_note_onset:
            cur_timesig_start_time += nth_per_measure
        # now cur_measure_start_time is the REAL first measure

        if not is_supported_time_signature(n, d):
            raise NotImplementedError(f'Encounter unsupported time signature {(n, d)}')
        # print(f'TimeSig {i}:', cur_timesig_start_time, next_timesig_start_time, nth_per_measure)

        for cur_measure_start_time in range(cur_timesig_start_time, next_timesig_start_time, nth_per_measure):
            measure_token_list.append(
                MeasureToken(
                    onset=cur_measure_start_time,
                    time_signature=(n, d)
                )
            )
        # end for cur_measure_start_time
    # end for time_sig_tuple

    tempo_token_list = []
    tempo_tuple_list = [] # (change_time, bpm)
    if len(midi.tempo_changes) == 0:
        tempo_tuple_list = [(0, 120)]
    else:
        # remove duplicated
        prev_bpm = None
        for i, tempo_change in enumerate(midi.tempo_changes):
            # remove the tempo token that didn't impacted the actual tempo
            if i < len(midi.tempo_changes) - 1:
                if midi.tempo_changes[i+1].time < measure_token_list[0].onset * ticks_per_nth:
                    continue
            if tempo_change.time >= last_note_end * ticks_per_nth:
                continue
            bpm = tempo_change.tempo
            if prev_bpm != bpm:
                change_time = quatize_to_nth(tempo_change.time, ticks_per_nth)
                tempo_tuple_list.append((change_time, int(bpm)))
            prev_bpm = bpm
    # print(tempo_tuple_list)

    if tempo_method.startswith('position'):
        tempo_token_list = []
        # use tempo token to record tempo info for get_position_token to handle
        for i, (change_time, bpm) in enumerate(tempo_tuple_list):
            tempo_token_list.append(
                TempoToken(
                    onset=max(measure_token_list[0].onset, change_time),
                    bpm=quantize_tempo(bpm, tempo_quantization)
                )
            )
        # add the dummy token as last element that for get_position_tokens
        tempo_token_list.append(
            TempoToken(
                onset=float('inf'),
                bpm=-1
            )
        )
    # elif tempo_method.startswith('measure')
    else:
        # add a onset=inf to handle edge condition
        tempo_tuple_list.append((float('inf'), -1))
        tempo_cursor = 0
        for i, cur_measure in enumerate(measure_token_list):

            cur_measure_onset = cur_measure.onset
            if i == 0:
                prev_measure_onset = float('-inf')
            else:
                prev_measure_onset = measure_token_list[i-1].onset
            if i == len(measure_token_list) - 1:
                n, d, = cur_measure.time_signature
                cur_measure_offset = round(nth_per_beat * n * (4 /d))
            else:
                cur_measure_offset = measure_token_list[i+1].onset

            # calculate the weighted tempo of current measure
            cur_measure_tempo_list = []
            cur_measure_tempo_length_list = []
            weighted_tempo = None
            # we know in last calculation the cursor stopped at the lastest position in the measure
            # this measure's tempo could only be different to previous one's tempo if:
            # - the cursor is stop at the middle of previous measure: it means the last calculated measure is previous
            #   measure and has at least two tempos in it
            # - the next tempo cusor is in current measure
            if (prev_measure_onset < tempo_tuple_list[tempo_cursor][0]
                or cur_measure_onset <= tempo_tuple_list[tempo_cursor+1][0] < cur_measure_offset):

                while True:
                    # look at the time between current and next tempo change
                    cur_measure_tempo_list.append(tempo_tuple_list[tempo_cursor][1])
                    cur_tempo_start = max(tempo_tuple_list[tempo_cursor][0], cur_measure_onset)
                    # because the last change is set at infinity so no need to worry index error
                    cur_tempo_end = min(tempo_tuple_list[tempo_cursor+1][0], cur_measure_offset)
                    cur_measure_tempo_length_list.append(cur_tempo_end - cur_tempo_start)
                    if tempo_tuple_list[tempo_cursor+1][0] > cur_measure_offset:
                        # if the next tempo_change_times is not in the middle of current measure or
                        # not at the beginning of next measure, stop increasing cursor
                        # because next measure will needs the value of current tempo to calculate
                        # tempo_cursor+1 wont cause index error because there is a onset=inf token at the end
                        break
                    tempo_cursor += 1

                if len(cur_measure_tempo_list) >= 1:
                    weighted_tempo = sum(
                        t * w / (cur_measure_offset - cur_measure_onset)
                        for t, w in zip(cur_measure_tempo_list, cur_measure_tempo_length_list)
                    )
                    weighted_tempo = quantize_tempo(weighted_tempo, tempo_quantization)
                    # print(weighted_tempo)

            if tempo_method == 'measure_event':
                if weighted_tempo is None:
                    weighted_tempo = tempo_token_list[-1].bpm
                tempo_token_list.append(
                    TempoToken(
                        onset=cur_measure_onset,
                        bpm=weighted_tempo
                    )
                )
            # elif tempo_method == 'measure_attribute':
            else:
                if weighted_tempo is None:
                    weighted_tempo = measure_token_list[i-1].bpm
                measure_token_list[i] = cur_measure._replace(bpm=weighted_tempo)

    # print('\n'.join(map(str, measure_token_list))
    # print('\n'.join(map(str, tempo_token_list))
    # measure_token_list.sort() # dont have to sort
    return  measure_token_list, tempo_token_list


def get_position_tokens(note_measure_tokens_list: list, tempo_token_list: list, tempo_method: str) -> list:
    position_token_list = []
    cur_measure_onset_time = 0
    cur_note_onset_time = 0

    tempo_cursor = 0
    cur_tempo = tempo_token_list[0].bpm if tempo_method == 'position_attribute' else None

    for token in note_measure_tokens_list:
        if token.onset > cur_note_onset_time:
            if token.type_priority == TYPE_PRIORITY['MeasureToken']:
                cur_measure_onset_time = token.onset
            elif token.type_priority == TYPE_PRIORITY['NoteToken']:
                if tempo_method == 'position_attribute':
                    # plus one there wont cause index error because there is a onset=inf token at the end
                    if tempo_token_list[tempo_cursor+1].onset <= token.onset:
                        tempo_cursor += 1
                        cur_tempo = tempo_token_list[tempo_cursor].bpm
                    position_token_list.append(
                        PositionToken(
                            onset=token.onset,
                            position=token.onset-cur_measure_onset_time,
                            bpm=cur_tempo
                        )
                    )
                else:
                    position_token_list.append(
                        PositionToken(
                            onset=token.onset,
                            position=token.onset-cur_measure_onset_time
                        )
                    )
                cur_note_onset_time = token.onset
    # position_token_list.sort()
    return position_token_list


def get_head_tokens(midi: MidiFile) -> list:
    track_token_list = [
        TrackToken(
            track_number=i,
            instrument=inst.program
        )
        for i, inst in enumerate(midi.instruments)
    ]
    return [BeginOfScoreToken()] + track_token_list


def midi_to_token_list(
        midi: MidiFile,
        nth: int,
        max_duration: int,
        velocity_step: int,
        tempo_quantization: list,
        tempo_method: str) -> list:

    note_token_list = get_note_tokens(midi, nth, max_duration, velocity_step)

    # tempo_token_list would be empty list if tempo_method is 'measure_attribute'
    measure_token_list, tempo_token_list = get_time_structure_tokens(midi, note_token_list, nth, tempo_quantization, tempo_method)
    assert len(tempo_token_list) > 0 or tempo_method == 'measure_attribute', 'no tempo information retrieved'
    assert len(note_token_list) * 2 > len(measure_token_list) * len(midi.instruments), 'music too sparse'
    note_measure_tokens_list = note_token_list + measure_token_list
    note_measure_tokens_list.sort()
    assert note_measure_tokens_list[0].type_priority == 2, 'first element of body is not measure token'

    # align the first time to zero
    first_onset_time = note_measure_tokens_list[0].onset
    if first_onset_time != 0:
        note_measure_tokens_list = [
            token._replace(onset=token.onset-first_onset_time)
            for token in note_measure_tokens_list
        ]
        if len(tempo_token_list) > 0:
            tempo_token_list = [
                token._replace(onset=token.onset-first_onset_time)
                for token in tempo_token_list
            ]

    pos_token_list = get_position_tokens(note_measure_tokens_list, tempo_token_list, tempo_method)
    if len(tempo_token_list) > 0 and tempo_method.startswith('position'):
        tempo_token_list.pop() # take away the onset=inf token at the end
    if tempo_method.endswith('attribute'):
        body_token_list = pos_token_list + note_measure_tokens_list
    else:
        body_token_list = pos_token_list + note_measure_tokens_list + tempo_token_list
    body_token_list.sort()

    head_tokens_list = get_head_tokens(midi)
    full_token_list = head_tokens_list + body_token_list + [EndOfScoreToken()]

    return full_token_list

def midi_to_text_list(
        midi_filepath: str,
        nth: int,
        max_track_number: int,
        max_duration: int,
        velocity_step: int,
        tempo_quantization: list, # [tempo_min, tempo_max, tempo_step]
        tempo_method: str,
        use_merge_drums: bool = True,
        use_merge_sparse: bool = True,
        debug: bool = False) -> list:
    """
        Parameters:
        - midi_filepath: midi file path
        - nth: to quantize notes to nth (96, 64 or 32)
        - max_track_number: the maximum tracks nubmer to keep in text, if the input midi has more
        'instruments' than this value, some tracks would be merged or discard
        - max_duration: max length of duration in unit of quarter note (beat)
        - tempo_quantization: (min, max, step), where min and max are INCLUSIVE
        - tempo_method: can be 'measure_attribute', 'position_attribute', 'measure_event', 'position_event'
          - 'attribute' means tempo info is part of the measure or position event token
          - 'event' means tempo will be its own event token. it will be in the sequence where tempo change occurs.
            the token will be placed after the measure token and before position token
        - use_merge_drums: to merge drums tracks or not
        - use_merge_sparse: to merge sparse tracks or not
    """

    assert nth > 4 and nth % 4 == 0, 'nth is not multiple of 4'
    assert (nth // 4) * max_duration <= 128, f'max_duration in tick is greater than 128: {nth * max_duration}'
    assert max_track_number <= 255, f'max_track_number is greater than 255'
    assert tempo_quantization[0] <= tempo_quantization[1] and tempo_quantization[2] > 0, 'bad tempo_quantization'
    assert tempo_method in ['measure_attribute', 'position_attribute', 'measure_event', 'position_event'], \
        'bad tempo_method name'

    start_time = time()

    midi = MidiFile(midi_filepath)
    assert midi.ticks_per_beat >= (nth//2), \
        f'midi.ticks_per_beat ({midi.ticks_per_beat}) is less than half of nth ({nth})'
    if debug:
        print('tick/beat:', midi.ticks_per_beat)

    for ins in midi.instruments:
        ins.remove_invalid_notes(verbose=False)

    merge_tracks(midi, max_track_number, use_merge_drums, use_merge_sparse)
    assert len(midi.instruments) > 0

    for i, inst in enumerate(midi.instruments):
        if inst.is_drum:
            midi.instruments[i].program = 128

    if debug:
        print('Track:', len(midi.instruments))

    token_list = midi_to_token_list(midi, nth, max_duration, velocity_step, tempo_quantization, tempo_method)

    if debug:
        print('Token list length:', len(token_list))
        print('Time:', time()-start_time)
        print('--------')

    text_list = list(map(token_to_text, token_list))
    # text = ' '.join(map(str, text_list))
    return text_list


def handle_note_continuation(
        is_cont: bool,
        note_attrs: tuple,
        ticks_per_nth: int,
        cur_time_in_ticks: int,
        pending_cont_notes: dict):
    # pending_cont_notes is dict object so it is pass by reference
    pitch, duration, velocity, track_number = note_attrs
    # check if there is a note to continue at this time position
    if cur_time_in_ticks in pending_cont_notes:
        info = (pitch, velocity, track_number)
        if info in pending_cont_notes[cur_time_in_ticks]:
            if len(pending_cont_notes[cur_time_in_ticks][info]) > 0:
                onset = pending_cont_notes[cur_time_in_ticks][info].pop()
            # print(cur_time_in_ticks, 'onset pop', pending_cont_notes)
            if len(pending_cont_notes[cur_time_in_ticks][info]) == 0:
                pending_cont_notes.pop(cur_time_in_ticks)
        else:
            onset = cur_time_in_ticks
    else:
        onset = cur_time_in_ticks

    if is_cont:
        # this note is going to connect to a note after max_duration
        info = (pitch, velocity, track_number)
        pending_cont_time = cur_time_in_ticks + duration * ticks_per_nth
        if pending_cont_time not in pending_cont_notes:
            pending_cont_notes[pending_cont_time] = dict()
        if info in pending_cont_notes[pending_cont_time]:
            pending_cont_notes[pending_cont_time][info].append(onset)
        else:
            pending_cont_notes[pending_cont_time] = {info : [onset]}
        # print(cur_time_in_ticks, 'onset append', pending_cont_notes)
    else:
        return Note(velocity, pitch, onset, cur_time_in_ticks + duration * ticks_per_nth)
    return None


def piece_to_midi(piece: str, nth: int) -> MidiFile:
    midi = MidiFile(ticks_per_beat=nth)
    ticks_per_nth = 4 # convenient huh

    text_list = piece.split(' ')
    assert text_list[0] == 'BOS' and text_list[-1] == 'EOS'
    assert text_list[1][0] == 'R', 'No track token in head'

    is_head = True
    cur_time_in_ticks = 0
    cur_measure_length_in_ticks = -1
    cur_measure_onset_in_ticks = -1
    cur_time_signature = None
    cur_tempo = None
    pending_cont_notes = dict()
    for text in text_list[1:-1]:
        typename = text[0]
        if is_head:
            if typename == 'R':
                track_number, instrument = (int(x, TOKEN_INT2STR_BASE) for x in text[1:].split(':'))
                assert len(midi.instruments) == track_number
                midi.instruments.append(
                    Instrument(program=instrument%128, is_drum=instrument==128, name=f'Track_{track_number}')
                )
            else:
                is_head = False
        if not is_head:
            if typename == 'M':
                numer, denom, *tempo_change = (int(x, TOKEN_INT2STR_BASE) for x in re.split(r'[:/+]', text[1:]))
                if cur_measure_onset_in_ticks == -1: # first measure
                    cur_measure_onset_in_ticks = 0
                    cur_time_in_ticks = 0
                else:
                    assert cur_measure_length_in_ticks > 0
                    cur_measure_onset_in_ticks += cur_measure_length_in_ticks
                    cur_time_in_ticks = cur_measure_onset_in_ticks

                cur_measure_length_in_ticks = round(4 * numer * (midi.ticks_per_beat / denom))
                if cur_time_signature != (numer, denom):
                    cur_time_signature = (numer, denom)
                    midi.time_signature_changes.append(TimeSignature(numer, denom, cur_time_in_ticks))

                if len(tempo_change):
                    if cur_tempo != tempo_change[0]:
                        midi.tempo_changes.append(TempoChange(tempo_change[0], cur_time_in_ticks))
                # print("M", attr, cur_measure_onset_in_ticks)

            elif typename == 'T':
                midi.tempo_changes.append(TempoChange(int(text[1:], TOKEN_INT2STR_BASE), cur_time_in_ticks))
                # print('T', attr, cur_time_in_ticks)

            elif typename == 'P':
                time_position, *tempo_change = (int(x, TOKEN_INT2STR_BASE) for x in re.split(r'[:+]', text[1:]))
                cur_time_in_ticks = time_position * ticks_per_nth + cur_measure_onset_in_ticks
                if len(tempo_change):
                    if cur_tempo != tempo_change[0]:
                        midi.tempo_changes.append(TempoChange(tempo_change[0], cur_time_in_ticks))
                # print('P', attr)

            elif typename == 'N':
                is_cont = text[1] == '~'
                if is_cont:
                    note_attr = tuple(int(x, TOKEN_INT2STR_BASE) for x in text[3:].split(':'))
                else:
                    note_attr = tuple(int(x, TOKEN_INT2STR_BASE) for x in text[2:].split(':'))
                n = handle_note_continuation(is_cont, note_attr, ticks_per_nth, cur_time_in_ticks, pending_cont_notes)
                if n is not None:
                    midi.instruments[note_attr[3]].notes.append(n)

            elif typename == 'S':
                shape_string, *pdvt = text[1:].split(':')
                relnote_list = []
                for s in shape_string.split(';'):
                    is_cont = s[-1] == '~'
                    if is_cont:
                        relnote = [int(a, TOKEN_INT2STR_BASE) for a in s[:-1].split(',')]
                    else:
                        relnote = [int(a, TOKEN_INT2STR_BASE) for a in s.split(',')]
                    relnote = [True] + relnote
                    relnote_list.append(relnote)

                base_pitch, time_unit, velocity, track_num = (int(x, TOKEN_INT2STR_BASE) for x in pdvt)
                for is_cont, rel_onset, rel_pitch, rel_dur in relnote_list:
                    note_attr = (base_pitch+rel_pitch, time_unit*rel_dur, velocity, track_num)
                    n = handle_note_continuation(is_cont, note_attr, ticks_per_nth, cur_time_in_ticks+rel_onset*time_unit, pending_cont_notes)
                    if n is not None:
                        midi.instruments[track_num].notes.append(n)
            else:
                raise TokenParseError(f'bad token string: {text}')
    # end loop text_list
    return midi


def make_para_yaml(args: dict) -> str:
    lines = [
        f'{k}: {v}\n'
        for k, v in args.items()
        if v is not None
    ]
    # print(''.join(lines) +  '---\n')
    return ''.join(lines) +  '---\n'

def parse_para_yaml(yaml: str) -> dict:
    pairs = yaml.split('\n')
    args_dict = dict()
    for p in pairs:
        q = p.split(': ')
        if len(q) <= 1:
            continue
        k, v = q
        # tempo_quatization is list
        if k == 'tempo_quantization':
            args_dict[k] = list(map(int, v[1:-1].split(', ')))
        elif k == 'tempo_method':
            args_dict[k] = v
        else:
            args_dict[k] = int(v)
    return args_dict

# use iterator to handle large file
class PieceIterator:
    def __init__(self, corpus_file) -> None:
        self.file = corpus_file
        self.length = None

    def __len__(self) -> int:
        if self.length is None:
            self.file.seek(0)
            self.length = sum(
                1 if line.startswith('BOS') else 0
                for line in self.file
            )
        return self.length

    def __iter__(self):
        self.file.seek(0)
        body_start = False
        for line in self.file:
            if line == '---\n':
                body_start = True
                continue
            if body_start:
                yield line[:-1] # remove \n at the end

def file_to_paras_and_piece_iterator(corpus_file):
    yaml_string = ''
    corpus_file.seek(0)
    for line in corpus_file:
        if line == '---\n':
            break
        yaml_string += line
    paras = parse_para_yaml(yaml_string)
    return paras, PieceIterator(corpus_file)
