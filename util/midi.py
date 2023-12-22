from decimal import Decimal, ROUND_HALF_UP
from typing import Tuple, List, Dict, Union
# from time import time

from miditoolkit import MidiFile, Instrument, TempoChange, TimeSignature, Note

from . import tokens
from .tokens import (
    BeginOfScoreToken, SectionSeperatorToken, EndOfScoreToken,
    NoteToken, TempoToken, PositionToken, MeasureToken, TrackToken,
    TYPE_PRIORITY,
    get_supported_time_signatures, token_to_str, b36strtoi,
)


# def merge_drums(midi: MidiFile) -> None:
#     # merge all drum tracks into one
#     merged_perc_notes = []
#     for track in midi.instruments:
#         if len(track.notes) > 0 and track.is_drum:
#             merged_perc_notes.extend(track.notes)
#     if len(merged_perc_notes) > 0:
#         # remove duplicate
#         merged_perc_notes = list(set(merged_perc_notes))
#         merged_perc_notes.sort(key=lambda x: (x.start, x.end))
#         merged_perc_inst = Instrument(
#             program=128, is_drum=True, name='merged_drums'
#         )
#         merged_perc_inst.notes = merged_perc_notes
#         not_perc_inst_list = [
#             track
#             for track in midi.instruments
#             if not track.is_drum
#         ]
#         midi.instruments = not_perc_inst_list + [merged_perc_inst]


def merge_tracks(midi: MidiFile, max_track_number: int) -> None:

    # if use_merge_drums:
    #     merge_drums(midi)

    # remove tracks with no notes
    midi.instruments = [
        track
        for track in midi.instruments
        if len(track.notes) > 0
    ]

    if len(midi.instruments) > max_track_number:
        tracks = list(midi.instruments) # shallow copy
        # sort tracks with decreasing note number
        tracks.sort(key=lambda x: -len(x.notes))
        good_tracks_mapping: Dict[int, List[Instrument]] = dict()
        bad_tracks_mapping: Dict[int, List[Instrument]] = dict()

        for track in tracks[:max_track_number]:
            track_program = 128 if track.is_drum else track.program
            if track_program in good_tracks_mapping:
                good_tracks_mapping[track_program].append(track)
            else:
                good_tracks_mapping[track_program] = [track]
        for track in tracks[max_track_number:]:
            track_program = 128 if track.is_drum else track.program
            if track_program in bad_tracks_mapping:
                bad_tracks_mapping[track_program].append(track)
            else:
                bad_tracks_mapping[track_program] = [track]

        # print(midi_file_path)
        for bad_track_program, bad_track_list in bad_tracks_mapping.items():
            if bad_track_program in good_tracks_mapping:
                for bad_track in bad_track_list:
                    # merge with track with fewest note
                    last_good_track = good_tracks_mapping[bad_track_program][-1]
                    last_good_track.notes.extend(bad_track.notes)
                    # re-sort the good track
                    good_tracks_mapping[bad_track_program].sort(
                        key=lambda x: -len(x.notes)
                    )
            elif len(bad_track_list) > 1:
                # merge all into one
                for other_bad_track in bad_track_list[1:]:
                    bad_track_list[0].notes.extend(other_bad_track.notes)
            else:
                # cannot do anything
                pass

        merged_tracks = [
            track
            for track_list in good_tracks_mapping.values()
            for track in track_list
        ]
        merged_tracks += [
            track
            for track_list in bad_tracks_mapping.values()
            for track in track_list
        ]

        assert len(merged_tracks) <= max_track_number, \
            'Track number exceed limit after merge'
            # f'Track number exceed limit after merge: {len(merged_tracks)}'

        midi.instruments = merged_tracks


def quantize_tick(tick: int, source_tqb: int, target_tpb: int) -> int:
    return int(
        (Decimal(tick) / source_tqb * target_tpb).quantize(0, ROUND_HALF_UP)
    )


def change_tpq(midi: MidiFile, new_tpq: int):
    # midi = copy.deepcopy(midi)
    source_tqb = midi.ticks_per_beat
    for track in midi.instruments:
        for n in track.notes:
            n.start = quantize_tick(n.start, source_tqb, new_tpq)
            n.end = quantize_tick(n.end, source_tqb, new_tpq)
            if n.end == n.start: # to prevent note perishment
                n.end = n.start + 1
    for tempo in midi.tempo_changes:
        tempo.time = quantize_tick(tempo.time, source_tqb, new_tpq)
    for time_sig in midi.time_signature_changes:
        time_sig.time = quantize_tick(time_sig.time, source_tqb, new_tpq)
    # return midi


def quantize_tempo(
        tempo: Union[float, int],
        tempo_quantization: Tuple[int, int, int]) -> int:
    begin, end, step = tempo_quantization
    tempo = Decimal(tempo)
    dec_step = Decimal(step)
    # snap
    t = int((tempo / dec_step).quantize(0, rounding=ROUND_HALF_UP)) * step
    # clamp
    t = max(begin, min(end, t))
    return t


def quantize_velocity(velocty: int, step: int) -> int:
    # snap
    dec_step = Decimal(step)
    v = int((velocty / dec_step).quantize(0, rounding=ROUND_HALF_UP)) * step
    # clamp
    min_v = step
    max_v = (127 // step) * step # use // to round down
    v = max(min_v, min(max_v, v))
    return v


# when tpq is 12, in tempo of bpm 120, 2^18 - 1 ticks are about 182 minutes
# if a note have start or end after this, we think the file is likely corrupted
NOTE_TIME_CORRUPTION_LIMIT = 0x3FFFF# 18 bits

# if a note have duration > 11 minutes, we think the file is likely corrupted
NOTE_DURATION_CORRUPTION_LIMIT = 0x3FFF # 14 bits

# we ignore any event happening after this (about 91 minutes)
EVENT_TIME_LIMIT = 0x1FFFF # 17 bits

def get_note_tokens(
        midi: MidiFile,
        max_duration: int,
        velocity_step: int,
        use_cont_note: bool) -> List[NoteToken]:
    """
    Return all note token in the midi as a list sorted in the
    ascending orders of onset time, track number, pitch, duration,
    velocity and instrument.
    """
    for track in midi.instruments:
        for i, note in enumerate(track.notes):
            assert (note.start <= NOTE_TIME_CORRUPTION_LIMIT
                    or note.end <= NOTE_TIME_CORRUPTION_LIMIT), \
                'Note event time too large, likely corrupted'
            assert note.end - note.start <= NOTE_DURATION_CORRUPTION_LIMIT, \
                'Note duration too large, likely corrupted'

    note_token_list = [
        NoteToken(
            onset=note.start,
            track_number=track_number,
            pitch=note.pitch,
            duration=note.end-note.start,
            velocity=quantize_velocity(note.velocity, velocity_step)
        )
        for track_number, track in enumerate(midi.instruments)
        for note in track.notes
        if (note.start < note.end and note.start >= 0
            and note.start <= EVENT_TIME_LIMIT
            and note.end <= EVENT_TIME_LIMIT
            and (not track.is_drum or 35 <= note.pitch <= 81))
            # remove negtives and percussion notes that
            # is not in the Genral MIDI percussion map
    ]

    # handle too long duration
    # continuing is represented by negative duration
    continuing_duration = -max_duration
    note_list_length = len(note_token_list)
    for i in range(note_list_length):
        note_token = note_token_list[i]
        if note_token.duration > max_duration:
            if not use_cont_note:
                note_token.duration = max_duration
                continue
            cur_dur = note_token.duration
            cur_onset = note_token.onset
            while cur_dur > 0:
                if cur_dur > max_duration:
                    note_token_list.append(
                        note_token._replace(
                            onset=cur_onset,
                            duration=continuing_duration
                        )
                    )
                    cur_onset += max_duration
                    cur_dur -= max_duration
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
        if n.duration <= max_duration
    ]
    note_token_list = list(set(note_token_list))
    note_token_list.sort()

    return note_token_list


def get_time_structure_tokens(
        midi: MidiFile,
        note_token_list: list,
        tpq: int,
        tempo_quantization: Tuple[int, int, int]
    ) -> Tuple[List[MeasureToken], List[TempoToken]]:
    """
    Return measure list and tempo list. The first measure contains the
    first note and the last note ends in the last measure.
    """
    # note_token_list is sorted
    first_note_start = note_token_list[0].onset
    last_note_end = note_token_list[-1].onset + note_token_list[-1].duration
    # print('first, last:', first_note_start, last_note_end)

    supported_time_signatures = get_supported_time_signatures()

    if len(midi.time_signature_changes) == 0:
        # time_sig_list = [TimeSignature(4, 4, 0)] # default 4/4
        assert False, 'No time signature information retrieved'
    else:
        time_sig_list = []
        # remove duplicated time_signatures
        prev_nd = (None, None)
        for i, time_sig in enumerate(midi.time_signature_changes):
            if time_sig.time > last_note_end:
                break
            if prev_nd != (time_sig.numerator, time_sig.denominator):
                time_sig_list.append(time_sig)
            prev_nd = (time_sig.numerator, time_sig.denominator)

        # some midi file has their first time signature begin at time 1
        # but the rest of the changing times are set as if the first
        # time signature started at time 0. weird.
        if len(time_sig_list) > 0:
            if time_sig_list[0].time == 1:
                if len(time_sig_list) > 1:
                    ticks_per_measure = (
                        4 * tpq * time_sig_list[0].numerator
                        // time_sig_list[0].denominator
                    )
                    if time_sig_list[1].time % ticks_per_measure == 0:
                        time_sig_list[0].time = 0
                else:
                    time_sig_list[0].time = 0
    assert len(time_sig_list) > 0, 'No time signature information retrieved'
    # print(time_sig_list)

    # assert time_sig_list[0].time <= first_note_start, \
    #     'Time signature undefined before first note'

    # if the time signature starts after the first note, try 4/4 start at time 0
    if time_sig_list[0].time > first_note_start:
        time_sig_list = [(TimeSignature(4, 4, 0))] + time_sig_list

    measure_token_list = []
    for i, time_sig in enumerate(time_sig_list):
        time_sig_tuple = (time_sig.numerator, time_sig.denominator)
        assert time_sig_tuple in supported_time_signatures, \
            'Unsupported time signature'
            # f'Unsupported time signature {time_sig_tuple}'

        ticks_per_measure = 4 * tpq * time_sig.numerator // time_sig.denominator
        cur_timesig_start = time_sig.time

        # if note already end
        if cur_timesig_start > last_note_end:
            break

        # find end
        if i < len(time_sig_list) - 1:
            next_timesig_start = time_sig_list[i+1].time
        else:
            # measure should end when no notes are ending any more
            time_to_last_note_end = last_note_end - cur_timesig_start
            measure_number = (time_to_last_note_end // ticks_per_measure) + 1
            next_timesig_start = (
                cur_timesig_start + measure_number * ticks_per_measure
            )

        # if note not started yet
        if next_timesig_start <= first_note_start:
            continue

        # check if the difference of starting time between current and next
        # time signature are multiple of current measure's length
        assert (
            (next_timesig_start - cur_timesig_start) % ticks_per_measure == 0
        ), 'Bad starting time of a time signature'

        # find the first measure that at least contain the first note
        while cur_timesig_start + ticks_per_measure <= first_note_start:
            cur_timesig_start += ticks_per_measure
        # now cur_measure_start is the REAL first measure

        for cur_measure_start in range(
            cur_timesig_start, next_timesig_start, ticks_per_measure):
            if cur_measure_start > last_note_end:
                break
            measure_token_list.append(
                MeasureToken(
                    onset=cur_measure_start,
                    time_signature=(time_sig.numerator, time_sig.denominator)
                )
            )
        # end for cur_measure_start
    # end for time_sig_tuple

    # check if last measure contain last note
    last_measure = measure_token_list[-1]
    last_measure_start = last_measure.onset
    last_measure_length = (
        4 * tpq * last_measure.time_signature[0]
        // last_measure.time_signature[1]
    )
    last_measure_end = last_measure_start + last_measure_length
    assert last_measure_start <= last_note_end < last_measure_end, \
        "Last note did not ends in last measure"

    tempo_token_list = []
    tempo_list = [] # TempoChange(tempo, time)
    if len(midi.tempo_changes) == 0:
        tempo_list = [TempoChange(120, measure_token_list[0].onset)]
    else:
        # remove duplicated time and tempos
        prev_time = None
        prev_tempo = None
        for i, tempo in enumerate(midi.tempo_changes):
            if tempo.time > last_note_end:
                break
            # if tempo change starts before first measure
            # change time to the start time of first measure
            if tempo.time < measure_token_list[0].onset:
                cur_time = measure_token_list[0].onset
            else:
                cur_time = tempo.time
            cur_tempo = quantize_tempo(tempo.tempo, tempo_quantization)
            if cur_tempo != prev_tempo:
                # if previous tempo change is on the same time as current one
                # remove the previous one
                if prev_time == cur_time:
                    tempo_list.pop()
                tempo_list.append(TempoChange(cur_tempo, cur_time))
            prev_tempo = cur_tempo
            prev_time = cur_time
    assert len(tempo_list) > 0, 'No tempo information retrieved'
    # sometime it is not sorted, dunno why
    tempo_list.sort(key=lambda t: t.time)

    # same weird issue as time signature
    # but tempo can appear in any place so this is all we can do
    if tempo_list[0].time == 1 and measure_token_list[0].onset == 0:
        tempo_list[0].time = 0
    # assert tempo_list[0].time == measure_token_list[0].onset, \
    #     'Tempo not start at time 0'

    # if the tempo change starts after the first measure,
    # use the default 120 at first measure
    if tempo_list[0].time > measure_token_list[0].onset:
        tempo_list = [
            TempoChange(120, measure_token_list[0].onset)
        ] + tempo_list

    tempo_token_list = [
        TempoToken(
            onset=t.time,
            bpm=t.tempo
        )
        for t in tempo_list
    ]

    # print('\n'.join(map(str, measure_token_list)))
    # print('\n'.join(map(str, tempo_token_list)))
    # measure_token_list.sort() # dont have to sort
    tempo_token_list.sort() # just to be sure
    return  measure_token_list, tempo_token_list


def get_position_tokens(
        note_token_list: List[NoteToken],
        measure_token_list: List[MeasureToken],
        tempo_token_list: List[TempoToken]) -> List[PositionToken]:
    nmt_token_list = measure_token_list + note_token_list + tempo_token_list
    nmt_token_list.sort()

    position_token_list = []

    cur_measure_onset = 0
    last_added_position_onset = 0

    for token in nmt_token_list:

        if token.type_priority == TYPE_PRIORITY['MeasureToken']:
            cur_measure_onset = token.onset

        elif token.type_priority == TYPE_PRIORITY['TempoToken']:
            position_token_list.append(
                PositionToken(
                    onset=token.onset,
                    position=token.onset-cur_measure_onset
                )
            )
            last_added_position_onset = token.onset
        elif token.type_priority == TYPE_PRIORITY['NoteToken']:
            if token.onset > last_added_position_onset:
                position_token_list.append(
                    PositionToken(
                        onset=token.onset,
                        position=token.onset-cur_measure_onset
                    )
                )
                last_added_position_onset = token.onset
    return position_token_list


def get_head_tokens(midi: MidiFile) -> List[TrackToken]:
    track_token_list = [
        TrackToken(
            track_number=i,
            instrument=track.program
        )
        for i, track in enumerate(midi.instruments)
    ]
    return track_token_list


def detect_long_empty_measures(token_list: list) -> None:
    consecutive_measure_token_count = 0
    for token in token_list:
        if token.type_priority == TYPE_PRIORITY['MeasureToken']:
            consecutive_measure_token_count += 1
        elif token.type_priority == TYPE_PRIORITY['NoteToken']:
            consecutive_measure_token_count = 0
        assert consecutive_measure_token_count < 20, \
            'Very long empty measures detected, likely corrupted'


def midi_to_token_list(
        midi: MidiFile,
        tpq: int,
        max_duration: int,
        velocity_step: int,
        use_cont_note: bool,
        tempo_quantization: Tuple[int, int, int]) -> List[Tuple]:
    note_token_list = get_note_tokens(
        midi, max_duration, velocity_step, use_cont_note
    )
    assert len(note_token_list) > 0, 'No notes in midi'

    measure_token_list, tempo_token_list = get_time_structure_tokens(
        midi, note_token_list, tpq, tempo_quantization
    )
    assert measure_token_list[0].onset <= note_token_list[0].onset, \
        'First measure is after first note'

    pos_token_list = get_position_tokens(
        note_token_list, measure_token_list, tempo_token_list
    )
    body_token_list = (
        pos_token_list
        + note_token_list
        + measure_token_list
        + tempo_token_list
    )
    body_token_list.sort()

    head_token_list = get_head_tokens(midi)
    full_token_list = (
        [BeginOfScoreToken()]
        + head_token_list + [SectionSeperatorToken()]
        + body_token_list + [EndOfScoreToken()]
    )

    detect_long_empty_measures(full_token_list)
    return full_token_list


def midi_to_piece(
        midi: MidiFile,
        tpq: int,
        max_track_number: int,
        max_duration: int,
        velocity_step: int,
        use_cont_note: bool,
        tempo_quantization: Tuple[int, int, int]) -> str:
    """
    Parameters:
    - `midi`: A miditoolkit MidiFile object
    - `tpq`: Quantize onset and duration to multiples of the length of
      `1/tpq` of quarter note. Have to be even number.
    - `max_track_number`: The maximum tracks nubmer to keep in text,
      if the input midi has more 'instruments' than this value, some
      tracks would be merged or discarded.
    - `max_duration`: The max length of duration in tick.
    - `velocity_step`: Quantize velocity to multiples of velocity_step.
    - `use_cont_note`: Use contiuing notes or not. If not, cut short
      the over-length notes.
    - `tempo_quantization`: Three values are (min, max, step). where
      min and max are INCLUSIVE.
    """

    assert isinstance(tpq, int) and tpq > 2 and tpq % 2 == 0, \
        'tpq is not even number'
    assert max_track_number <= 255, \
        f'max_track_number is greater than 255: {max_track_number}'
    assert isinstance(velocity_step, int), 'Bad velocity_step'
    assert (all(isinstance(q, int) for q in tempo_quantization)
            and tempo_quantization[0] <= tempo_quantization[1]
            and tempo_quantization[2] > 0), 'Bad tempo_quantization'

    assert len(midi.instruments) > 0, 'No tracks in MidiFile'
    for inst in midi.instruments:
        inst.remove_invalid_notes(verbose=False)
    # print('original ticks per beat:', midi.ticks_per_beat)
    change_tpq(midi, tpq)

    merge_tracks(midi, max_track_number)

    for i, track in enumerate(midi.instruments):
        if track.is_drum:
            midi.instruments[i].program = 128
    token_list = midi_to_token_list(
        midi,
        tpq,
        max_duration,
        velocity_step,
        use_cont_note,
        tempo_quantization
    )

    text_list = list(map(token_to_str, token_list))
    return ' '.join(text_list)


def handle_note_continuation(
        is_cont: bool,
        note_attrs: Tuple[int, int, int, int],
        cur_time: int,
        pending_cont_notes: Dict[int, Dict[Tuple[int, int, int], List[int]]]
    ) -> Union[None, Note]:
    # pending_cont_notes is dict object so it is pass by reference
    pitch, duration, velocity, track_number = note_attrs
    info = (pitch, velocity, track_number)
    # check if there is a note to continue at this time position
    onset = -1
    if cur_time in pending_cont_notes:
        if info in pending_cont_notes[cur_time]:
            if len(pending_cont_notes[cur_time][info]) > 0:
                onset = pending_cont_notes[cur_time][info].pop()
            #     print(cur_time, 'onset', onset, 'pop', pending_cont_notes)
            # else:
            #     print(cur_time, 'onset no pop', pending_cont_notes)
            if len(pending_cont_notes[cur_time][info]) == 0:
                pending_cont_notes[cur_time].pop(info)
            if len(pending_cont_notes[cur_time]) == 0:
                pending_cont_notes.pop(cur_time)
        else:
            onset = cur_time
    else:
        onset = cur_time

    if is_cont:
        # this note is going to connect to a note after max_duration
        pending_cont_time = cur_time + duration
        if pending_cont_time not in pending_cont_notes:
            pending_cont_notes[pending_cont_time] = dict()
        if info in pending_cont_notes[pending_cont_time]:
            pending_cont_notes[pending_cont_time][info].append(onset)
        else:
            pending_cont_notes[pending_cont_time][info] = [onset]
        # print(cur_time, 'onset', onset, 'append', pending_cont_notes)
    else:
        # assert 0 <= pitch < 128
        return Note(
            start=onset,
            end=cur_time+duration,
            pitch=pitch,
            velocity=velocity)
    return None


def piece_to_midi(
        piece: str,
        tpq: int,
        ignore_pending_note_error: bool = True) -> MidiFile:
    # tick time == tpq time
    midi = MidiFile(ticks_per_beat=tpq)

    text_list = piece.split(' ')
    assert (text_list[0] == tokens.BEGIN_TOKEN_STR
            and text_list[-1] == tokens.END_TOKEN_STR), \
        (f'No BOS and EOS at start and end.'
         f'Get: {text_list[0]} and {text_list[-1]}')

    cur_time = 0
    cur_position = -1
    is_head = True
    cur_measure_length = 0
    cur_measure_onset = 0
    cur_time_signature = None
    pending_cont_notes: Dict[int, Dict[Tuple[int, int, int], List[int]]]
    pending_cont_notes = dict()
    track_number_program_mapping = dict()
    # from track number to the track index in file
    track_number_index_mapping = dict()
    for text in text_list[1:-1]:
        typename = text[0]
        if typename == tokens.TRACK_EVENTS_CHAR:
            assert is_head, 'Track token at body'
            instrument, track_number = map(b36strtoi, text[1:].split(':'))
            assert track_number not in track_number_program_mapping, \
                'Repeated track number'
            track_number_program_mapping[track_number] = instrument

        elif typename == tokens.SEP_TOKEN_STR[0]:
            assert is_head, 'Seperator token in body'
            is_head = False
            assert len(track_number_program_mapping) > 0, 'No track in head'
            for track_number, program in track_number_program_mapping.items():
                track_number_index_mapping[track_number] = len(midi.instruments)
                midi.instruments.append(
                    Instrument(
                        program=(program%128),
                        is_drum=(program==128),
                        name=f'Track_{track_number}'
                    )
                )

        elif typename == tokens.MEASURE_EVENTS_CHAR:
            assert not is_head
            cur_position = -1

            numer, denom = (b36strtoi(x) for x in text[1:].split('/'))
            cur_measure_onset += cur_measure_length
            cur_time = cur_measure_onset

            cur_measure_length = 4 * tpq * numer // denom
            if cur_time_signature != (numer, denom):
                cur_time_signature = (numer, denom)
                midi.time_signature_changes.append(
                    TimeSignature(
                        numerator=numer,
                        denominator=denom,
                        time=cur_time
                    )
                )

        elif typename == tokens.POSITION_EVENTS_CHAR:
            assert not is_head, 'Position token at head'
            cur_position = b36strtoi(text[1:])
            cur_time = cur_position + cur_measure_onset

        elif typename == tokens.TEMPO_EVENTS_CHAR:
            assert not is_head, 'Tempo token at head'
            assert cur_position >= 0, 'No position before tempo'
            midi.tempo_changes.append(
                TempoChange(tempo=b36strtoi(text[1:]), time=cur_time)
            )

        elif typename == tokens.NOTE_EVENTS_CHAR:
            assert not is_head, 'Note token at head'
            assert cur_position >= 0, 'No position before note'

            if text[1] == '~':
                is_cont = True
                note_attrs = tuple(map(b36strtoi, text[3:].split(':')))
            else:
                is_cont = False
                note_attrs = tuple(map(b36strtoi, text[2:].split(':')))

            # note_attrs is (pitch, duration, velocity, track_number)
            assert note_attrs[3] in track_number_program_mapping, \
                'Note not in used track'
            n = handle_note_continuation(
                is_cont,
                note_attrs,
                cur_time,
                pending_cont_notes
            )
            if n is not None:
                cur_track_index = track_number_index_mapping[note_attrs[3]]
                midi.instruments[cur_track_index].notes.append(n)

        elif typename == tokens.MULTI_NOTE_EVENTS_CHAR:
            assert not is_head, 'Multi-note token at head'
            assert cur_position >= 0, 'No position before multi-note'

            shape_string, *other_attrs = text[1:].split(':')
            relnote_list = []
            for s in shape_string[:-1].split(';'):
                if s[-1] == '~':
                    relnote = [True] + [b36strtoi(a) for a in s[:-1].split(',')]
                else:
                    relnote = [False] + [b36strtoi(a) for a in s.split(',')]
                relnote_list.append(relnote)

            base_pitch, stretch_factor, velocity, track_number = (
                map(b36strtoi, other_attrs)
            )
            assert track_number in track_number_program_mapping, \
                'Multi-note not in used track'

            cur_track_index = track_number_index_mapping[track_number]
            for is_cont, rel_onset, rel_pitch, rel_dur in relnote_list:
                note_attrs = (
                    base_pitch + rel_pitch,
                    rel_dur * stretch_factor,
                    velocity,
                    track_number
                )
                # when using BPE, sometimes model will generated values
                # such that (relative_pitch + base_pitch) outside of limit
                # just ignore it
                if not 0 <= note_attrs[0] < 128:
                    continue
                onset_time = cur_time + rel_onset * stretch_factor
                n = handle_note_continuation(
                    is_cont,
                    note_attrs,
                    onset_time,
                    pending_cont_notes
                )
                if n is not None:
                    midi.instruments[cur_track_index].notes.append(n)
        else:
            raise ValueError(f'Bad token string: {text}')

    iter_count = 0
    while len(pending_cont_notes) > 0 and iter_count <= 3:
        # with multinote, sometimes a note will appears earlier than
        # the continuing note it is going to be appending to
        while True:
            adding_note_count = 0
            for track_index, inst in enumerate(midi.instruments):
                notes_to_remove = []
                notes_to_add = []
                for n in inst.notes:
                    if n.start not in pending_cont_notes:
                        continue
                    info = (n.pitch, n.velocity, track_index)
                    if info not in pending_cont_notes[n.start]:
                        continue
                    notes_to_remove.append(n)
                    new_note = handle_note_continuation(
                        False,
                        (n.pitch, n.end-n.start, n.velocity, track_index),
                        n.start,
                        pending_cont_notes)
                    notes_to_add.append(new_note)
                adding_note_count += len(notes_to_add)
                for n in notes_to_remove:
                    inst.notes.remove(n)
                for n in notes_to_add:
                    inst.notes.append(n)
            if adding_note_count == 0:
                break
        iter_count += 1

    assert ignore_pending_note_error or len(pending_cont_notes) == 0, \
        f'There are unclosed continuing notes: {pending_cont_notes}'

    # handle unclosed continuing notes by treating them as regular notes
    # that end at their current pending time
    for pending_time, info_onsets_dict in pending_cont_notes.items():
        for info, onset_list in info_onsets_dict.items():
            pitch, velocity, track_number = info
            for onset in onset_list:
                n = Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=onset,
                    end=pending_time
                )
                cur_track_index = track_number_index_mapping[track_number]
                midi.instruments[cur_track_index].notes.append(n)
    return midi


def get_first_k_measures(text_list: str, k: int) -> List[str]:
    """
    Input expect a valid text list.

    Return the text list of the first to k-th (inclusive) measures of
    the piece.

    The end-of-sequence token would be DROPPED.

    If k = 0, return just the head section.
    """
    assert isinstance(k, int) and k >= 0, \
        f'k must be positive integer or zero, get {k}'
    m_count = 0
    end_index = 0
    for i, text in enumerate(text_list):
        if text[0] in (tokens.MEASURE_EVENTS_CHAR, tokens.END_TOKEN_STR):
            m_count += 1
        if m_count > k:
            end_index = i
            break
    if end_index == 0:
        # just return the original text list
        return text_list
    return text_list[:end_index]


def get_after_k_measures(text_list: str, k: int) -> List[str]:
    """
    Input expect a valid text list.

    Return the text list of the (k + 1)-th to the last measures.

    The end-of-sequence token would be KEEPED.
    """
    assert isinstance(k, int) and k > 0, f'k must be positive integer, get {k}'
    m_count = 0
    head_end_index = 0
    begin_index = 0
    for i, text in enumerate(text_list):
        if text == tokens.SEP_TOKEN_STR:
            head_end_index = i + 1
        elif text[0] == tokens.MEASURE_EVENTS_CHAR:
            m_count += 1
        if m_count > k:
            begin_index = i
            break
    if begin_index == 0:
        # just return a piece that have no notes
        return text_list[:head_end_index] + [tokens.END_TOKEN_STR]
    return text_list[:head_end_index] + text_list[begin_index:]


def get_first_k_ticks(text_list: str, tpq, k) -> List[str]:
    cur_time = 0
    cur_measure_onset = 0
    cur_measure_length = 0
    end_index = 0
    for i, text in enumerate(text_list):
        typename = text[0]
        if typename == tokens.MEASURE_EVENTS_CHAR:
            numer, denom = (b36strtoi(x) for x in text[1:].split('/'))
            cur_measure_onset += cur_measure_length
            cur_time = cur_measure_onset
            cur_measure_length = 4 * tpq * numer // denom

        elif typename == tokens.POSITION_EVENTS_CHAR:
            position = b36strtoi(text[1:])
            cur_time = position + cur_measure_onset

        elif typename == tokens.TEMPO_EVENTS_CHAR:
            if ':' in text[1:]:
                position = text[1:].split(':')[1]
                cur_time = position + cur_measure_onset

        elif typename == tokens.NOTE_EVENTS_CHAR:
            attrs = text.split(':')
            if len(attrs) == 6:
                position = b36strtoi(attrs[5])
                cur_time = position + cur_measure_onset

        elif typename == tokens.MULTI_NOTE_EVENTS_CHAR:
            attrs = text[1:].split(':')
            if len(attrs) == 6:
                position = b36strtoi(attrs[5])
                cur_time = position + cur_measure_onset

        if cur_time > k:
            end_index = i
            break

    if end_index == 0:
        # just return the original text list
        return text_list
    return text_list[:end_index]
