import random
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


def merge_drums(midi: MidiFile) -> None:
    # merge all drum tracks into one
    merged_perc_notes = []
    for track in midi.instruments:
        if len(track.notes) > 0 and track.is_drum:
            merged_perc_notes.extend(track.notes)
    if len(merged_perc_notes) > 0:
        # remove duplicate
        merged_perc_notes = list(set(merged_perc_notes))
        merged_perc_notes.sort(key=lambda x: (x.start, x.end))
        merged_perc_inst = Instrument(program=128, is_drum=True, name='merged_drums')
        merged_perc_inst.notes = merged_perc_notes
        not_perc_inst_list = [track for track in midi.instruments if not track.is_drum]
        midi.instruments = not_perc_inst_list + [merged_perc_inst]

def merge_tracks(
        midi: MidiFile,
        max_track_number: int,
        use_merge_drums: bool) -> None:

    if use_merge_drums:
        merge_drums(midi)

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
        good_tracks_mapping = dict()
        bad_tracks_mapping = dict()

        for track in tracks[:max_track_number]:
            track_program = track.program if not track.is_drum else 128
            if track_program in good_tracks_mapping:
                good_tracks_mapping[track_program].append(track)
            else:
                good_tracks_mapping[track_program] = [track]
        for track in tracks[max_track_number:]:
            track_program = track.program if not track.is_drum else 128
            if track_program in bad_tracks_mapping:
                bad_tracks_mapping[track_program].append(track)
            else:
                bad_tracks_mapping[track_program] = [track]

        # print(midi_file_path)
        for bad_track_program, bad_track_list in bad_tracks_mapping.items():
            if bad_track_program in good_tracks_mapping:
                for bad_track in bad_track_list:
                    # find one track to merge
                    rand_good_track = random.choice(good_tracks_mapping[bad_track_program])
                    rand_good_track.notes.extend(bad_track.notes)
            else:
                # discard all tracks in bad_tracks_mapping
                pass

        good_tracks_mapping = [
            track
            for track_list in good_tracks_mapping.values()
            for track in track_list
        ]
        assert len(good_tracks_mapping) <= max_track_number, \
            f'Track number exceed max_track_number after merge: {len(good_tracks_mapping)}'
        midi.instruments = good_tracks_mapping


def tick_to_nth(tick: int, ticks_per_nth: Union[float, int]) -> int:
    return round(tick/ticks_per_nth)


def quantize_to_nth(midi: MidiFile, nth: int):
    # midi = copy.deepcopy(midi)
    ticks_per_nth = midi.ticks_per_beat / (nth // 4)
    for track in midi.instruments:
        for n in track.notes:
            n.start = tick_to_nth(n.start, ticks_per_nth)
            n.end = tick_to_nth(n.end, ticks_per_nth)
            if n.end == n.start: # to prevent note perishment
                n.end = n.start + 1
    for tempo in midi.tempo_changes:
        tempo.time = tick_to_nth(tempo.time, ticks_per_nth)
    for time_sig in midi.time_signature_changes:
        time_sig.time = tick_to_nth(time_sig.time, ticks_per_nth)
    # return midi


def quantize_tempo(tempo: int, tempo_quantization: Tuple[int, int, int]) -> int:
    tq = tempo_quantization
    # snap
    t = round(tq[0] + tq[2] * round((tempo - tq[0]) / tq[2]))
    # clamp
    t = min(tq[1], max(tq[0], t))
    return t


def quantize_velocity(velocty: int, velocity_step: int) -> int:
    # snap
    v = round(velocty / velocity_step) * velocity_step
    # clamp
    min_v = velocity_step
    max_v = int(127 / velocity_step) * velocity_step # use int() to round down
    v = min(max_v, max(min_v, v))
    return v


NOTE_TIME_LIMIT = 0x3FFFF # 2^18 - 1
# when tick/nth is 32, in tempo of bpm 120, 2^18 - 1 ticks are about 68 minutes
NOTE_DURATION_LIMIT = 0x3FFF # 2^14 - 1
# if a note may have duration > 17 minutes, we think the file is likely corrupted

def get_note_tokens(
        midi: MidiFile,
        max_duration: int,
        velocity_step: int,
        use_cont_note: bool) -> list:
    """
    Return all note token in the midi as a list sorted in the
    ascending orders of onset time, track number, pitch, duration,
    velocity and instrument.
    """
    for track in midi.instruments:
        for i, note in enumerate(track.notes):
            # sometimes mido/miditoolkit gives weird value to note.start and note.end that
            # some extra 1s appear in significant bits
            # sometimes the file itself is corrupted already
            assert note.start <= NOTE_TIME_LIMIT and note.end <= NOTE_TIME_LIMIT, \
                'Note start/end too large, likely corrupted'
            assert note.end - note.start <= NOTE_DURATION_LIMIT, \
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
            and (not track.is_drum or 35 <= note.pitch <= 81))
        # remove negtives and remove percussion notes not in the Genral MIDI percussion map
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
        nth: int,
        tempo_quantization: Tuple[int, int, int],
        supported_time_signatures: set) -> Tuple[list, list]:
    """
    Return measure list and tempo list. The first measure contains the
    first note and the last note ends in the last measure.
    """
    # note_token_list is sorted
    first_note_start = note_token_list[0].onset
    last_note_end = note_token_list[-1].onset + note_token_list[-1].duration
    # print('first, last:', first_note_start, last_note_end)

    if len(midi.time_signature_changes) == 0:
        # default is 4/4
        time_sig_list = [TimeSignature(4, 4, 0)]
        # assert False, 'No time signature information'
    else:
        time_sig_list = []
        # remove duplicated time_signatures
        prev_nd = (None, None)
        for i, time_sig in enumerate(midi.time_signature_changes):
            if prev_nd != (time_sig.numerator, time_sig.denominator):
                time_sig_list.append(time_sig)
            prev_nd = (time_sig.numerator, time_sig.denominator)

        # some midi file has their first time signature begin at time 1
        # but the rest of the changing times are set as if the first
        # time signature started at time 0. weird.
        if len(time_sig_list) > 0:
            if time_sig_list[0].time == 1:
                if len(time_sig_list) > 1:
                    nth_per_measure = round(
                        nth * time_sig_list[0].numerator / time_sig_list[0].denominator
                    )
                    if time_sig_list[1].time % nth_per_measure == 0:
                        time_sig_list[0].time = 0
                else:
                    time_sig_list[0].time = 0
    assert len(time_sig_list) > 0, 'No time signature information retrieved'
    # print(time_sig_list)

    # assert time_sig_list[0].time <= first_note_start, \
    #        'Time signature undefined before first note'

    # if the time signature starts after the first note, try 4/4 start at time 0
    if time_sig_list[0].time > first_note_start:
        time_sig_list = [(TimeSignature(4, 4, 0))] + time_sig_list

    measure_token_list = []
    for i, time_sig in enumerate(time_sig_list):
        assert (time_sig.numerator, time_sig.denominator) in supported_time_signatures, \
            'Unsupported time signature'
            # f'Unsupported time signature {(time_sig.numerator, time_sig.denominator)}'
        # print(f'TimeSig {i}:', cur_timesig_start, next_timesig_start, nth_per_measure)

        nth_per_measure = round(nth * time_sig.numerator / time_sig.denominator)
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
            num_of_measures = (time_to_last_note_end // nth_per_measure) + 1
            next_timesig_start = cur_timesig_start + num_of_measures * nth_per_measure

        # if note not started yet
        if next_timesig_start <= first_note_start:
            continue

        # check if the difference of starting time between current and next time signature
        # are multiple of current measure's length
        assert (next_timesig_start - cur_timesig_start) % nth_per_measure == 0, \
            'Bad starting time of a time signature'
            # f'({next_timesig_start}-{cur_timesig_start})%{nth_per_measure}'

        # find the first measure that at least contain the first note
        while cur_timesig_start + nth_per_measure <= first_note_start:
            cur_timesig_start += nth_per_measure
        # now cur_measure_start is the REAL first measure

        for cur_measure_start in range(cur_timesig_start, next_timesig_start, nth_per_measure):
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
    last_measure_onset = last_measure.onset
    last_measure_length = round(
        nth * last_measure.time_signature[0] / last_measure.time_signature[1]
    )
    assert last_measure_onset <= last_note_end < last_measure_onset + last_measure_length, \
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
                # if the prev tempo change is same time as current one, remove prev one
                if prev_time == cur_time:
                    tempo_list.pop()
                tempo_list.append(TempoChange(cur_tempo, cur_time))
            prev_tempo = cur_tempo
            prev_time = cur_time
    assert len(tempo_list) > 0, 'No tempo information retrieved'
    tempo_list.sort(key=lambda t: t.time) # sometime it is not sorted, dunno why

    # same weird issue as time signature
    # but tempo can appear in any place so this is all we can do
    if tempo_list[0].time == 1 and measure_token_list[0].onset == 0:
        tempo_list[0].time = 0
    # assert tempo_list[0].time == measure_token_list[0].onset, 'Tempo not start at time 0'
    # if the tempo change starts after the first measure, use the default 120 at first measure
    if tempo_list[0].time > measure_token_list[0].onset:
        tempo_list = [TempoChange(120, measure_token_list[0].onset)] + tempo_list

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


def get_position_infos(
        note_token_list: list,
        measure_token_list: list,
        tempo_token_list: list) -> list:
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


def get_head_tokens(midi: MidiFile) -> list:
    track_token_list = [
        TrackToken(
            track_number=i,
            instrument=track.program
        )
        for i, track in enumerate(midi.instruments)
    ]
    return track_token_list


def midi_to_token_list(
        midi: MidiFile,
        nth: int,
        max_duration: int,
        velocity_step: int,
        use_cont_note: bool,
        tempo_quantization: Tuple[int, int, int],
        supported_time_signatures: set) -> list:

    note_token_list = get_note_tokens(midi, max_duration, velocity_step, use_cont_note)
    assert len(note_token_list) > 0, 'No notes in midi'
    measure_token_list, tempo_token_list = get_time_structure_tokens(
        midi, note_token_list, nth, tempo_quantization, supported_time_signatures
    )
    assert measure_token_list[0].onset <= note_token_list[0].onset, \
        'First measure is after first note'
    pos_token_list = get_position_infos(note_token_list, measure_token_list, tempo_token_list)
    body_token_list = pos_token_list + note_token_list + measure_token_list + tempo_token_list
    body_token_list.sort()

    head_token_list = get_head_tokens(midi)
    full_token_list = (
        [BeginOfScoreToken()]
        + head_token_list + [SectionSeperatorToken()]
        + body_token_list + [EndOfScoreToken()]
    )

    return full_token_list


def midi_to_piece(
        midi: MidiFile,
        nth: int,
        max_track_number: int,
        max_duration: int,
        velocity_step: int,
        use_cont_note: bool,
        tempo_quantization: Tuple[int, int, int],
        use_merge_drums: bool) -> str:
    """
    Parameters:
    - `midi`: A miditoolkit MidiFile object
    - `nth`: Quantize onset and duration to multiples of the length of
      nth note. Have to ve multiple of 4. (Common: 32, 48, or 96)
    - `max_track_number`: The maximum tracks nubmer to keep in text,
      if the input midi has more 'instruments' than this value, some
      tracks would be merged or discarded.
    - `max_duration`: The max length of duration in unit of nth note.
    - `velocity_step`: Quantize velocity to multiples of velocity_step.
    - `use_cont_note`: Use contiuing notes or not. If not, cut off the
      over-length notes.
    - `tempo_quantization`: Three values are (min, max, step). where
      min and max are INCLUSIVE.
    - `use_merge_drums`: Merge drums tracks or not
    """

    assert nth > 4 and nth % 4 == 0, 'nth is not multiple of 4'
    assert max_track_number <= 255, f'max_track_number is greater than 255: {max_track_number}'
    assert tempo_quantization[0] <= tempo_quantization[1] and tempo_quantization[2] > 0, \
        'Bad tempo_quantization'

    assert len(midi.instruments) > 0, 'No tracks in MidiFile'
    for inst in midi.instruments:
        inst.remove_invalid_notes(verbose=False)
    # print('original ticks per beat:', midi.ticks_per_beat)
    quantize_to_nth(midi, nth)

    merge_tracks(midi, max_track_number, use_merge_drums)

    supported_time_signatures = get_supported_time_signatures()

    for i, track in enumerate(midi.instruments):
        if track.is_drum:
            midi.instruments[i].program = 128
    token_list = midi_to_token_list(
        midi,
        nth,
        max_duration,
        velocity_step,
        use_cont_note,
        tempo_quantization,
        supported_time_signatures
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


def piece_to_midi(piece: str, nth: int, ignore_pending_note_error: bool = True) -> MidiFile:
    # tick time == nth time
    midi = MidiFile(ticks_per_beat=nth//4)

    text_list = piece.split(' ')
    assert text_list[0] == tokens.BEGIN_TOKEN_STR and text_list[-1] == tokens.END_TOKEN_STR, \
        f'No BOS and EOS at start and end, instead: {text_list[0]} and {text_list[-1]}'

    cur_time = 0
    cur_position = -1
    is_head = True
    cur_measure_length = 0
    cur_measure_onset = 0
    cur_time_signature = None
    pending_cont_notes = dict()
    track_number_program_map = dict()
    track_number_index_map = dict() # track number don't have to be the track index in file
    for text in text_list[1:-1]:
        typename = text[0]
        if typename == tokens.TRACK_EVENTS_CHAR:
            assert is_head, 'Track token at body'
            instrument, track_number = (b36strtoi(x) for x in text[1:].split(':'))
            assert track_number not in track_number_program_map, 'Repeated track number'
            track_number_program_map[track_number] = instrument

        elif typename == tokens.SEP_TOKEN_STR[0]:
            assert is_head, 'Seperator token in body'
            is_head = False
            assert len(track_number_program_map) > 0, 'No track in head'
            for index, (track_number, program) in enumerate(track_number_program_map.items()):
                track_number_index_map[track_number] = index
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

            cur_measure_length = round(nth * numer / denom)
            if cur_time_signature != (numer, denom):
                cur_time_signature = (numer, denom)
                midi.time_signature_changes.append(
                    TimeSignature(numerator=numer, denominator=denom, time=cur_time)
                )

        elif typename == tokens.POSITION_EVENTS_CHAR:
            assert not is_head, 'Position token at head'
            cur_position = b36strtoi(text[1:])
            cur_time = cur_position + cur_measure_onset

        elif typename == tokens.TEMPO_EVENTS_CHAR:
            assert not is_head, 'Tempo token at head'
            assert cur_position >= 0, 'No position before tempo'
            midi.tempo_changes.append(TempoChange(tempo=b36strtoi(text[1:]), time=cur_time))

        elif typename == tokens.NOTE_EVENTS_CHAR:
            assert not is_head, 'Note token at head'
            assert cur_position >= 0, 'No position before note'

            if text[1] == '~':
                is_cont = True
                note_attrs = tuple(b36strtoi(x) for x in text[3:].split(':'))
            else:
                is_cont = False
                note_attrs = tuple(b36strtoi(x) for x in text[2:].split(':'))

            # note_attrs is (pitch, duration, velocity, track_number)
            assert note_attrs[3] in track_number_program_map, 'Note not in used track'
            n = handle_note_continuation(
                is_cont,
                note_attrs,
                cur_time,
                pending_cont_notes
            )
            if n is not None:
                midi.instruments[track_number_index_map[note_attrs[3]]].notes.append(n)

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

            base_pitch, stretch_factor, velocity, track_number = map(b36strtoi, other_attrs)
            assert track_number in track_number_program_map, 'Multi-note not in used track'

            for is_cont, rel_onset, rel_pitch, rel_dur in relnote_list:
                note_attrs = (
                    base_pitch + rel_pitch,
                    rel_dur * stretch_factor,
                    velocity,
                    track_number
                )
                # when using BPE, sometimes model will generated the combination that
                # the relative_pitch + base_pitch exceeds limit
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
                    midi.instruments[track_number_index_map[track_number]].notes.append(n)
        else:
            raise ValueError(f'Bad token string: {text}')

    try_count = 0
    while len(pending_cont_notes) > 0 and try_count < 4:
        # with multinote, sometimes a note will appears earlier than
        # the continuing note it is going to be appending to
        while True:
            adding_note_count = 0
            for track_index, inst in enumerate(midi.instruments):
                notes_to_remove = []
                notes_to_add = []
                for n in inst.notes:
                    if n.start in pending_cont_notes:
                        info = (n.pitch, n.velocity, track_index)
                        if info in pending_cont_notes[n.start]:
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
        try_count += 1

    if not ignore_pending_note_error:
        assert len(pending_cont_notes) == 0, \
            f'There are unclosed continuing notes: {pending_cont_notes}'
    else:
        # handle unclosed continuing notes by treating them as regular notes that
        # end at their current pending time
        for pending_time, info_onsets_dict in pending_cont_notes.items():
            for info, onset_list in info_onsets_dict.items():
                pitch, velocity, track_number = info
                for onset in onset_list:
                    n = Note(velocity=velocity, pitch=pitch, start=onset, end=pending_time)
                    midi.instruments[track_number_index_map[track_number]].notes.append(n)
    return midi


def get_first_k_measures(text_list: str, k: int) -> List[str]:
    """
    Input expect a valid text list.

    Return the text list of the first to k-th (inclusive) measures of
    the piece.

    The end-of-sequence token would be DROPPED.

    If k = 0, return just the head section.
    """
    assert isinstance(k, int) and k >= 0, f'k must be positive integer or zero, get {k}'
    m_count = 0
    end_index = 0
    for i, text in enumerate(text_list):
        if text[0] == tokens.MEASURE_EVENTS_CHAR or text == tokens.END_TOKEN_STR:
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


def get_first_k_nths(text_list: str, nth, k) -> List[str]:
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
            cur_measure_length = round(nth * numer / denom)

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
