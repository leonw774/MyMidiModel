import random
from typing import Tuple, List, Dict, Union
# from time import time

from miditoolkit import MidiFile, Instrument, TempoChange, TimeSignature, Note

from .tokens import (
    NoteToken, TempoToken, PositionToken, MeasureToken, TrackToken, BeginOfScoreToken, EndOfScoreToken,
    TYPE_PRIORITY,
    is_supported_time_signature,
    token_to_str,
    b36str2int,
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
        new_instruments = [track for track in midi.instruments if not track.is_drum] + [merged_perc_inst]
        midi.instruments = new_instruments

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
        # place drum track or the most note track at first
        tracks.sort(key=lambda x: (not x.is_drum, -len(x.notes)))
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


def quantize_tempo(tempo: int, tempo_quantization: Tuple[int, int, int]) -> int:
    # snap
    t = round(tempo_quantization[0] + tempo_quantization[2] * round((tempo - tempo_quantization[0]) / tempo_quantization[2]))
    # clamp
    t = min(tempo_quantization[1], max(tempo_quantization[0], t))
    return t

NOTE_TIME_LIMIT = 0x1FFFFF # 2^21 - 1
# In midi format, time-delta is representated in variable-length bytes
# the format is 1xxxxxxx 1xxxxxxx ... 0xxxxxxx, the msb of each byte is 1 if it is not the last byte
# in this format, three bytes can represent up to 2^21 - 1
# when nth is 96, in tempo of bpm 240, 2^21 - 1 ticks is 364 minute
NOTE_DURATION_LIMIT = 0x3FFFF # 2^18 - 1
# if a note has duration > 45-ish minute, we think the file is likely corrupted

def get_note_tokens(midi: MidiFile, max_duration: int, velocity_step: int, use_cont_note: bool) -> list:
    """
        Return all note token in the midi as a list sorted in the ascending orders of
        onset time, track number, pitch, duration, velocity and instrument.
    """
    half_vel_step = velocity_step // 2

    for track in midi.instruments:
        for i, note in enumerate(track.notes):
            # sometimes mido/miditoolkit gives weird value to note.start and note.end that
            # some extra 1s appears in significant bits
            # sometimes the file itself is corrupted already
            assert note.start <= NOTE_TIME_LIMIT and note.end <= NOTE_TIME_LIMIT, 'note start/end too large, likely corrupted'
            assert note.end - note.start <= NOTE_DURATION_LIMIT, 'note duration too large, likely corrupted'

    note_token_list = [
        NoteToken(
            onset=note.start,
            track_number=track_number,
            pitch=note.pitch,
            duration=note.end-note.start,
            velocity=min(127, (note.velocity//velocity_step)*velocity_step+half_vel_step)
        )
        for track_number, track in enumerate(midi.instruments)
        for note in track.notes
        if note.start < note.end and note.start >= 0 and note.end > 0
    ]

    # handle too long duration
    # continuing means this note does not have NOTE_OFF and is going to be continued by another note after max_duration
    # it is represented with negtive duration to seperate from notes that's really max_duration long
    continuing_duration = -max_duration if use_cont_note else max_duration
    note_list_length = len(note_token_list)
    for i in range(note_list_length):
        note_token = note_token_list[i]
        if note_token.duration > max_duration:
            # if note_token.duration > max_duration * 20:
            #     print(bin(note_token.onset))
            #     print(bin(note_token.duration))
            #     print(bin(note_token.onset+note_token.duration))
            #     print('---')
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
        tempo_quantization: Tuple[int, int, int]) -> Tuple[list, list]:
    """
        This return measure list and tempo list. The first measure have to contain the
        first note and the last note must end in the last measure
    """
    nth_per_beat = nth // 4
    # note_token_list is sorted
    first_note_start = note_token_list[0].onset
    last_note_end = note_token_list[-1].onset + note_token_list[-1].duration
    # print('first, last:', first_note_start, last_note_end)

    assert len(midi.time_signature_changes) > 0, 'No time signature information'

    time_sig_list = []
    # remove duplicated time_signatures
    prev_nd = (None, None)
    for i, time_sig in enumerate(midi.time_signature_changes):
        if prev_nd != (time_sig.numerator, time_sig.denominator):
            time_sig_list.append(time_sig)
        prev_nd = (time_sig.numerator, time_sig.denominator)

    # Some midi file has their first time signature begin at 1, but the rest of the time signature's
    # changing time are set as if the first time signature started at 0. Weird.
    # Not sure if we should "fix" it or just let them get excluded.
    if time_sig_list[0].time == 1:
        if len(time_sig_list) > 1:
            nth_per_measure = round(nth_per_beat * time_sig_list[0].numerator * (4 / time_sig_list[0].denominator))
            if time_sig_list[1].time % nth_per_measure == 0:
                time_sig_list[0].time = 0
    # print(time_sig_dict_list)

    assert time_sig_list[0].time <= first_note_start, 'Time signature undefined before first note'

    measure_token_list = []
    for i, time_sig in enumerate(time_sig_list):
        assert is_supported_time_signature(time_sig.numerator, time_sig.denominator), \
            'Unsupported time signature'
            # f'Unsupported time signature {(time_sig.numerator, time_sig.denominator)}'
        # print(f'TimeSig {i}:', cur_timesig_start, next_timesig_start, nth_per_measure)

        nth_per_measure = round(nth_per_beat * time_sig.numerator * (4 / time_sig.denominator))
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

        # check if the difference of starting time between current and next are multiple of current measure's length
        assert (next_timesig_start - cur_timesig_start) % nth_per_measure == 0, \
            'Bad starting time of a time signature'
            # f'Bad starting time of a time signatures: ({next_timesig_start}-{cur_timesig_start})%{nth_per_measure}'

        # find the first measure that at least contain the first note
        while cur_timesig_start + nth_per_measure <= first_note_start:
            cur_timesig_start += nth_per_measure
        # now cur_measure_start_time is the REAL first measure

        for cur_measure_start_time in range(cur_timesig_start, next_timesig_start, nth_per_measure):
            if cur_measure_start_time > last_note_end:
                break
            measure_token_list.append(
                MeasureToken(
                    onset=cur_measure_start_time,
                    time_signature=(time_sig.numerator, time_sig.denominator)
                )
            )
        # end for cur_measure_start_time
    # end for time_sig_tuple

    # check if last measure contain last note
    last_measure = measure_token_list[-1]
    last_measure_onset = last_measure.onset
    last_measure_length = round(nth_per_beat * last_measure.time_signature[0] * (4 / last_measure.time_signature[1]))
    assert last_measure_onset <= last_note_end < last_measure_onset + last_measure_length, \
        "Last note did not ends in last measure"

    tempo_token_list = []
    tempo_list = [] # (tempo, time)
    if len(midi.tempo_changes) == 0:
        tempo_list = [TempoChange(120, 0)]
    else:
        # remove duplicated time and tempos
        prev_time = None
        prev_tempo = None
        for i, tempo in enumerate(midi.tempo_changes):
            if tempo.time > last_note_end:
                break

            if tempo.time < measure_token_list[0].onset:
                if i < len(midi.tempo_changes) - 1:
                    if midi.tempo_changes[i+1].time > measure_token_list[0].onset:
                        cur_time = measure_token_list[0].onset
                    else:
                        continue
                else:
                    cur_time = measure_token_list[0].onset
            else:
                cur_time = tempo.time

            cur_tempo = quantize_tempo(tempo.tempo, tempo_quantization)
            if cur_tempo != prev_tempo:
                # if is same time, replace last tempo
                if prev_time == cur_time:
                    tempo_list.pop()
                tempo_list.append(TempoChange(cur_tempo, cur_time))
            prev_tempo = cur_tempo
            prev_time = cur_time
    assert len(tempo_list) > 0, 'No tempo information retrieved'

    # same issue as time signature, but tempo can appear in any place so this is all we can do
    if tempo_list[0].time == 1 and measure_token_list[0].onset == 0:
        tempo_list[0].time = 0

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
    tempo_token_list.sort() # sometime it is not sorted, dunno why
    return  measure_token_list, tempo_token_list


def get_position_infos(note_token_list: list, measure_token_list: list, tempo_token_list: list, position_method: str):
    nmt_token_list = measure_token_list + note_token_list + tempo_token_list
    nmt_token_list.sort()

    position_token_list = []
    if position_method == 'attribute':
        note_token_list.clear()
        tempo_token_list.clear()

    cur_measure_onset = 0
    last_added_position_onset = 0

    for token in nmt_token_list:

        if token.type_priority == TYPE_PRIORITY['MeasureToken']:
            cur_measure_onset = token.onset

        elif token.type_priority == TYPE_PRIORITY['TempoToken']:
            if position_method == 'event':
            # if token.onset > last_added_position_onset: # don't need this because tempo must come before notes
                position_token_list.append(
                    PositionToken(
                        onset=token.onset,
                        position=token.onset-cur_measure_onset
                    )
                )
                last_added_position_onset = token.onset
            else:
                tempo_token_list.append(
                    token._replace(
                        position=token.onset-cur_measure_onset
                    )
                )
        elif token.type_priority == TYPE_PRIORITY['NoteToken']:
            if position_method == 'event':
                if token.onset > last_added_position_onset:
                    position_token_list.append(
                        PositionToken(
                            onset=token.onset,
                            position=token.onset-cur_measure_onset
                        )
                    )
                    last_added_position_onset = token.onset
            else:
                note_token_list.append(
                    token._replace(
                        position=token.onset-cur_measure_onset
                    )
                )
    return position_token_list


def get_head_tokens(midi: MidiFile) -> list:
    track_token_list = [
        TrackToken(
            track_number=i,
            instrument=track.program
        )
        for i, track in enumerate(midi.instruments)
    ]
    return [BeginOfScoreToken()] + track_token_list


def midi_to_token_list(
        midi: MidiFile,
        nth: int,
        max_duration: int,
        velocity_step: int,
        use_cont_note: bool,
        tempo_quantization: Tuple[int, int, int],
        position_method: str) -> list:

    note_token_list = get_note_tokens(midi, max_duration, velocity_step, use_cont_note)
    assert len(note_token_list) > 0, 'No notes in midi'
    measure_token_list, tempo_token_list = get_time_structure_tokens(midi, note_token_list, nth, tempo_quantization)
    assert measure_token_list[0].onset <= note_token_list[0].onset, 'First measure is after first note'
    pos_token_list = get_position_infos(note_token_list, measure_token_list, tempo_token_list, position_method)
    body_token_list = pos_token_list + note_token_list + measure_token_list + tempo_token_list
    body_token_list.sort()

    head_token_list = get_head_tokens(midi)
    full_token_list = head_token_list + body_token_list + [EndOfScoreToken()]

    return full_token_list

def midi_to_text_list(
        midi_file_path: str,
        nth: int,
        max_track_number: int,
        max_duration: int,
        velocity_step: int,
        use_cont_note: bool,
        tempo_quantization: Tuple[int, int, int], # [tempo_min, tempo_max, tempo_step]
        position_method: str,
        use_merge_drums: bool) -> list:
    """
        Parameters:
        - midi_file_path: midi file path
        - nth: to quantize notes to nth (96, 64 or 32)
        - max_track_number: the maximum tracks nubmer to keep in text, if the input midi has more
        'instruments' than this value, some tracks would be merged or discard
        - max_duration: max length of duration in unit of nth note
        - velocity_step: velocity to be quantize as (velocity_step//2, 3*velocity_step//2, 5*velocity_step//2, ...)
        - use_cont_note: have contiuing notes or not
        - position_method: can be 'attribute' or 'event'
          - 'attribute' means position info is part of the note token
          - 'event' means position info will be its own event token.
            the token will be placed after the position token and before the note tokens in the same position
        - tempo_quantization: (min, max, step), where min and max are INCLUSIVE
        - use_merge_drums: to merge drums tracks or not
    """

    assert nth > 4 and nth % 4 == 0, 'nth is not multiple of 4'
    assert max_track_number <= 255, f'max_track_number is greater than 255: {max_track_number}'
    assert tempo_quantization[0] <= tempo_quantization[1] and tempo_quantization[2] > 0, 'Bad tempo_quantization'

    # start_time = time()
    try:
        midi = MidiFile(midi_file_path)
    except Exception as e:
        raise RuntimeError('miditoolkit failed to parse the file') from e
    assert len(midi.instruments) > 0, 'No tracks in MidiFile'
    for inst in midi.instruments:
        inst.remove_invalid_notes(verbose=False)
    # print('original ticks per beat:', midi.ticks_per_beat)
    quantize_to_nth(midi, nth)

    merge_tracks(midi, max_track_number, use_merge_drums)

    for i, track in enumerate(midi.instruments):
        if track.is_drum:
            midi.instruments[i].program = 128
    token_list = midi_to_token_list(midi, nth, max_duration, velocity_step, use_cont_note, tempo_quantization, position_method)

    text_list = list(map(token_to_str, token_list))
    # text = ' '.join(map(str, text_list))
    return text_list


def handle_note_continuation(
        is_cont: bool,
        note_attrs: Tuple[int, int, int, int],
        cur_time: int,
        pending_cont_notes: Dict[int, Dict[Tuple[int, int, int], List[int]]]):
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
        if not 0 <= pitch < 128:
            # when using BPE, sometimes model will generated the combination that
            # the relative_pitch + base_pitch exceeds limit
            return None
        return Note(
            start=onset,
            end=cur_time+duration,
            pitch=pitch,
            velocity=velocity)
    return None


def piece_to_midi(piece: str, nth: int, ignore_panding_note_error: bool = False) -> MidiFile:
    # tick time == nth time
    midi = MidiFile(ticks_per_beat=nth//4)

    text_list = piece.split(' ')
    assert text_list[0] == 'BOS' and text_list[-1] == 'EOS', \
        f'No BOS and EOS at start and end, instead: {list(text_list[0])} and {list(text_list[-1])}'

    cur_time = 0
    is_head = True
    cur_measure_length = 0
    cur_measure_onset = 0
    cur_time_signature = None
    pending_cont_notes = dict()
    track_program_mapping = dict()
    for text in text_list[1:-1]:
        typename = text[0]
        if typename == 'R':
            assert is_head, 'R token at body'
            track_number, instrument = (b36str2int(x) for x in text[1:].split(':'))
            assert track_number not in track_program_mapping, 'Repeated track number'
            # shoud we use more strict track list?: not allow permutation, just 1, ..., n
            assert track_number == len(track_program_mapping), 'Track number not increasing by one'
            track_program_mapping[track_number] = instrument

        elif typename == 'M':
            # if this is the first measure
            if is_head:
                assert len(track_program_mapping) > 0, 'No track in head'
                track_numbers_list = list(track_program_mapping.keys())
                track_numbers_list.sort()
                # track number must at least be a permutation of 1, ..., n
                assert track_numbers_list == list(range(len(track_numbers_list))),\
                    'Track numbers are not consecutive integers starting from 1'
                for track_number in track_numbers_list:
                    program = track_program_mapping[track_number]
                    midi.instruments.append(
                        Instrument(program=(program%128), is_drum=(program==128), name=f'Track_{track_number}')
                    )
                is_head = False

            numer, denom = (b36str2int(x) for x in text[1:].split('/'))
            cur_measure_onset += cur_measure_length
            cur_time = cur_measure_onset

            cur_measure_length = round(nth * numer / denom)
            if cur_time_signature != (numer, denom):
                cur_time_signature = (numer, denom)
                midi.time_signature_changes.append(TimeSignature(numerator=numer, denominator=denom, time=cur_time))

        elif typename == 'P':
            assert not is_head, 'P token at head'
            position = b36str2int(text[1:])
            cur_time = position + cur_measure_onset

        elif typename == 'T':
            assert not is_head, 'T token at head'
            if ':' in text[1:]:
                tempo, position = (b36str2int(x) for x in text[1:].split(':'))
                cur_time = position + cur_measure_onset
                midi.tempo_changes.append(TempoChange(tempo=tempo, time=cur_time))
            else:
                midi.tempo_changes.append(TempoChange(tempo=b36str2int(text[1:]), time=cur_time))

        elif typename == 'N':
            assert not is_head, 'N token at head'
            is_cont = (text[1] == '~')
            if is_cont:
                note_attr = tuple(b36str2int(x) for x in text[3:].split(':'))
            else:
                note_attr = tuple(b36str2int(x) for x in text[2:].split(':'))
            # note_attr = pitch, duration, velocity, track_number, (position)
            assert note_attr[3] in track_program_mapping, 'Note not in used track'
            if len(note_attr) == 5:
                cur_time = note_attr[4] + cur_measure_onset
                note_attr = note_attr[:4]
            n = handle_note_continuation(is_cont, note_attr, cur_time, pending_cont_notes)
            if n is not None:
                midi.instruments[note_attr[3]].notes.append(n)

        elif typename == 'S':
            assert not is_head, 'S token at head'
            shape_string, *other_attr = text[1:].split(':')
            relnote_list = []
            for s in shape_string[:-1].split(';'):
                is_cont = ()
                if s[-1] == '~':
                    relnote = [True] + [b36str2int(a) for a in s[:-1].split(',')]
                else:
                    relnote = [False] + [b36str2int(a) for a in s.split(',')]
                relnote_list.append(relnote)
            base_pitch, time_unit, velocity, track_number, *position = (b36str2int(x) for x in other_attr)
            assert track_number in track_program_mapping, 'Note not in used track'
            if len(position) == 1:
                cur_time = position[0] + cur_measure_onset
            for is_cont, rel_onset, rel_pitch, rel_dur in relnote_list:
                note_attr = (base_pitch + rel_pitch, rel_dur * time_unit, velocity, track_number)
                onset_time = cur_time + rel_onset * time_unit
                n = handle_note_continuation(is_cont, note_attr, onset_time, pending_cont_notes)
                if n is not None:
                    midi.instruments[track_number].notes.append(n)
        else:
            raise ValueError(f'Bad token string: {text}')

    try_count = 0
    while len(pending_cont_notes) > 0 and try_count < 4:
        # with multinote, sometimes a note will appears earlier than the continuing note it is going to be appending to
        while True:
            adding_note_count = 0
            for track_number, inst in enumerate(midi.instruments):
                notes_to_remove = []
                notes_to_add = []
                for n in inst.notes:
                    if n.start in pending_cont_notes:
                        info = (n.pitch, n.velocity, track_number)
                        if info in pending_cont_notes[n.start]:
                            notes_to_remove.append(n)
                            new_note = handle_note_continuation(
                                False,
                                (n.pitch, n.end-n.start, n.velocity, track_number),
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

    if not ignore_panding_note_error:
        assert len(pending_cont_notes) == 0, f'There are unclosed continuing notes: {pending_cont_notes}'
    else:
    # handle unclosed continuing notes
        for offset, info_onsets_dict in pending_cont_notes.items():
            for info, onset_list in info_onsets_dict.items():
                pitch, velocity, track_number = info
                for onset in onset_list:
                    n = Note(velocity=velocity, pitch=pitch, start=onset, end=offset)
                    midi.instruments[track_number].notes.append(n)
    return midi
