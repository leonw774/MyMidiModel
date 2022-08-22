from io import IOBase
import random
import re
from time import time

from muspy import read_midi, Music, Track, Tempo, TimeSignature, Note

from .tokens import *


def merge_drums(midi: Music) -> None:
    """
        merge all drums into one instruments
    """
    merged_perc_notes = []
    for track in midi.tracks:
        if len(track.notes) > 0 and track.is_drum:
            merged_perc_notes.extend(track.notes)
    if len(merged_perc_notes) > 0:
        # remove duplicate
        merged_perc_notes = list(set(merged_perc_notes))
        merged_perc_notes.sort(key=lambda x: x.start)
        merged_perc_inst = Track(program=128, is_drum=True, name="merged_drums")
        merged_perc_inst.notes = merged_perc_notes
        new_instruments = [track for track in midi.tracks if not track.is_drum] + [merged_perc_inst]
        midi.tracks = new_instruments

def merge_tracks(
        midi: Music,
        max_track_number: int,
        use_merge_drums: bool) -> None:

    if use_merge_drums:
        merge_drums(midi)

    if len(midi.tracks) > max_track_number:
        tracks = list(midi.tracks) # shallow copy
        # place drum track or the most note track at first
        tracks.sort(key=lambda x: (not x.is_drum, -len(x.notes)))
        good_tracks_mapping = dict()
        bad_tracks_mapping = dict()

        for track in tracks[:max_track_number]:
            track_attr = (track.program, track.is_drum)
            if track_attr in good_tracks_mapping:
                good_tracks_mapping[track_attr].append(track)
            else:
                good_tracks_mapping[track_attr] = [track]
        for track in tracks[max_track_number:]:
            track_attr = (track.program, track.is_drum)
            if track_attr in bad_tracks_mapping:
                bad_tracks_mapping[track_attr].append(track)
            else:
                bad_tracks_mapping[track_attr] = [track]

        # print(midi_file_path)
        for bad_track_attr, bad_track_list in bad_tracks_mapping.items():
            if bad_track_attr in good_tracks_mapping:
                for bad_track in bad_track_list:
                    # find one track to merge
                    rand_good_track = random.choice(good_tracks_mapping[bad_track_attr])
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
            f'track number exceed max_track_number after merge: {len(good_tracks_mapping)}'
        midi.tracks = good_tracks_mapping


def quatize_to_nth(time_in_tick: int, ticks_per_nth: float) -> int:
    q = round(time_in_tick / ticks_per_nth)
    return q if q != 0 else 1


def quantize_tempo(tempo, tempo_quantization):
    # snap
    t = round(tempo_quantization[0] + tempo_quantization[2] * round((tempo - tempo_quantization[0]) / tempo_quantization[2]))
    # clamp
    t = min(tempo_quantization[1], max(tempo_quantization[0], t))
    return t


def get_note_tokens(midi: Music, nth: int, max_duration: int, velocity_step: int) -> list:
    """
        Return all note token in the midi as a list sorted in the ascending orders of
        onset time, track number, pitch, duration, velocity and instrument.
    """
    half_vel_step = velocity_step // 2
    nth_per_beat = nth // 4
    ticks_per_nth = midi.resolution / nth_per_beat

    # because sometimes mido give value that is more than 32 bits to note.start and note.end
    # and those extra bits makes the while loop hangs forever
    for track in midi.tracks:
        for note in track.notes:
            if note.start > (1 << 32):
                note.start = note.start & 0xFFFFFFFF
            if note.end > (1 << 32):
                note.end = note.end & 0xFFFFFFFF
            assert note.end >= note.start, "invalid note"

    note_token_list = [
        NoteToken(
            onset=quatize_to_nth(note.start, ticks_per_nth),
            track_number=track_number,
            pitch=note.pitch,
            duration=quatize_to_nth(note.end-note.start, ticks_per_nth),
            velocity=(note.velocity//velocity_step)*velocity_step+half_vel_step
        )
        for track_number, inst in enumerate(midi.tracks)
        for note in inst.notes
    ]

    # handle too long duration
    # continuing means this note this going to connect to a note after max_duration
    # it is represented as negtive to seperate from notes that's really max_duration long
    continuing_duration = -max_duration
    note_list_length = len(note_token_list)
    for i in range(note_list_length):
        note_token = note_token_list[i]
        if note_token.duration > max_duration:
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
        midi: Music,
        note_token_list: list,
        nth: int,
        tempo_quantization: list) -> list:
    """
        This return measure and tempo  The first measure have to contain the
        first note and the last note must end at the last measure
    """
    nth_per_beat = nth // 4
    ticks_per_nth = midi.resolution / nth_per_beat
    # note_token_list is sorted
    first_note_onset = note_token_list[0].onset
    last_note_end = note_token_list[-1].onset + note_token_list[-1].duration
    # print('first, last:', first_note_onset, last_note_end)

    assert len(midi.time_signatures) > 0, 'No time signature information'

    time_sig_dict_list = []
    # remove duplicated time_signatures
    for i, time_sig in enumerate(midi.time_signatures):
        time_sig_dict = {
            'n': time_sig.numerator,
            'd': time_sig.denominator,
            'time': quatize_to_nth(time_sig.time, ticks_per_nth)
        }
        assert time_sig.numerator != 0 and time_sig.denominator != 0, 'bad time signature'
        if len(time_sig_dict_list) == 0:
            time_sig_dict_list.append(time_sig_dict)
        elif (time_sig_dict['n'] != time_sig_dict_list[-1]['n']
                or time_sig_dict['d'] != time_sig_dict_list[-1]['d']):
            time_sig_dict_list.append(time_sig_dict)
    # print(time_sig_dict_list)

    # Some midi file has their first time signature begin at 1, but the rest of the time signature's
    # changing time are set as if the first time signature started at 0. Weird.
    # Not sure if we should "fix" it or just let them get excluded.
    if time_sig_dict_list[0]['time'] == 1:
        if len(time_sig_dict_list) > 1:
            nth_per_measure = round(nth_per_beat * time_sig_dict_list[0]['n'] * (4 / time_sig_dict_list[0]['d']))
            if time_sig_dict_list[1]['time'] % nth_per_measure == 0:
                time_sig_dict_list[0]['time'] = 0

    assert time_sig_dict_list[0]['time'] < first_note_onset, 'Time signature undefined before first note'

    measure_token_list = []
    for i, time_sig in enumerate(time_sig_dict_list):
        nth_per_measure = round(nth_per_beat * time_sig['n'] * (4 / time_sig['d']))
        cur_timesig_start_time = time_sig['time']
        # find end
        if i < len(time_sig_dict_list) - 1:
            next_timesig_start_time = time_sig_dict_list[i+1]['time']
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

        # check if the difference of starting time between current and next are multiple of current measure's length
        assert (next_timesig_start_time - cur_timesig_start_time) % nth_per_measure == 0, f'bad starting time of time signatures\n({next_timesig_start_time}-{cur_timesig_start_time}) % {nth_per_measure}'

        # find the first measure that at least contain the first note
        while cur_timesig_start_time + nth_per_measure <= first_note_onset:
            cur_timesig_start_time += nth_per_measure
        # now cur_measure_start_time is the REAL first measure

        assert is_supported_time_signature(time_sig['n'], time_sig['d']), f'Encounter unsupported time signature {(time_sig["n"], time_sig["d"])}'
        # print(f'TimeSig {i}:', cur_timesig_start_time, next_timesig_start_time, nth_per_measure)

        for cur_measure_start_time in range(cur_timesig_start_time, next_timesig_start_time, nth_per_measure):
            measure_token_list.append(
                MeasureToken(
                    onset=cur_measure_start_time,
                    time_signature=(time_sig['n'], time_sig['d'])
                )
            )
        # end for cur_measure_start_time
    # end for time_sig_tuple

    tempo_token_list = []
    tempo_dict_list = [] # (qpm, change_time)
    if len(midi.tempos) == 0:
        tempo_dict_list = [{
            'bpm': 120,
            'time': 0
        }]
    else:
        # remove duplicate change times
        prev_time = None
        for i, tempo in enumerate(midi.tempos):
            if i < len(midi.tempos) - 1:
                if midi.tempos[i+1].time < measure_token_list[0].onset * ticks_per_nth:
                    continue
            if tempo.time >= last_note_end * ticks_per_nth:
                continue
            cur_time = quatize_to_nth(tempo.time, ticks_per_nth)
            if prev_time != cur_time:
                bpm = quantize_tempo(tempo.qpm, tempo_quantization)
                tempo_dict_list.append({
                    'bpm': bpm,
                    'time': cur_time})
            prev_time = cur_time
    # print(tempo_dict_list)

    # same issue as time signature, but tempo can appear in any place so this is all we can do
    if tempo_dict_list[0]['time'] == 1 and time_sig_dict_list[0]['time'] == 0:
        tempo_dict_list[0]['time'] = 0

    tempo_token_list = []
    # use tempo token to record tempo info for `get_position_token` to handle
    # also remove duplicated the tempo qpms
    prev_bpm = None
    for i, tempo_dict in enumerate(tempo_dict_list):
        if prev_bpm != bpm:
            tempo_token_list.append(
                TempoToken(
                    onset=max(measure_token_list[0].onset, tempo_dict['time']),
                    bpm=tempo_dict['bpm']
                )
            )
        prev_bpm = bpm

    # print('\n'.join(map(str, measure_token_list))
    # print('\n'.join(map(str, tempo_token_list))
    # measure_token_list.sort() # dont have to sort
    tempo_token_list.sort() # sometime it is not sorted, dunno why
    return  measure_token_list, tempo_token_list


def get_position_infos(note_token_list: list, measure_token_list: list, tempo_token_list: list, position_method: str):
    note_measure_token_list = measure_token_list + note_token_list
    note_measure_token_list.sort()
    assert note_measure_token_list[0].type_priority == 2, 'first element of body is not measure token'

    position_token_list = []
    if position_method == 'attribute':
        note_token_list = []

    cur_measure_onset_time = 0
    cur_note_onset_time = 0
    last_added_position_onset = 0
    tempo_cursor = 0

    for token in note_measure_token_list:

        if token.type_priority == TYPE_PRIORITY['MeasureToken']:
            cur_measure_onset_time = token.onset

        if tempo_cursor < len(tempo_token_list):
            while tempo_token_list[tempo_cursor].onset <= token.onset and tempo_cursor < len(tempo_token_list):
                assert tempo_token_list[tempo_cursor].onset >= cur_measure_onset_time, 'negtive position'
                if position_method == 'event':
                    position_token_list.append(
                        PositionToken(
                            onset=tempo_token_list[tempo_cursor].onset,
                            position=tempo_token_list[tempo_cursor].onset-cur_measure_onset_time
                        )
                    )
                    last_added_position_onset = tempo_token_list[tempo_cursor].onset
                else:
                    tempo_token_list[tempo_cursor] = tempo_token_list[tempo_cursor]._replace(
                        position=tempo_token_list[tempo_cursor].onset-cur_measure_onset_time
                    )
                tempo_cursor += 1
                if tempo_cursor >= len(tempo_token_list):
                    break

        if token.type_priority == TYPE_PRIORITY['NoteToken'] and token.onset > cur_note_onset_time:
            if position_method == 'event':
                if token.onset > last_added_position_onset:
                    position_token_list.append(
                        PositionToken(
                            onset=token.onset,
                            position=token.onset-cur_measure_onset_time
                        )
                    )
                    last_added_position_onset = token.onset
            else:
                note_token_list.append(
                    token._replace(
                        position=token.onset-cur_measure_onset_time
                    )
                )
            cur_note_onset_time = token.onset

    return position_token_list


def get_head_tokens(midi: Music) -> list:
    track_token_list = [
        TrackToken(
            track_number=i,
            instrument=inst.program
        )
        for i, inst in enumerate(midi.tracks)
    ]
    return [BeginOfScoreToken()] + track_token_list


def midi_to_token_list(
        midi: Music,
        nth: int,
        max_duration: int,
        velocity_step: int,
        tempo_quantization: list,
        position_method: str) -> list:

    note_token_list = get_note_tokens(midi, nth, max_duration, velocity_step)

    # tempo_token_list would be empty list if tempo_method is 'measure_attribute'
    measure_token_list, tempo_token_list = get_time_structure_tokens(midi, note_token_list, nth, tempo_quantization)
    assert len(tempo_token_list) > 0, 'no tempo information retrieved'
    assert len(note_token_list) * 4 > len(measure_token_list) * len(midi.tracks), 'music too sparse'

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
        position_method: str,
        tempo_quantization: list, # [tempo_min, tempo_max, tempo_step]
        use_merge_drums: bool = True) -> list:
    """
        Parameters:
        - midi_file_path: midi file path
        - nth: to quantize notes to nth (96, 64 or 32)
        - max_track_number: the maximum tracks nubmer to keep in text, if the input midi has more
        'instruments' than this value, some tracks would be merged or discard
        - max_duration: max length of duration in unit of nth note
        - velocity_step: velocity to be quantize as (velocity_step//2, 3*velocity_step//2, 5*velocity_step//2, ...)
        - position_method: can be 'attribute' or 'event'
          - 'attribute' means position info is part of the note token
          - 'event' means position info will be its own event token.
            the token will be placed after the position token and before the note tokens in the same position
        - tempo_quantization: (min, max, step), where min and max are INCLUSIVE
        - use_merge_drums: to merge drums tracks or not
    """

    assert nth > 4 and nth % 4 == 0, 'nth is not multiple of 4'
    assert max_duration <= 127, f'max_duration greater than 127: {max_duration}'
    assert max_track_number <= 255, f'max_track_number is greater than 255: {max_track_number}'
    assert tempo_quantization[0] <= tempo_quantization[1] and tempo_quantization[2] > 0, 'bad tempo_quantization'

    start_time = time()

    midi = read_midi(midi_file_path)
    assert midi.resolution >= (nth//2), \
        f'midi.resolution ({midi.resolution}) is less than half of nth ({nth})'
    # print('ticks per beat:', midi.resolution)

    merge_tracks(midi, max_track_number, use_merge_drums)
    assert len(midi.tracks) > 0

    for i, track in enumerate(midi.tracks):
        if track.is_drum:
            midi.tracks[i].program = 128
    # print('Track:', len(midi.tracks))

    token_list = midi_to_token_list(midi, nth, max_duration, velocity_step, tempo_quantization, position_method)
    # print('Token list length:', len(token_list))
    # print('Time:', time()-start_time)
    # print('--------')

    text_list = list(map(token_to_text, token_list))
    # text = ' '.join(map(str, text_list))
    return text_list


def handle_note_continuation(
        is_cont: bool,
        note_attrs: tuple,
        cur_time: int,
        pending_cont_notes: dict):
    # pending_cont_notes is dict object so it is pass by reference
    pitch, duration, velocity, track_number = note_attrs
    info = (pitch, velocity, track_number)
    # check if there is a note to continue at this time position
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
        # print(cur_time, 'onset, onset, append', pending_cont_notes)
    else:
        return Note(velocity, pitch, onset, cur_time + duration)
    return None


def piece_to_midi(piece: str, nth: int) -> Music:
    # tick time == nth time
    midi = Music(resolution=nth//4)

    text_list = piece.split(' ')
    assert text_list[0] == 'BOS' and text_list[-1] == 'EOS'
    assert text_list[1][0] == 'R', 'No track token in head'

    # debug_str = ''
    cur_time = 0
    cur_measure_length = -1
    cur_measure_onset = -1
    cur_time_signature = None
    pending_cont_notes = dict()
    for text in text_list[1:-1]:
        typename = text[0]
        if typename == 'R':
            track_number, instrument = (tokstr2int(x) for x in text[1:].split(':'))
            assert len(midi.tracks) == track_number
            midi.tracks.append(
                Track(program=(instrument%128), is_drum=(instrument==128), name=f'Track_{track_number}')
            )

        elif typename == 'M':
            numer, denom = (tokstr2int(x) for x in text[1:].split('/'))
            if cur_measure_onset == -1: # first measure
                cur_measure_onset = 0
                cur_time = 0
            else:
                assert cur_measure_length > 0
                cur_measure_onset += cur_measure_length
                cur_time = cur_measure_onset

            cur_measure_length = 4 * numer * round(midi.resolution / denom)
            if cur_time_signature != (numer, denom):
                cur_time_signature = (numer, denom)
                midi.time_signatures.append(TimeSignature(numerator=numer, denominator=denom, time=cur_time))
            # print("M", attr, cur_measure_onset)

        elif typename == 'P':
            position = tokstr2int(text[1:])
            cur_time = position + cur_measure_onset
            # print('P', attr)

        elif typename == 'T':
            if ':' in text[1:]:
                qpm, position = (tokstr2int(x) for x in text[1:].split(':'))
                cur_time = position + cur_measure_onset
                midi.tempo.append(Tempo(qpm=qpm, time=cur_time))
            else:
                midi.tempos.append(Tempo(qpm=tokstr2int(text[1:]), time=cur_time))
            # print('T', attr, cur_time)

        elif typename == 'N':
            is_cont = (text[1] == '~')
            if is_cont:
                note_attr = tuple(tokstr2int(x) for x in text[3:].split(':'))
            else:
                note_attr = tuple(tokstr2int(x) for x in text[2:].split(':'))
            if len(note_attr) == 5:
                cur_time = note_attr[4] + cur_measure_onset
                note_attr = note_attr[:4]
            n = handle_note_continuation(is_cont, note_attr, cur_time, pending_cont_notes)
            if n is not None:
                midi.tracks[note_attr[3]].notes.append(n)
            # debug_str += text + ' onset' + str(cur_time) + ' ' + str(n) + '\n'

        elif typename == 'S':
            shape_string, *puvt = text[1:].split(':')
            relnote_list = []
            for s in shape_string[:-1].split(';'):
                is_cont = (s[-1] == '~')
                if is_cont:
                    relnote = [tokstr2int(a) for a in s[:-1].split(',')]
                else:
                    relnote = [tokstr2int(a) for a in s.split(',')]
                relnote = [is_cont] + relnote
                relnote_list.append(relnote)
            base_pitch, time_unit, velocity, track_number = (tokstr2int(x) for x in puvt)
            # debug_str += text
            for is_cont, rel_onset, rel_pitch, rel_dur in relnote_list:
                note_attr = (base_pitch + rel_pitch, rel_dur * time_unit, velocity, track_number)
                onset_time_in_tick = cur_time + rel_onset * time_unit
                n = handle_note_continuation(is_cont, note_attr, onset_time_in_tick, pending_cont_notes)
                if n is not None:
                    # print(n)
                    midi.tracks[track_number].notes.append(n)
                # debug_str += ' onset' + str(onset_time_in_tick) + ' ' + str(n) + ','
            # debug_str += '\n'
        else:
            raise TokenParseError(f'bad token string: {text}')

    if len(pending_cont_notes) > 0:
        # try to repair
        for track_number, inst in enumerate(midi.tracks):
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
            for n in notes_to_remove:
                inst.notes.remove(n)
            for n in notes_to_add:
                inst.notes.append(n)

    assert len(pending_cont_notes) == 0, f'There are unclosed continuing notes: {pending_cont_notes}'
    # handle unclosed continuing notes
    # for offset, info_onsets_dict in pending_cont_notes:
    #     for info, onset_list in info_onsets_dict:
    #         pitch, velocity, track_number = info
    #         for onset in onset_list:
    #             n = Note(velocity=velocity, pitch=pitch, start=onset, end=offset)
    #             midi.tracks[track_number].notes.append(n)
    return midi
