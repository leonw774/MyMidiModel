from io import IOBase
import random
import re
from time import time

from miditoolkit import MidiFile, Instrument, TempoChange, TimeSignature, Note

from .tokens import *


def merge_drums(midi: MidiFile) -> None:
    """
        merge all drums into one instruments
    """
    merged_perc_notes = []
    for track in midi.instruments:
        if len(track.notes) > 0 and track.is_drum:
            merged_perc_notes.extend(track.notes)
    if len(merged_perc_notes) > 0:
        # remove duplicate
        merged_perc_notes = list(set(merged_perc_notes))
        merged_perc_notes.sort(key=lambda x: x.start)
        merged_perc_inst = Instrument(program=128, is_drum=True, name="merged_drums")
        merged_perc_inst.notes = merged_perc_notes
        new_instruments = [track for track in midi.Instruments if not track.is_drum] + [merged_perc_inst]
        midi.instruments = new_instruments

def merge_tracks(
        midi: MidiFile,
        max_track_number: int,
        use_merge_drums: bool) -> None:

    if use_merge_drums:
        merge_drums(midi)

    if len(midi.instruments) > max_track_number:
        tracks = list(midi.instruments) # shallow copy
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
        midi.instruments = good_tracks_mapping


def tick_to_nth(tick, ticks_per_nth):
    return round(tick/ticks_per_nth)


def quantize_to_nth(midi, nth):
    ticks_per_nth = midi.ticks_per_beat / (nth // 4)
    for track in midi.instruments:
        for n in track.notes:
            n.start = tick_to_nth(n.start, ticks_per_nth)
            n.end = tick_to_nth(n.end, ticks_per_nth)
    for tempo in midi.tempo_changes:
        tempo.time = tick_to_nth(tempo.time, ticks_per_nth)
    for time_sig in midi.time_signature_changes:
        time_sig.time = tick_to_nth(time_sig.time, ticks_per_nth)


def quantize_tempo(tempo, tempo_quantization):
    # snap
    t = round(tempo_quantization[0] + tempo_quantization[2] * round((tempo - tempo_quantization[0]) / tempo_quantization[2]))
    # clamp
    t = min(tempo_quantization[1], max(tempo_quantization[0], t))
    return t


def get_note_tokens(midi: MidiFile, max_duration: int, velocity_step: int) -> list:
    """
        Return all note token in the midi as a list sorted in the ascending orders of
        onset time, track number, pitch, duration, velocity and instrument.
    """
    half_vel_step = velocity_step // 2

    # sometimes mido give weird value to note.start and note.end
    # there are some those extra 1s near bit 30~32 that makes the while loop hangs forever
    # so we have to limit the duration, note.start and note.end under a certain threshold
    bit_clip = 1 << 20
    clip_mask = bit_clip - 1
    for track in midi.instruments:
        for note in track.notes:
            assert note.start >= 0 and note.end > 0, f"Invalid note: {(note)}"
            if note.start > bit_clip:
                note.start &= clip_mask
            if note.end > bit_clip:
                note.end &= clip_mask
            assert note.end >= note.start, f"Invalid note: {(note)}"
            # dont throw exception at note.start==note.end, we just need to remove them

    note_token_list = [
        NoteToken(
            onset=note.start,
            track_number=track_number,
            pitch=note.pitch,
            duration=note.end-note.start,
            velocity=(note.velocity//velocity_step)*velocity_step+half_vel_step
        )
        for track_number, inst in enumerate(midi.instruments)
        for note in inst.notes
        if note.start != note.end
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
        midi: MidiFile,
        note_token_list: list,
        nth: int,
        tempo_quantization: list) -> list:
    """
        This return measure and tempo  The first measure have to contain the
        first note and the last note must end at the last measure
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
        nth_per_measure = round(nth_per_beat * time_sig.numerator * (4 / time_sig.denominator))
        cur_timesig_start = time_sig.time
        # find end
        if i < len(time_sig_list) - 1:
            next_timesig_start = time_sig_list[i+1].time
        else:
            # measure should end when no notes are ending any more
            time_to_last_note_end = last_note_end - cur_timesig_start
            num_of_measures = (time_to_last_note_end // nth_per_measure) + 1
            next_timesig_start = cur_timesig_start + num_of_measures * nth_per_measure

        # if note already end
        if cur_timesig_start > last_note_end:
            break
        # if note not started yet
        if next_timesig_start <= first_note_start:
            continue

        # check if the difference of starting time between current and next are multiple of current measure's length
        assert (next_timesig_start - cur_timesig_start) % nth_per_measure == 0, \
            f'Bad starting time of time signatures: ({next_timesig_start}-{cur_timesig_start})%{nth_per_measure}'

        # find the first measure that at least contain the first note
        while cur_timesig_start + nth_per_measure <= first_note_start:
            cur_timesig_start += nth_per_measure
        # now cur_measure_start_time is the REAL first measure

        assert is_supported_time_signature(time_sig.numerator, time_sig.denominator), \
            f'Unsupported time signature {(time_sig["n"], time_sig["d"])}'
        # print(f'TimeSig {i}:', cur_timesig_start, next_timesig_start, nth_per_measure)

        for cur_measure_start_time in range(cur_timesig_start, next_timesig_start, nth_per_measure):
            measure_token_list.append(
                MeasureToken(
                    onset=cur_measure_start_time,
                    time_signature=(time_sig.numerator, time_sig.denominator)
                )
            )
        # end for cur_measure_start_time
    # end for time_sig_tuple

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

    for i, token in enumerate(nmt_token_list):

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
        tempo_quantization: list,
        position_method: str) -> list:

    note_token_list = get_note_tokens(midi, max_duration, velocity_step)
    measure_token_list, tempo_token_list = get_time_structure_tokens(midi, note_token_list, nth, tempo_quantization)
    assert measure_token_list[0].onset <= note_token_list[0].onset, 'First measure is after first note'
    assert len(note_token_list) * 4 > len(measure_token_list) * len(midi.instruments), 'music too sparse'
    pos_token_list = get_position_infos(note_token_list, measure_token_list, tempo_token_list, position_method)
    print(tempo_token_list)
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
    assert max_track_number <= 255, f'max_track_number is greater than 255: {max_track_number}'
    assert tempo_quantization[0] <= tempo_quantization[1] and tempo_quantization[2] > 0, 'bad tempo_quantization'

    # start_time = time()
    midi = MidiFile(midi_file_path)
    assert len(midi.instruments) > 0
    for i in range(len(midi.instruments)):
        midi.instruments[i].remove_invalid_notes(verbose=False)
    # print('original ticks per beat:', midi.ticks_per_beat)
    quantize_to_nth(midi, nth)

    merge_tracks(midi, max_track_number, use_merge_drums)

    for i, track in enumerate(midi.instruments):
        if track.is_drum:
            midi.instruments[i].program = 128

    token_list = midi_to_token_list(midi, nth, max_duration, velocity_step, tempo_quantization, position_method)

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
        # print(cur_time, 'onset', onset, 'append', pending_cont_notes)
    else:
        return Note(
            start=onset,
            end=cur_time+duration,
            pitch=pitch,
            velocity=velocity)
    return None


def piece_to_midi(piece: str, nth: int) -> MidiFile:
    # tick time == nth time
    midi = MidiFile(ticks_per_beat=nth//4)

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
            assert len(midi.instruments) == track_number
            midi.instruments.append(
                Instrument(program=(instrument%128), is_drum=(instrument==128), name=f'Track_{track_number}')
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

            cur_measure_length = round(nth * numer / denom)
            if cur_time_signature != (numer, denom):
                cur_time_signature = (numer, denom)
                midi.time_signature_changes.append(TimeSignature(numerator=numer, denominator=denom, time=cur_time))
            # print("M", attr, cur_measure_onset)

        elif typename == 'P':
            position = tokstr2int(text[1:])
            cur_time = position + cur_measure_onset
            # print('P', attr)

        elif typename == 'T':
            if ':' in text[1:]:
                tempo, position = (tokstr2int(x) for x in text[1:].split(':'))
                cur_time = position + cur_measure_onset
                midi.tempo.append(TempoChange(tempo=tempo, time=cur_time))
            else:
                midi.tempo_changes.append(TempoChange(tempo=tokstr2int(text[1:]), time=cur_time))
            # print('T', text, cur_time)

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
                midi.instruments[note_attr[3]].notes.append(n)
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
                    midi.instruments[track_number].notes.append(n)
                # debug_str += ' onset' + str(onset_time_in_tick) + ' ' + str(n) + ','
            # debug_str += '\n'
        else:
            raise TokenParseError(f'bad token string: {text}')

    if len(pending_cont_notes) > 0:
        # try to repair
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
    #             midi.instruments[track_number].notes.append(n)
    return midi
