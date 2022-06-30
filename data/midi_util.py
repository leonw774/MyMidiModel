import random
from miditoolkit.midi.parser import MidiFile
from miditoolkit.midi.containers import Note, Instrument, TimeSignature, TempoChange
from time import time
from tokens import *


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


def get_measure_related_tokens(
        midi: MidiFile,
        note_token_list: list,
        nth: int,
        tempo_quantization: list) -> list:
    """
        This return measure, tempo, and time signature token. The first measure have to contain the
        first note and the last note must end at the last measure
    """
    nth_per_beat = nth // 4
    ticks_per_nth = midi.ticks_per_beat / nth_per_beat
    time_signature_changes = midi.time_signature_changes
    # note_token_list is sorted
    first_note_onset = note_token_list[0].onset
    last_note_end = note_token_list[-1].onset + note_token_list[-1].duration
    # if debug:
    #     print('first, last:', first_note_onset, last_note_end)

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

    tempo_min, tempo_max, tempo_step = tempo_quantization
    tempo_changes = midi.tempo_changes
    if len(tempo_changes) == 0:
        tempo_change_time = [0]
        tempo_bpms = [120]
    else:
        # remove duplicated
        prev_bpm = None
        tempo_change_time = []
        tempo_bpms = []
        for tempo_change in tempo_changes:
            bpm = tempo_change.tempo
            if prev_bpm != bpm:
                tempo_change_time.append(quatize_to_nth(tempo_change.time, ticks_per_nth))
                tempo_bpms.append(int(bpm))
            prev_bpm = bpm
    # to handle edge condition
    tempo_change_time.append(float('inf'))
    tempo_bpms.append(-1)
    # print(tempo_change_time)
    # print(tempo_bpms)

    tempo_token_list = []
    tempo_cursor = 0
    time_sig_token_list = []
    measure_token_list = []
    for i, time_sig_tuple in enumerate(time_sig_tuple_list):
        nth_per_measure = round(nth_per_beat * time_sig_tuple[0] * (4 / time_sig_tuple[1]))
        # find start
        cur_timesig_start_time = time_sig_tuple[2]
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

        if not is_supported_time_signature(time_sig_tuple[0], time_sig_tuple[1]):
            raise NotImplementedError(f'Encounter unsupported time signature {time_sig_tuple}')
        # print(f'TimeSig {i}:', cur_measure_start_time, next_timesig_start_time, nth_per_measure)

        for cur_measure_start_time in range(cur_timesig_start_time, next_timesig_start_time, nth_per_measure):
            measure_token_list.append(
                MeasureToken(
                    onset=cur_measure_start_time,
                    time_signature=time_sig_tuple)
            )
            # handle tempo
            cur_measure_tempo_list = []
            cur_measure_tempo_length_list = []
            # if the current tempo_change_time is in previous measure
            # because we know in previous measure the tempo_change_time stopped at the position
            # closest to this measure, so this measure's tempo could be different to previous's
            # but when current tempo_change_time is at least two measures away, it is only possible
            # to have tempo change when the next tempo change is in current measure
            if (cur_measure_start_time - nth_per_measure < tempo_change_time[tempo_cursor] or
                cur_measure_start_time <= tempo_change_time[tempo_cursor+1] < cur_measure_start_time + nth_per_measure):
                while True:
                    # look at the time between current and next tempo change
                    cur_measure_tempo_list.append(tempo_bpms[tempo_cursor])
                    cur_tempo_start = max(tempo_change_time[tempo_cursor], cur_measure_start_time)
                    # because the last change is set at infinity so no need to worry index error
                    cur_tempo_end = min(tempo_change_time[tempo_cursor+1], cur_measure_start_time + nth_per_measure)
                    cur_measure_tempo_length_list.append(cur_tempo_end - cur_tempo_start)
                    if tempo_change_time[tempo_cursor+1] > cur_measure_start_time + nth_per_measure:
                        # if the next tempo_change_time is not in current measure, stop increasing cursor
                        # because next measure needs value of current tempo to calculate
                        # when it reach the last "real" tempo change, its "next" tick change is at infinity
                        # so the tempo_cursor will never reach it
                        break
                    tempo_cursor += 1
                if len(cur_measure_tempo_list) > 1:
                    weighted_tempo = sum(
                        t * w / nth_per_measure
                        for t, w in zip(cur_measure_tempo_list, cur_measure_tempo_length_list)
                    )
                    # snap
                    weighted_tempo = round(tempo_min + tempo_step * round((weighted_tempo - tempo_min) / tempo_step))
                    # clamp
                    weighted_tempo = min(tempo_max, max(tempo_min, weighted_tempo))
                    prev_tempo = tempo_token_list[-1].bpm if len(tempo_token_list) != 0 else -1
                    if weighted_tempo != prev_tempo:
                        tempo_token_list.append(TempoToken(onset=cur_measure_start_time, bpm=weighted_tempo))

    measure_related_token_list = measure_token_list + time_sig_token_list + tempo_token_list
    # measure_related_token_list.sort() # dont have to sort
    return measure_related_token_list


def get_position_tokens(note_measure_tokens_list: list) -> list:
    position_token_list = []
    cur_measure_onset_time = 0
    cur_note_onset_time = 0
    for token in note_measure_tokens_list:
        if token.onset > cur_note_onset_time:
            if token.type_priority == TYPE_PRIORITY['MeasureToken']:
                cur_measure_onset_time = token.onset
            elif token.type_priority == TYPE_PRIORITY['NoteToken']:
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


def midi_to_tokens(
        midi: MidiFile,
        nth: int,
        max_duration: int,
        velocity_step: int,
        tempo_quantization: list,
        debug: bool = False) -> list:

    start_time=time()
    note_token_list = get_note_tokens(midi, nth, max_duration, velocity_step)
    if debug:
        print('Get notes:', len(note_token_list), 'time:', time()-start_time)
        start_time=time()

    measure_related_tokens_list = get_measure_related_tokens(midi, note_token_list, nth, tempo_quantization)
    assert len(note_token_list) * 2 > len(measure_related_tokens_list) * len(midi.instruments), 'music too sparse'
    note_measure_tokens_list = measure_related_tokens_list + note_token_list
    note_measure_tokens_list.sort()
    assert note_measure_tokens_list[0].type_priority == 2, 'first element of body is not measure token'
    if debug:
        print('Get measures:', len(measure_related_tokens_list), 'time:', time()-start_time)
        start_time=time()

    # align the first time to zero
    first_onset_time = note_measure_tokens_list[0].onset
    if first_onset_time != 0:
        note_measure_tokens_list = [
            token._replace(onset=token.onset-first_onset_time)
            for token in note_measure_tokens_list
        ]

    pos_token_list = get_position_tokens(note_measure_tokens_list)
    body_token_list = pos_token_list + note_measure_tokens_list
    body_token_list.sort()
    if debug:
        print('Get position:', len(pos_token_list), 'time:', time()-start_time)
        start_time=time()

    head_tokens_list = get_head_tokens(midi)
    full_token_list = head_tokens_list + body_token_list + [EndOfScoreToken()]
    if debug:
        print('Full token:', len(full_token_list))

    return full_token_list


def midi_to_text(
        midi_filepath: str,
        nth: int,
        max_track_number: int,
        max_duration: int,
        velocity_step: int,
        tempo_quantization: list, # [tempo_min, tempo_max, tempo_step]
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
        - use_merge_drums: to merge drums tracks or not
        - use_merge_sparse: to merge sparse tracks or not
    """

    assert nth > 4 and nth % 4 == 0, 'nth is not multiple of 4'
    assert tempo_quantization[0] <= tempo_quantization[1] and tempo_quantization[2] > 0, 'bad tempo_quantization'

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

    token_list = midi_to_tokens(midi, nth, max_duration, velocity_step, tempo_quantization, debug)

    # with open('tokens.txt', 'w+', encoding='utf8') as f:
    #     f.write('\n'.join(map(str, token_list)))
    if debug:
        start_time = time()
    text_list = map(token_to_text, token_list)
    text = ' '.join(map(str, text_list))
    if debug:
        print('Make text from tokens:', time()-start_time)
        print('--------')

    return text


def text_to_midi(texts: str, nth: int, debug=False) -> MidiFile:
    midi = MidiFile(ticks_per_beat=nth)
    ticks_per_nth = 4 # convenient huh

    text_list = texts.split(' ')
    assert text_list[0] == 'BOS' and text_list[-1] == 'EOS'
    assert text_list[1][0] == 'R', 'No track token in head'

    is_head = True
    cur_time_in_ticks = 0
    cur_measure_length_in_ticks = -1
    cur_measure_onset_in_ticks = -1
    cur_time_signature = None
    pending_continuing_notes = dict()
    for text in text_list[1:-1]:
        typename, attr = text_to_token(text)
        if is_head:
            if typename == 'R':
                track_number, instrument = attr
                assert len(midi.instruments) == track_number
                midi.instruments.append(
                    Instrument(program=instrument%128, is_drum=instrument==128, name=f'Track_{track_number}')
                )
            else:
                is_head = False
        if not is_head:
            if typename == 'M':
                if cur_measure_onset_in_ticks == -1: # first measure
                    cur_measure_onset_in_ticks = 0
                    cur_time_in_ticks = 0
                    cur_time_signature = attr
                    midi.time_signature_changes.append(TimeSignature(attr[0], attr[1], cur_time_in_ticks))
                    cur_measure_length_in_ticks = round(4 * attr[0] * (midi.ticks_per_beat / attr[1]))
                else:
                    assert cur_measure_length_in_ticks > 0
                    cur_measure_onset_in_ticks += cur_measure_length_in_ticks
                    cur_time_in_ticks = cur_measure_onset_in_ticks
                if cur_time_signature != attr:
                    midi.time_signature_changes.append(TimeSignature(n, d, cur_time_in_ticks))
                # if debug:
                #     print("M", attr, cur_measure_onset_in_ticks)

            elif typename == 'T':
                midi.tempo_changes.append(TempoChange(attr, cur_time_in_ticks))
                # if debug:
                #     print('T', attr, cur_time_in_ticks)

            elif typename == 'P':
                cur_time_in_ticks = attr * ticks_per_nth + cur_measure_onset_in_ticks
                # if debug:
                #     print('P', attr)

            elif typename == 'N':
                # add note to instrument
                pitch, duration, velocity, track_number = attr
                if cur_time_in_ticks in pending_continuing_notes:
                    info = (pitch, velocity, track_number)
                    if info in pending_continuing_notes[cur_time_in_ticks]:
                        if len(pending_continuing_notes[cur_time_in_ticks][info]) > 0:
                            onset = pending_continuing_notes[cur_time_in_ticks][info].pop()
                        # print(cur_time_in_ticks, 'onset pop', pending_continuing_notes)
                        if len(pending_continuing_notes[cur_time_in_ticks][info]) == 0:
                            pending_continuing_notes.pop(cur_time_in_ticks)
                else:
                    onset = cur_time_in_ticks

                if duration < 0:
                    # this note is going to connect to a note after max_duration
                    info = (pitch, velocity, track_number)
                    pending_continuing_time = cur_time_in_ticks + (-duration) * ticks_per_nth
                    if pending_continuing_time not in pending_continuing_notes:
                        pending_continuing_notes[pending_continuing_time] = dict()
                    if info in pending_continuing_notes[pending_continuing_time]:
                        pending_continuing_notes[pending_continuing_time][info].append(onset)
                    else:
                        pending_continuing_notes[pending_continuing_time] = {info : [onset]}
                    # print(cur_time_in_ticks, 'onset append', pending_continuing_notes)
                else:
                    n = Note(velocity, pitch, onset, cur_time_in_ticks + duration * ticks_per_nth)
                    midi.instruments[track_number].notes.append(n)
                    # if debug:
                    #     print('N', pitch, duration, velocity, track_number, instrument)
    # end loop text_list
    return midi


def make_para_yaml(args: dict) -> str:
    s = '---\n'
    lines = [
        f'{k}: {v}\n'
        for k, v in args.items()
        if v is not None
    ]
    s += ''.join(lines)
    s += '---\n'
    return s

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
        else:
            args_dict[k] = int(v)
    return args_dict

def file_to_paras_and_pieces(file_path: str):
    with open(file_path, 'r', encoding='utf8') as corpus_file:
        raw_file_texts = corpus_file.read()
    # get settings and main texts
    texts = raw_file_texts
    _, head, body = texts.split('---')
    paras = parse_para_yaml(head)
    body = body.lstrip('\n').rstrip('\n')
    pieces = body.split('\n')
    return paras, pieces
