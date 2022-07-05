python3 data/midi_to_text.py --tempo-method measure_attribute --verbose --debug -o test_measure_attribute.txt data/example_midis/test.mid
python3 data/midi_to_text.py --tempo-method position_attribute --verbose -o test_position_attribute.txt data/example_midis/test.mid
python3 data/midi_to_text.py --tempo-method measure_event --verbose -o test_measure_event.txt data/example_midis/test.mid
python3 data/midi_to_text.py --tempo-method position_event --verbose -o test_position_event.txt data/example_midis/test.mid

python3 data/text_to_midi.py test_measure_attribute.txt
python3 data/text_to_midi.py test_position_attribute.txt
python3 data/text_to_midi.py test_measure_event.txt
python3 data/text_to_midi.py test_position_event.txt

python3 data/midi_to_text.py --tempo-method measure_attribute --verbose --debug -o test_measure_attribute_restored.txt test_measure_attribute.txt_restored_0.mid
python3 data/midi_to_text.py --tempo-method position_attribute --verbose --debug -o test_position_attribute_restored.txt test_position_attribute.txt_restored_0.mid
python3 data/midi_to_text.py --tempo-method measure_event --verbose --debug -o test_measure_event_restored.txt test_measure_event.txt_restored_0.mid
python3 data/midi_to_text.py --tempo-method position_event --verbose --debug -o test_position_event_restored.txt test_position_event.txt_restored_0.mid
