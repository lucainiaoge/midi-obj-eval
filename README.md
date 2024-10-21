# midi-obj-eval
A midi objective evaluation toolkit with features of [mgeval](https://github.com/RichardYang40148/mgeval/tree/master)

# usage

To evaluate single midi file, refer to the following command:
```
python single_midi_eval.py -midi-path ./set_classical/original_bagatell.mid -out-dir results
```

To compare two midi files, refer to the following command:
```
python midi_file_comparison.py -midi-path1 ./set_classical/original_bagatell.mid -midi-path2 ./set_jazz/generated_bagatell.mid -out-dir results
```

To evaluate multiple midi files, refer to the following command:
```
python multiple_midi_eval.py -midi-dir ./set_classical -out-dir results
```
