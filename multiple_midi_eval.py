import os
import glob
import json
import argparse
import numpy as np
from core import extract_pretty_midi_features_multiple
from core import get_num_notes, get_used_pitch_multiple, get_pitch_class_histogram
from core import get_pitch_class_transition_matrix
from core import get_avg_ioi
from single_midi_eval import evaluate_single_midi, plot_pitch_class_transition_matrix

def get_midi_files_from_dir(midi_dir):
    midi_filenames = []
    midi_filepaths = []
    possible_files = os.listdir(midi_dir)
    for filename in possible_files:
        ext = os.path.splitext(filename)[1]
        if ext is None:
            continue
        ext = ext.lower()
        if ext == ".mid" or ext == "midi":
            midi_filenames.append(filename)
            midi_filepaths.append(os.path.join(midi_dir, filename))
    return midi_filenames, midi_filepaths

def evaluate_multiple_midi_eval(midi_filepaths, return_numpy=False):
    num_notes = 0
    pitch_class_histogram = 0
    pitch_class_transition_matrix = 0
    avg_ioi = 0

    list_of_pretty_midi_features = extract_pretty_midi_features_multiple(midi_filepaths)
    used_pitch = get_used_pitch_multiple(list_of_pretty_midi_features)

    for pretty_midi_features in list_of_pretty_midi_features:
        this_num_notes = get_num_notes(pretty_midi_features)
        num_notes += this_num_notes
        pitch_class_histogram += get_pitch_class_histogram(pretty_midi_features) * this_num_notes
        pitch_class_transition_matrix += get_pitch_class_transition_matrix(pretty_midi_features, normalize=0)
        avg_ioi += get_avg_ioi(pretty_midi_features) * this_num_notes

    pitch_class_histogram /= num_notes
    pitch_class_transition_matrix /= pitch_class_transition_matrix.sum()
    avg_ioi /= num_notes

    metrics = {
        'num_notes': num_notes,
        'used_pitch': used_pitch,
        'pitch_class_histogram': pitch_class_histogram,
        'pitch_class_transition_matrix': pitch_class_transition_matrix,
        'avg_ioi': avg_ioi,
    }
    if return_numpy:
        return metrics
    for key in metrics.keys():
        if isinstance(metrics[key], (np.ndarray, np.generic)):
            metrics[key] = metrics[key].tolist()
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in multiple midi evaluation.')
    parser.add_argument(
        '-midi-dir', type=str,
        help='The directory of the midi files'
    )
    parser.add_argument(
        '-out-dir', type=str, default="./results",
        help='The output directory to save metrics'
    )
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    _, midi_filepaths = get_midi_files_from_dir(args.midi_dir)

    metrics = evaluate_multiple_midi_eval(midi_filepaths)

    _, midi_dir_name = os.path.split(args.midi_dir)
    out_json_filename = midi_dir_name + '_total_metrics.json'
    out_pctm_filename = midi_dir_name + '_total_pctm.pdf'

    out_json_filepath = os.path.join(args.out_dir, out_json_filename)
    out_pctm_filepath = os.path.join(args.out_dir, out_pctm_filename)

    with open(out_json_filepath, "w") as outfile:
        json.dump(metrics, outfile)

    plot_pitch_class_transition_matrix(
        metrics["pitch_class_transition_matrix"],
        out_pctm_filepath
    )
    print("Saved total metrics to {}".format(args.out_dir))