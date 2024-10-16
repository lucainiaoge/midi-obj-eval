import os
import json
import argparse
import numpy as np
from core import PITCH_CLASSES
from core import extract_pretty_midi_features
from core import get_num_notes, get_used_pitch, get_pitch_class_histogram
from core import get_pitch_class_transition_matrix
from core import get_avg_ioi
import matplotlib.pyplot as plt

def evaluate_single_midi(midi_filepath, return_numpy = False):
    pretty_midi_features = extract_pretty_midi_features(midi_filepath)
    num_notes = get_num_notes(pretty_midi_features)
    used_pitch = get_used_pitch(pretty_midi_features)
    pitch_class_histogram = get_pitch_class_histogram(pretty_midi_features)
    pitch_class_transition_matrix = get_pitch_class_transition_matrix(pretty_midi_features, normalize=2)
    avg_ioi = get_avg_ioi(pretty_midi_features)
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

def plot_pitch_class_transition_matrix(pitch_class_transition_matrix, save_path):
    fig, ax = plt.subplots(1)
    ax.set_xticks(np.arange(len(PITCH_CLASSES)), labels=PITCH_CLASSES)
    ax.set_yticks(np.arange(len(PITCH_CLASSES)), labels=PITCH_CLASSES)
    ax.imshow(pitch_class_transition_matrix)
    fig.savefig(save_path)
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in single midi evaluation.')
    parser.add_argument(
        '-midi-path', type=str,
        help='The midi file to evaluate'
    )
    parser.add_argument(
        '-out-dir', type=str, default="./results",
        help='The output directory to save metrics'
    )
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    _, midi_name = os.path.split(args.midi_path)
    midi_name = os.path.splitext(midi_name)[0]

    metrics = evaluate_single_midi(args.midi_path, return_numpy=False)

    out_json_filename = midi_name + '_metrics.json'
    out_pctm_filename = midi_name + '_pctm.pdf'

    out_json_filepath = os.path.join(args.out_dir, out_json_filename)
    out_pctm_filepath = os.path.join(args.out_dir, out_pctm_filename)

    with open(out_json_filepath, "w") as outfile:
        json.dump(metrics, outfile)

    plot_pitch_class_transition_matrix(
        metrics["pitch_class_transition_matrix"],
        out_pctm_filepath
    )
    print("Saved metrics to {}".format(args.out_dir))