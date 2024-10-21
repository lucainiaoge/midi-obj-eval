import os
import glob
import json
import argparse
import numpy as np
from core import PITCH_CLASSES
from core import extract_pretty_midi_features
from core import get_num_notes, get_used_pitch, get_pitch_class_histogram
from core import get_pitch_class_transition_matrix
from core import get_avg_ioi
import matplotlib.pyplot as plt
from single_midi_eval import evaluate_single_midi, plot_pitch_class_transition_matrix, plot_pitch_class_histogram
# from multiple_midi_eval import get_midi_files_from_dir, evaluate_multiple_midi_eval

def kl_div_discrete(dist1: np.ndarray, dist2: np.ndarray, epsilon=1e-5):
    dist1 = dist1 + epsilon
    dist2 = dist2 + epsilon
    return np.sum(dist1 * np.log(dist1 / dist2))

def compare_single_midi_metrics(metrics1, metrics2):
    metric_pairs = {}
    for key in metrics1.keys():
        assert key in metrics2, f"{key} should also be in the other metric dict"
        metric_pairs[key] = (metrics1[key], metrics2[key])
        if key == 'pitch_class_histogram':
            metric_pairs["pitch_class_kl"] = kl_div_discrete(
                np.array(metrics1[key]),
                np.array(metrics2[key]),
            )
    return metric_pairs

def plot_pitch_class_histogram_pair(pitch_class_histogram_pair, save_path, names = (None, None)):
    name1 = "series1" if names[0] is None else names[0]
    name2 = "series2" if names[1] is None else names[1]
    fig, ax = plt.subplots(1)
    x_axis = np.arange(len(PITCH_CLASSES))
    bias, width = 0.2, 0.4
    ax.bar(x_axis - bias, height=pitch_class_histogram_pair[0], width=width, label=name1)
    ax.bar(x_axis + bias, height=pitch_class_histogram_pair[1], width=width, label=name2)
    ax.set_xticks(x_axis, PITCH_CLASSES)
    ax.set_xlabel("Pitch Classes")
    ax.set_ylabel("Frequency")
    ax.set_title("Pitch Classes Histogram")
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)

def plot_pitch_class_transition_matrix_pair(pitch_class_transition_matrix_pair, save_path, names = (None, None)):
    name1 = "series1" if names[0] is None else names[0]
    name2 = "series2" if names[1] is None else names[1]
    fig, axs = plt.subplots(1, 2)
    axs[0].set_xticks(np.arange(len(PITCH_CLASSES)), labels=PITCH_CLASSES)
    axs[0].set_yticks(np.arange(len(PITCH_CLASSES)), labels=PITCH_CLASSES)
    axs[0].imshow(pitch_class_transition_matrix_pair[0])
    axs[0].set_title(name1)
    axs[1].set_xticks(np.arange(len(PITCH_CLASSES)), labels=PITCH_CLASSES)
    axs[1].set_yticks(np.arange(len(PITCH_CLASSES)), labels=PITCH_CLASSES)
    axs[1].imshow(pitch_class_transition_matrix_pair[1])
    axs[1].set_title(name2)
    fig.savefig(save_path)
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in multiple midi evaluation.')
    parser.add_argument(
        '-midi-path1', type=str,
        help='A directory of the midi files'
    )
    parser.add_argument(
        '-midi-path2', type=str,
        help='Another directory of the midi files'
    )
    parser.add_argument(
        '-out-dir', type=str, default="./results",
        help='The output directory to save metrics'
    )
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    _, midi_name1 = os.path.split(args.midi_path1)
    midi_name1 = os.path.splitext(midi_name1)[0]
    metrics1 = evaluate_single_midi(args.midi_path1, return_numpy=False)

    _, midi_name2 = os.path.split(args.midi_path2)
    midi_name2 = os.path.splitext(midi_name2)[0]
    metrics2 = evaluate_single_midi(args.midi_path2, return_numpy=False)

    metric_pairs = compare_single_midi_metrics(metrics1, metrics2)

    comparison_prefix = midi_name1 + '_vs_' + midi_name2
    out_json_filename = comparison_prefix + '_metrics.json'
    out_pctm_filename = comparison_prefix + '_pctm.pdf'
    out_pitch_hist_filename = comparison_prefix + '_pitch_hist.pdf'

    out_json_filepath = os.path.join(args.out_dir, out_json_filename)
    out_pctm_filepath = os.path.join(args.out_dir, out_pctm_filename)
    out_pitch_hist_filepath = os.path.join(args.out_dir, out_pitch_hist_filename)

    with open(out_json_filepath, "w") as outfile:
        json.dump(metric_pairs, outfile)

    plot_pitch_class_transition_matrix_pair(
        metric_pairs["pitch_class_transition_matrix"],
        out_pctm_filepath,
        names=(midi_name1, midi_name2)
    )
    plot_pitch_class_histogram_pair(
        metric_pairs["pitch_class_histogram"],
        out_pitch_hist_filepath,
        names=(midi_name1, midi_name2)
    )
    print("Saved total metrics to {}".format(args.out_dir))