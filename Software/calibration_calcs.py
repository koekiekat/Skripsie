from itertools import combinations
import json

from background_functions import read_audio_file
from stft import short_time_calc
from dtw import dtw_calc  # adjust this import to match wherever dtw_calc actually lives


def _load_template_segment(template, fs_new=1000):
    """
    Given one saved template entry ({index, call_type, start_time, end_time,
    wav_path}), reload its audio from disk and compute its STFT the same
    way the rest of the pipeline does.
    """
    f_s, x = read_audio_file(template["wav_path"])
    start_sample = int(template["start_time"] * f_s)
    end_sample = int(template["end_time"] * f_s)
    segment = x[start_sample:end_sample]
    _, _, Zxx, _ = short_time_calc(segment, f_s, fs_new)
    return Zxx


def compute_template_dtw_costs(templates_path, n_templates, fs_new=1000):
    """
    Load a saved templates JSON file, take the first `n_templates` entries,
    and compute the DTW cost of every template against every other template
    (all unique pairs, no self-comparisons, no duplicate reverse pairs since
    dtw_calc is symmetric).

    Returns:
        costs: 1D list of DTW costs, one per pair
        pairs: list of (i, j) index pairs (positions within the first
               n_templates entries) matching each entry in `costs`
    """
    with open(templates_path) as f:
        templates = json.load(f)

    if len(templates) < n_templates:
        print(f"Warning: file only has {len(templates)} template(s), "
              f"using all of them instead of {n_templates}.")
    templates = templates[:n_templates]

    # Precompute each template's spectrogram once, rather than recomputing
    # it every time it appears in a pair.
    spectrograms = [_load_template_segment(t, fs_new) for t in templates]

    costs = []
    pairs = []
    for i, j in combinations(range(len(templates)), 2):
        cost = dtw_calc(spectrograms[i], spectrograms[j])
        costs.append(cost)
        pairs.append((i, j))

    return costs, pairs