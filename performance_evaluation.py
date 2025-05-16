import music21 as m21
import numpy as np
import pandas as pd
from collections import Counter
import math
import csv
import itertools

from train_A1a import ABL
from melody_generator import OUTPUTS_PATH

"""
local_metrics = {
        "Pitch Variance": {"value": float(pitch_variance(segments)), "comment": "Higher is better for more pitch variation."},
        "Pitch Range": {"value": float(pitch_range(segments)), "comment": "Higher is better for a wider range of pitches."},
        "Rhythmic Variance": {"value": float(rhythmic_variance(segments)), "comment": "Higher is better for more varied rhythms."},
        "Note Density": {"value": float(note_density(segments)), "comment": "Higher is better for a denser melody."},
        "Rest Ratio": {"value": float(rest_ratio(segments)), "comment": "Lower is better for continuous note flow; higher for contrast."},
        "Interval Variability": {"value": float(interval_variability(segments)), "comment": "Higher is better for more complex intervals."},
        "Note Repetition": {"value": float(note_repetition(segments)), "comment": "Lower is better to avoid monotony."},
        "Contour Stability": {"value": float(contour_stability(segments)), "comment": "Lower is better for smoother contour; higher for expressiveness."},
        "Syncopation": {"value": float(syncopation(segments)), "comment": "Higher is better for more offbeat rhythms."},
        "Harmonic Tension": {"value": float(harmonic_tension(segments)), "comment": "Lower is better for stability; higher for drama."},
        "KL Divergence": {"value": float(kl_divergence(segments, s)), "comment": "Lower is better for consistent pitch distribution across segments."}
    }

    global_metrics = {
        "Pitch Entropy": {"value": float(pitch_entropy(s)), "comment": "Higher is better for diverse pitch usage."},
        "Rhythmic Entropy": {"value": float(rhythmic_entropy(s)), "comment": "Higher is better for varied rhythms."},
        "Melodic Interval Distribution": {"value": melodic_interval_distribution(s), "comment": "More uniform distribution is better for variety."},
        "Motif Diversity Index": {"value": float(motif_diversity_index(s)), "comment": "Higher is better for more diverse motifs."},
        "Harmonic Complexity": {"value": float(harmonic_complexity(s)), "comment": "Higher is better for complex harmony."},
        "Tempo Variability": {"value": float(tempo_variability(s)), "comment": "Lower is better for coherence; higher for expressiveness. (Not computed as data is missing)"},
        "Contour Variability": {"value": float(contour_variability(s)), "comment": "Higher is better for varied contour."},
        "Phrase Length Variability": {"value": float(phrase_length_variability(s)), "comment": "Higher is better for varied phrase lengths."},
        "Tonal Drift": {"value": float(tonal_drift(s)), "comment": "Higher is better for more key changes."},
        "Global KL Divergence": {"value": float(global_kl_divergence(s)), "comment": "Lower is better for similar sections; higher for contrast."}
    }
"""

MELODY_PATH = OUTPUTS_PATH
EVAL_PATH = 'melodies/' + ABL + '/eval_ ' + ABL + '.csv'


# Parse the input string into a music21 stream
def parse_melody(input_string):
    s = m21.stream.Stream()
    tokens = input_string.split()
    offset = 0
    i = 0
    while i < len(tokens):
        if tokens[i] != '_':
            value = tokens[i]
            duration = 0.25  # Base duration for the note/rest itself
            i += 1
            # Count subsequent underscores for additional duration
            while i < len(tokens) and tokens[i] == '_':
                duration += 0.25
                i += 1
            if value == 'r':
                s.insert(offset, m21.note.Rest(quarterLength=duration))
            else:
                s.insert(offset, m21.note.Note(midi=int(value), quarterLength=duration))
            offset += duration
        else:
            i += 1
    return s


# Segment the stream into parts for local metrics
def segment_stream(s, segment_length=4):  # 4 quarter lengths = 1 measure in 4/4
    segments = []
    current_segment = m21.stream.Stream()
    current_length = 0
    for element in s:
        if isinstance(element, (m21.note.Note, m21.note.Rest)):
            if current_length + element.quarterLength <= segment_length:
                current_segment.append(element)
                current_length += element.quarterLength
            else:
                remaining = segment_length - current_length
                if remaining > 0:
                    if isinstance(element, m21.note.Note):
                        partial = m21.note.Note(midi=element.pitch.midi, quarterLength=remaining)
                    else:  # Rest
                        partial = m21.note.Rest(quarterLength=remaining)
                    current_segment.append(partial)
                segments.append(current_segment)
                current_segment = m21.stream.Stream()
                current_length = element.quarterLength - remaining
                if current_length > 0:
                    if isinstance(element, m21.note.Note):
                        partial = m21.note.Note(midi=element.pitch.midi, quarterLength=current_length)
                    else:
                        partial = m21.note.Rest(quarterLength=current_length)
                    current_segment.append(partial)
                else:
                    current_segment.append(element)
                    current_length = element.quarterLength
    if current_segment:
        segments.append(current_segment)
    return segments


# Local Metrics
def pitch_variance(segments):
    variances = []
    for segment in segments:
        pitches = [n.pitch.midi for n in segment.notes if isinstance(n, m21.note.Note)]
        if pitches:
            variances.append(np.var(pitches))
    return np.mean(variances) if variances else 0


def pitch_range(segments):
    ranges = []
    for segment in segments:
        pitches = [n.pitch.midi for n in segment.notes if isinstance(n, m21.note.Note)]
        if pitches:
            ranges.append(max(pitches) - min(pitches))
    return np.mean(ranges) if ranges else 0


def rhythmic_variance(segments):
    variances = []
    for segment in segments:
        durations = [e.quarterLength for e in segment.elements]
        if durations:
            variances.append(np.var(durations))
    return np.mean(variances) if variances else 0


def note_density(segments):
    densities = []
    for segment in segments:
        total_duration = sum(e.quarterLength for e in segment.elements)
        num_notes = len(segment.notes)
        densities.append(num_notes / total_duration if total_duration else 0)
    return np.mean(densities) if densities else 0


def rest_ratio(segments):
    ratios = []
    for segment in segments:
        total_duration = sum(e.quarterLength for e in segment.elements)
        rest_duration = sum(r.quarterLength for r in segment.getElementsByClass('Rest'))
        ratios.append(rest_duration / total_duration if total_duration else 0)
    return np.mean(ratios) if ratios else 0


def interval_variability(segments):
    variances = []
    for segment in segments:
        pitches = [n.pitch.midi for n in segment.notes if isinstance(n, m21.note.Note)]
        if len(pitches) > 1:
            intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
            variances.append(np.var(intervals))
    return np.mean(variances) if variances else 0


def note_repetition(segments):
    reps = []
    for segment in segments:
        pitches = [n.pitch.midi for n in segment.notes if isinstance(n, m21.note.Note)]
        if pitches:
            max_reps = max([sum(1 for _ in group) for _, group in itertools.groupby(pitches)], default=0)
            reps.append(max_reps)
    return np.mean(reps) if reps else 0


def contour_stability(segments):
    changes = []
    for segment in segments:
        pitches = [n.pitch.midi for n in segment.notes if isinstance(n, m21.note.Note)]
        if len(pitches) > 1:
            direction_changes = 0
            for i in range(1, len(pitches) - 1):
                prev_dir = pitches[i] - pitches[i - 1]
                next_dir = pitches[i + 1] - pitches[i]
                if prev_dir * next_dir < 0:  # Opposite directions
                    direction_changes += 1
            changes.append(direction_changes)
    return np.mean(changes) if changes else 0


def syncopation(segments):
    syncs = []
    for segment in segments:
        sync_count = 0
        for n in segment.notes:
            if isinstance(n, m21.note.Note) and n.offset % 1 != 0:  # Off-beat
                sync_count += 1
        syncs.append(sync_count)
    return np.mean(syncs) if syncs else 0


def harmonic_tension(segments):
    tensions = []
    tension_map = {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2, 5: 0.3, 6: 0.5, 7: 0.1, 8: 0.2}  # Simplified tension values
    for segment in segments:
        pitches = [n.pitch.midi for n in segment.notes if isinstance(n, m21.note.Note)]
        if len(pitches) > 1:
            intervals = [abs(pitches[i + 1] - pitches[i]) % 12 for i in range(len(pitches) - 1)]
            segment_tension = [tension_map.get(min(i, 12 - i), 0.1) for i in intervals]
            tensions.append(np.mean(segment_tension))
    return np.mean(tensions) if tensions else 0


def kl_divergence(segments, s):
    global_dist = Counter([n.pitch.midi for n in s.notes if isinstance(n, m21.note.Note)])
    total_global = sum(global_dist.values())
    global_dist = {k: v / total_global for k, v in global_dist.items()}
    divergences = []
    for segment in segments:
        local_dist = Counter([n.pitch.midi for n in segment.notes if isinstance(n, m21.note.Note)])
        total_local = sum(local_dist.values())
        if total_local == 0:
            continue
        local_dist = {k: v / total_local for k, v in local_dist.items()}
        kl = 0
        for pitch in set(global_dist.keys()).union(local_dist.keys()):
            p_global = global_dist.get(pitch, 1e-10)  # Avoid log(0)
            p_local = local_dist.get(pitch, 1e-10)
            kl += p_local * math.log(p_local / p_global)
        divergences.append(kl)
    return np.mean(divergences) if divergences else 0


# Global Metrics
def pitch_entropy(s):
    pitches = [n.pitch.midi for n in s.notes if isinstance(n, m21.note.Note)]
    if not pitches:
        return 0
    count = Counter(pitches)
    total = len(pitches)
    return -sum((freq / total) * math.log2(freq / total) for freq in count.values())


def rhythmic_entropy(s):
    durations = [n.quarterLength for n in s.notes if isinstance(n, m21.note.Note)]
    if not durations:
        return 0
    count = Counter(durations)
    total = len(durations)
    return -sum((freq / total) * math.log2(freq / total) for freq in count.values())


def melodic_interval_distribution(s):
    pitches = [n.pitch.midi for n in s.notes if isinstance(n, m21.note.Note)]
    if len(pitches) < 2:
        return {}
    intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
    count = Counter(intervals)
    total = len(intervals)
    return {str(interval): (count[interval] / total) * 100 for interval in count}


def motif_diversity_index(s):
    pitches = [n.pitch.midi for n in s.notes if isinstance(n, m21.note.Note)]
    if len(pitches) < 3:
        return 0
    motifs = set()
    for i in range(len(pitches) - 2):
        motif = tuple(pitches[i:i + 3])  # Consider motifs of length 3
        motifs.add(motif)
    total_possible = len(pitches) - 2
    return len(motifs) / total_possible if total_possible else 0


def harmonic_complexity(s):
    pitches = [n.pitch.midi for n in s.notes if isinstance(n, m21.note.Note)]
    if len(pitches) < 2:
        return 0
    intervals = [abs(pitches[i + 1] - pitches[i]) % 12 for i in range(len(pitches) - 1)]
    unique_intervals = len(set(intervals))
    transition_variance = np.var([intervals[i + 1] - intervals[i] for i in range(len(intervals) - 1)]) if len(
        intervals) > 1 else 0
    return (unique_intervals / 12 + transition_variance / 10) / 2  # Normalized score 0-1


def tempo_variability(s):
    # No tempo info in input, assume constant tempo (e.g., 120 BPM)
    return 0  # Would require tempo markings or playback data to compute


def contour_variability(s):
    pitches = [n.pitch.midi for n in s.notes if isinstance(n, m21.note.Note)]
    if len(pitches) < 2:
        return 0
    changes = 0
    for i in range(1, len(pitches) - 1):
        prev_dir = pitches[i] - pitches[i - 1]
        next_dir = pitches[i + 1] - pitches[i]
        if prev_dir * next_dir < 0:
            changes += 1
    return changes / (len(pitches) - 1) * 100 if len(pitches) > 1 else 0


def phrase_length_variability(s):
    phrases = []
    current_length = 0
    for e in s.elements:
        if isinstance(e, m21.note.Rest) and current_length > 0:
            phrases.append(current_length)
            current_length = 0
        elif isinstance(e, (m21.note.Note, m21.note.Rest)):
            current_length += e.quarterLength
    if current_length > 0:
        phrases.append(current_length)
    return np.var(phrases) if phrases else 0


def tonal_drift(s):
    # Add measures to enable key analysis per measure
    s.makeMeasures(inPlace=True)
    keys = []
    for measure in s.getElementsByClass('Measure'):
        if not measure.notes:
            continue
        k = measure.analyze('key')
        keys.append(k.tonic.name + ' ' + k.mode)
    return len(set(keys)) - 1 if keys else 0  # Number of key changes


def global_kl_divergence(s):
    pitches = [n.pitch.midi for n in s.notes if isinstance(n, m21.note.Note)]
    durations = [n.quarterLength for n in s.notes if isinstance(n, m21.note.Note)]
    if len(pitches) < 10:
        return 0
    split_point = len(pitches) // 2
    pitch_dist1 = Counter(pitches[:split_point])
    pitch_dist2 = Counter(pitches[split_point:])
    rhythm_dist1 = Counter(durations[:split_point])
    rhythm_dist2 = Counter(durations[split_point:])

    total1, total2 = sum(pitch_dist1.values()), sum(pitch_dist2.values())
    pitch_kl = 0
    for pitch in set(pitch_dist1.keys()).union(pitch_dist2.keys()):
        p1 = pitch_dist1.get(pitch, 1e-10) / total1
        p2 = pitch_dist2.get(pitch, 1e-10) / total2
        pitch_kl += p1 * math.log(p1 / p2) if p1 > 1e-10 and p2 > 1e-10 else 0

    total1, total2 = sum(rhythm_dist1.values()), sum(rhythm_dist2.values())
    rhythm_kl = 0
    for dur in set(rhythm_dist1.keys()).union(rhythm_dist2.keys()):
        r1 = rhythm_dist1.get(dur, 1e-10) / total1
        r2 = rhythm_dist2.get(dur, 1e-10) / total2
        rhythm_kl += r1 * math.log(r1 / r2) if r1 > 1e-10 and r2 > 1e-10 else 0

    return (pitch_kl + rhythm_kl) / 2


def compute_metrics(input_string):
    s = parse_melody(input_string)
    segments = segment_stream(s)

    local_metrics = {
        "Pitch Variance": float(pitch_variance(segments)),
        "Pitch Range": float(pitch_range(segments)),
        "Rhythmic Variance": float(rhythmic_variance(segments)),
        "Note Density": float(note_density(segments)),
        "Rest Ratio": float(rest_ratio(segments)),
        "Interval Variability": float(interval_variability(segments)),
        "Note Repetition": float(note_repetition(segments)),
        "Contour Stability": float(contour_stability(segments)),
        "Syncopation": float(syncopation(segments)),
        "Harmonic Tension": float(harmonic_tension(segments)),
        "KL Divergence": float(kl_divergence(segments, s))
    }

    global_metrics = {
        "Pitch Entropy": float(pitch_entropy(s)),
        "Rhythmic Entropy": float(rhythmic_entropy(s)),
        "Motif Diversity Index": float(motif_diversity_index(s)),
        "Harmonic Complexity": float(harmonic_complexity(s)),
        "Contour Variability": float(contour_variability(s)),
        "Tonal Drift": float(tonal_drift(s))
    }

    # Exclude Melodic Interval Distribution since it's a dictionary
    return {**local_metrics, **global_metrics}


# Process melodies and write to CSV
def process_melodies_to_csv(output_csv=EVAL_PATH):
    # Split the input text by 'r' to get individual melodies
    with open(MELODY_PATH, 'r') as file:
        melodies = [line.strip() for line in file.readlines() if line.strip()]

        # Define CSV headers
        headers = ["Melody Number"] + [
            "Pitch Variance", "Pitch Range", "Rhythmic Variance", "Note Density", "Rest Ratio",
            "Interval Variability", "Note Repetition", "Contour Stability", "Syncopation",
            "Harmonic Tension", "KL Divergence",  # Local metrics
            "Pitch Entropy", "Rhythmic Entropy", "Motif Diversity Index", "Harmonic Complexity", "Contour Variability",
            "Tonal Drift"  # Global metrics (excluding Melodic Interval Distribution)
        ]

        # Open CSV file for writing
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()

            # Process each melody
            for i, melody in enumerate(melodies, 1):
                try:
                    metrics = compute_metrics(melody)
                    row = {"Melody Number": f"Melody{i}"}
                    row.update(metrics)
                    writer.writerow(row)
                    print(f"Processed melody {i}/{len(melodies)}.")
                except Exception as e:
                    print(f"Error processing Melody{i}/{len(melodies)}: {e}")

        return f"Metrics saved to {output_csv}"


def add_statistics():
    df = pd.read_csv(EVAL_PATH)
    numeric_cols = df.columns[1:]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    summary_data = []

    # SUM row
    sum_row = ['SUM'] + df[numeric_cols].sum().tolist()
    summary_data.append(sum_row)

    # MEAN row
    mean_row = ['MEAN'] + df[numeric_cols].mean().tolist()
    summary_data.append(mean_row)

    # MIN row
    min_row = ['MIN'] + df[numeric_cols].min().tolist()
    summary_data.append(min_row)

    # MAX row
    max_row = ['MAX'] + df[numeric_cols].max().tolist()
    summary_data.append(max_row)

    # MEDIAN row
    median_row = ['MEDIAN'] + df[numeric_cols].median().tolist()
    summary_data.append(median_row)

    # STANDARD DEVIATION row
    std_row = ['STD'] + df[numeric_cols].std().tolist()
    summary_data.append(std_row)

    # COUNT of non-null values
    count_row = ['COUNT'] + df[numeric_cols].count().tolist()
    summary_data.append(count_row)

    summary_df = pd.DataFrame(summary_data, columns=df.columns)
    final_df = pd.concat([df, summary_df], ignore_index=True)
    final_df.to_csv(EVAL_PATH, index=False)

    print("Last 7 rows of the resulting DataFrame:")
    print(final_df.tail(7))

    print("\nDataFrame Info:")
    print(f"Total rows: {len(final_df)}")
    print(f"Total columns: {len(final_df.columns)}")


if __name__ == '__main__':
    result = process_melodies_to_csv()
    print(result)
    add_statistics()