import numpy as np
from itertools import groupby
from scipy.stats import entropy


def sliding_windows(sequence, window_size):
    tokens = sequence.split()
    return [tokens[i:i + window_size] for i in range(len(tokens) - window_size + 1)]


def pitch_variance_local(sequence, window_size=8):
    """
    Calculates the variance of pitch values within a short window.

    Reason: Measures how much pitch fluctuates within local segments, indicating melodic diversity.
    Unit: Variance (higher values indicate greater pitch dispersion).
    High Value: Indicates more pitch variation, potentially richer melodies.
    Low Value: Suggests a monotonous or stepwise melody.
    """
    windows = sliding_windows(sequence, window_size)
    variances = []
    for window in windows:
        pitches = [int(x) for x in window if x.isdigit()]
        if len(pitches) > 1:
            variances.append(np.var(pitches))
    return np.mean(variances) if variances else 0


def pitch_range_local(sequence, window_size=8):
    """
    Measures the pitch range (max - min) in small segments.

    Reason: Determines the span of notes within local sections, reflecting expressiveness.
    Unit: Pitch interval (difference between max and min pitch values).
    High Value: Suggests a dynamic and expressive melody.
    Low Value: Indicates narrow-range melodies, possibly monotonous.
    """
    windows = sliding_windows(sequence, window_size)
    ranges = [max(map(int, w)) - min(map(int, w)) for w in windows if any(x.isdigit() for x in w)]
    return np.mean(ranges) if ranges else 0


def rhythmic_variance_local(sequence, window_size=8):
    """
    Computes variance of rhythm patterns in a segment.

    Reason: Captures rhythmic diversity in local phrases.
    Unit: Variance of note duration counts.
    High Value: Suggests varied and complex rhythms.
    Low Value: Indicates repetitive or predictable rhythms.
    """
    windows = sliding_windows(sequence, window_size)
    variances = []
    for window in windows:
        durations = [len(list(g)) for k, g in groupby(window) if k != '_']
        if durations:
            variances.append(np.var(durations))
    return np.mean(variances) if variances else 0


def note_density_local(sequence, window_size=8):
    """
    Computes the number of notes per unit segment.

    Reason: Measures how densely packed a melody is within short spans.
    Unit: Ratio of notes per window.
    High Value: Indicates more frequent note changes, suggesting busier melodies.
    Low Value: Suggests sparser melodic lines with more rests or sustained notes.
    """
    windows = sliding_windows(sequence, window_size)
    densities = [len([x for x in w if x != '_']) / len(w) for w in windows]
    return np.mean(densities)


def rest_ratio_local(sequence, window_size=8):
    """
    Computes the proportion of rests in a segment.

    Reason: Reflects pauses and silence in the melody.
    Unit: Ratio of rests to total tokens in a window.
    High Value: Suggests a melody with more silence or breaks.
    Low Value: Indicates continuous note flow with minimal rests.
    """
    windows = sliding_windows(sequence, window_size)
    ratios = [w.count('r') / len(w) for w in windows]
    return np.mean(ratios)


def interval_variability_local(sequence, window_size=8):
    """
    Calculates variability of melodic intervals within a short segment.

    Reason: Measures how much interval jumps change within a local window.
    Unit: Variance of interval sizes.
    High Value: Suggests unpredictable, complex melodies.
    Low Value: Indicates stepwise motion or minimal interval changes.
    """
    windows = sliding_windows(sequence, window_size)
    variances = []
    for window in windows:
        pitches = [int(x) for x in window if x.isdigit()]
        if len(pitches) > 1:
            intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
            variances.append(np.var(intervals))
    return np.mean(variances) if variances else 0


def note_repetition_local(sequence, window_size=8):
    """
    Computes how often the same note is repeated in local segments.

    Reason: Captures tendencies for note repetition within a phrase.
    Unit: Count of consecutive note repetitions.
    High Value: Indicates repetitive phrasing.
    Low Value: Suggests more varied note usage.
    """
    windows = sliding_windows(sequence, window_size)
    repetitions = [sum(1 for i in range(len(w) - 1) if w[i] == w[i + 1]) for w in windows]
    return np.mean(repetitions)


def contour_stability_local(sequence, window_size=8):
    """
    Measures how often the melodic contour changes direction.

    Reason: Reflects how frequently a melody moves up or down within segments.
    Unit: Count of contour direction changes.
    High Value: Indicates unstable, erratic movement.
    Low Value: Suggests smooth, predictable contour flow.
    """
    windows = sliding_windows(sequence, window_size)
    changes = []
    for window in windows:
        pitches = [int(x) for x in window if x.isdigit()]
        if len(pitches) > 1:
            directions = [np.sign(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
            changes.append(sum(1 for i in range(len(directions) - 1) if directions[i] != directions[i + 1]))
    return np.mean(changes) if changes else 0


def syncopation_local(sequence, window_size=8):
    """
    Computes a measure of syncopation in small segments.

    Reason: Identifies offbeat notes and unexpected rhythmic placements.
    Unit: Count of syncopated instances.
    High Value: Suggests more offbeat and unpredictable rhythms.
    Low Value: Indicates onbeat and predictable rhythms.
    """
    windows = sliding_windows(sequence, window_size)
    syncopations = [sum(1 for i in range(1, len(w)) if w[i] != 'r' and w[i - 1] == 'r') for w in windows]
    return np.mean(syncopations)


def harmonic_tension_local(sequence, window_size=8):
    """
    Estimates local harmonic tension based on interval size.

    Reason: Evaluates how tense or stable melodies feel in short spans.
    Unit: Average interval size per segment.
    High Value: Indicates more harmonic tension (large intervals, dissonances).
    Low Value: Suggests more stable, consonant melodies.
    """
    windows = sliding_windows(sequence, window_size)
    tensions = []
    for window in windows:
        pitches = [int(x) for x in window if x.isdigit()]
        if len(pitches) > 1:
            intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
            tensions.append(np.mean(intervals))
    return np.mean(tensions) if tensions else 0


def kl_divergence_local(sequence, window_size=8):
    """
    Measures the divergence of the note distribution in local segments compared to the overall note distribution in the sequence.

    Reason: KL divergence helps quantify how different a local segment's note distribution is from the overall sequence, indicating variation in pitch usage.

    Unit: Unitless (log scale)

    High Value: Greater deviation in note distributions (more varied segments).
    Low Value: More uniform note distribution across segments.
    """
    tokens = sequence.split()
    overall_distribution, _ = np.histogram([int(x) for x in tokens if x.isdigit()], bins=range(46), density=True)
    windows = sliding_windows(sequence, window_size)
    divergences = []
    for window in windows:
        local_pitches = [int(x) for x in window if x.isdigit()]
        if local_pitches:
            local_distribution, _ = np.histogram(local_pitches, bins=range(46), density=True)
            divergences.append(entropy(local_distribution + 1e-10, overall_distribution + 1e-10))
    return np.mean(divergences) if divergences else 0