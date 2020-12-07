import argparse

from pydub import AudioSegment
import pydub.scipy_effects
import numpy as np
import scipy
import matplotlib.pyplot as plt

import array
from collections import Counter
from pydub.utils import get_array_type
from Levenshtein import distance
# Creating a variable count to keep track of the graphs
count = 0
#dictionary having key value pairs of notes and their frequencies
NOTES = { 
    "A": 440,
    "A#": 466.1637615180899,
    "B": 493.8833012561241,
    "C": 523.2511306011972,
    "C#": 554.3652619537442,
    "D": 587.3295358348151,
    "D#": 622.2539674441618,
    "E": 659.2551138257398,
    "F": 698.4564628660078,
    "F#": 739.9888454232688,
    "G": 783.9908719634985,
    "G#": 830.6093951598903,
}


def frequency_func(sample, max_frequency=800):
    """
    This function takes input of maximum frequency and the AudioSample
    Returns arrays of frequencies how prevelant that frequency is in the sample
    """
    # Convert the pydub.AudioSample to raw numeric audio data
    bit_depth = sample.sample_width * 8
    array_type = get_array_type(bit_depth)
    numeric_audio_data = array.array(array_type, sample._data)
    n = len(numeric_audio_data)
    # Compute FFT and frequency value for each index in FFT array
    # two sides frequency range
    freq_array = np.arange(n) * (float(sample.frame_rate) / n) 
    # one side frequency range
    freq_array = freq_array[: (n // 2)]
    # zero-centering
    numeric_audio_data = numeric_audio_data - np.average(numeric_audio_data)
    # fft computing and normalization
    freq_magnitude = scipy.fft.fft(numeric_audio_data)  
    freq_magnitude = freq_magnitude[: (n // 2)] 
    if max_frequency:
        max_index = int(max_frequency * n / sample.frame_rate) + 1
        freq_array = freq_array[:max_index]
        freq_magnitude = freq_magnitude[:max_index]

    freq_magnitude = abs(freq_magnitude)
    freq_magnitude = freq_magnitude / np.sum(freq_magnitude)
    return freq_array, freq_magnitude


# def classify_note_attempt_1(freq_array, freq_magnitude):
#     i = np.argmax(freq_magnitude)
#     f = freq_array[i]
#     print("frequency {}".format(f))
#     print("magnitude {}".format(freq_magnitude[i]))
#     return freq_to_note(f)


# def classify_note_attempt_2(freq_array, freq_magnitude):
#     note_counter = Counter()
#     for i in range(len(freq_magnitude)):
#         if freq_magnitude[i] < 0.01:
#             continue
#         note = freq_to_note(freq_array[i])
#         if note:
#             note_counter[note] += freq_magnitude[i]
#     return note_counter.most_common(1)[0][0]


def note_classifier(freq_array, freq_magnitude):
    min_freq = 82
    note_counter = Counter()
    for i in range(len(freq_magnitude)):
        if freq_magnitude[i] < 0.01:
            continue

        for freq_multiplier, credit_multiplier in [
            (1, 1),
            (1 / 3, 3 / 4),
            (1 / 5, 1 / 2),
            (1 / 6, 1 / 2),
            (1 / 7, 1 / 2),
        ]:
            freq = freq_array[i] * freq_multiplier
            if freq < min_freq:
                continue
            note = freq_to_note(freq)
            if note:    
                note_counter[note] += freq_magnitude[i] * credit_multiplier
    print(note_counter)
    if note_counter != None:
        return note_counter.most_common(1)[0][0]

def freq_to_note(current_freq, tolerance=33):
    '''
        In this function we check whether the given frequency is within tolerance range of a note.
        Tolerance is measured in cents (1/100th of a semitone).
        If current_freq is within the tolerance range of a note, we return that note
        else we scale that note into the 440 octave. 
        If it is still not in the tolerance range of any note, we return None.
    '''
    # Calculating the tolerance range for each note
    tolerance_level = 2 ** (tolerance / 1200) 
    note_range = {
        k: (v / tolerance_level, v * tolerance_level) for (k, v) in NOTES.items()
    }

    # Getting the frequency into the 440 octave
    range_min = note_range["A"][0]
    range_max = note_range["G#"][1]
    if current_freq < range_min:
        while current_freq < range_min:
            current_freq *= 2
    else:
        while current_freq > range_max:
            current_freq /= 2

    # Checking if any notes match
    for (note, note_range) in note_range.items():
        if current_freq > note_range[0] and current_freq < note_range[1]:
            return note
    return None


# Assumes everything is either natural or sharp, no flats
# Returns the Levenshtein distance between the actual notes and the predicted notes
def calculate_distance(predicted, actual):
    # To make a simple string for distance calculations we make natural notes lower case
    # and sharp notes cap
    def transform(note):
        if "#" in note:
            return note[0].upper()
        return note.lower()

    return distance(
        "".join([transform(n) for n in predicted]), "".join([transform(n) for n in actual]),
    )


def main(file, note_file=None, note_starts_file=None, plot_starts=False, plot_fft_indices=[]):
    # If a note file and/or actual start times are supplied read them in
    actual_starts = []
    if note_starts_file:
        with open(note_starts_file) as f:
            for line in f:
                actual_starts.append(float(line.strip()))

    actual_notes = []
    if note_file:
        with open(note_file) as f:
            for line in f:
                actual_notes.append(line.strip())

    song = AudioSegment.from_file(file)
    song = song.high_pass_filter(80, order=4)
    note_starts = note_start_detector(song, plot_starts, actual_starts)

    predicted_notes = note_predicter(song, note_starts, actual_notes, plot_fft_indices)

    print("")
    if actual_notes:
        print("Actual Notes")
        print(actual_notes)
    print("Predicted Notes")
    print(predicted_notes)

    if actual_notes:
        lev_distance = calculate_distance(predicted_notes, actual_notes)
        print("Levenshtein distance: {}/{}".format(lev_distance, len(actual_notes)))

    
def note_start_detector(song, plot, actual_starts):
      ''' 
    The logic used here is: 
        Whenever a note is played, there is a sudden increase in volume. 
    So, to find out the starting times of all the notes, we set an edge threshold = 5 (edge_thresh in the program).
    We also set a  volume threshold = -35 so that no points having dBFS less than that can be considered to be starting points of notes
    Hence,  
        if the increase in the volume is more than 5, it will be considered to be a note.  

    song: pydub.AudioSegment
    plot: bool, whether to show a plot of start times
    actual_starts: []float, time into song of each actual note start (seconds)

    Returns a list of predicted start timings - note_start_times in miliseconds
    '''
    # Size of segments to break song into for volume calculations
    segment_ms = 50
    # Minimum volume necessary to be considered a note
    vol_thresh = -35
    # The increase from one sample to the next required to be considered a note
    edge_thresh = 5
    # Throw out any additional notes found in this window. This is the minimum distance between two notes. 
    min_ms_between = 100

    # Filter out lower frequencies to reduce noise
    song = song.high_pass_filter(80, order=4)
    # dBFS is decibels relative to the maximum possible loudness (or decibels relative to Full Scale)
    # The line below uses standard array slicing to break the song into segments
    volume = [segment.dBFS for segment in song[::segment_ms]]

    note_start_times = []
    # checking for volume and edge threshold parameters
    for i in range(1, len(volume)):
        if volume[i] > vol_thresh and volume[i] - volume[i - 1] > edge_thresh:
            ms = i * segment_ms
            # Ignore any if they are too close together
            # Checking for min ms condition
            if len(note_start_times) == 0 or ms - note_start_times[-1] >= min_ms_between:
                note_start_times.append(ms)

    # If actual note start times are provided print a comparison
    if len(actual_starts) > 0:
        print("Approximate actual note start times ({})".format(len(actual_starts)))
        print(" ".join(["{:5.2f}".format(s) for s in actual_starts]))
        print("Predicted note start times ({})".format(len(note_start_times)))
        print(" ".join(["{:5.2f}".format(ms / 1000) for ms in note_start_times]))

    
    # Plot the volume over time (sec)
    x_axis = np.arange(len(volume)) * (segment_ms / 1000)
    graph_plotter(x_axis, volume, "dBFS vs Time", "Time(in seconds)", "dBFS(Decibels relative to full scale)")
    if plot:
        plt.plot(x_axis, volume, "dBFS vs Time", "Time(in seconds)", "dBFS(Decibels relative to full scale)")

        # Add vertical lines for predicted note note_starts and actual note note_starts
        for s in actual_starts:
            plt.axvline(x=s, color="r", linewidth=0.5, linestyle="-")
        for ms in note_start_times:
            plt.axvline(x=(ms / 1000), color="g", linewidth=0.5, linestyle=":")

        plt.show()
    return note_start_times

def graph_plotter(x_axis, y_axis, title, xlabel, ylabel, text=-0.8, pos=-1):
    figure = plt.figure(figsize=(20,15))
    axes = figure.add_subplot()
    axes.plot(x_axis, y_axis, linewidth=2.5, color='blue')
    axes.tick_params(which='minor', length=3, color='black')
    axes.tick_params(which='major', length=5) 
    axes.tick_params(which='both', width=2) 
    axes.tick_params(labelcolor='black', labelsize=15, width=3.5)
    plt.ylabel(ylabel, {'fontsize': 21, 'color': 'y'})
    plt.xlabel(xlabel,  {'fontsize': 21, 'color': 'y'})
    plt.grid()
    global count
    plt.title(title, {'color': 'y', 'fontsize': 45})
    plt.savefig(f'graph{count}.png')
    count += 1

def note_predicter(song, note_starts, actual_notes, plot_fft_indices):
    predicted_notes = []
    for i, start in enumerate(note_starts):
        sample_from = start + 50
        sample_to = start + 550
        if i < len(note_starts) - 1:
            sample_to = min(note_starts[i + 1], sample_to)
        segment = song[sample_from:sample_to]
        freqs, freq_magnitudes = frequency_func(segment)

        predicted = note_classifier(freqs, freq_magnitudes)
        predicted_notes.append(predicted or "U") # U stands for Unknown

        # Printing all the general information
        print("")
        print("Note: {}".format(i))
        if i < len(actual_notes):
            print("Predicted: {} Actual: {}".format(predicted, actual_notes[i]))
        else:
            print("Predicted: {}".format(predicted))
        print("Predicted start: {}".format(start))
        length = sample_to - sample_from
        print("Sampled from {} to {} ({} ms)".format(sample_from, sample_to, length))
        print("Frequency sample period: {}hz".format(freqs[1]))

        # Print peak info
        peak_indicies, props = scipy.signal.find_peaks(freq_magnitudes, height=0.015)
        print("Peaks of more than 1.5 percent of total frequency contribution:")
        for j, peak in enumerate(peak_indicies):
            freq = freqs[peak]
            magnitude = props["peak_heights"][j]
            print("{:.1f}hz with magnitude {:.3f}".format(freq, magnitude))

        if i in plot_fft_indices:
            
            """plt.plot(freqs, freq_magnitudes, "b")
            plt.xlabel("Freq (Hz)")
            plt.ylabel("|X(freq)|")
            plt.show()"""
        graph_plotter(freqs, freq_magnitudes, "Magnitude of the frequency response", "Frequency(in Hertz)", "|X(omega)|")
    return predicted_notes


if _name_ == "_main_":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--note-file", type=str)
    parser.add_argument("--note-note_starts-file", type=str)
    parser.add_argument("--plot-note_starts", action="store_true")
    parser.add_argument(
        "--plot-fft-index",
        type=int,
        nargs="*",
        help="Index of detected note to plot graph of FFT for",
    )
    args = parser.parse_args()
    main(
        args.file,
        note_file=args.note_file,
        note_starts_file=args.note_starts_file,
        plot_starts=args.plot_starts,
        plot_fft_indices=(args.plot_fft_index or []),
    )
