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
count = 0
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


def frequency_spectrum(sample, max_frequency=800):
    """
    Derive frequency spectrum of a signal pydub.AudioSample
    Returns an array of frequencies and an array of how prevelant that frequency is in the sample
    """
    # Convert pydub.AudioSample to raw audio data
    # Copied from Jiaaro's answer on https://stackoverflow.com/questions/32373996/pydub-raw-audio-data
    bit_depth = sample.sample_width * 8
    array_type = get_array_type(bit_depth)
    raw_audio_data = array.array(array_type, sample._data)
    n = len(raw_audio_data)

    # Compute FFT and frequency value for each index in FFT array
    # Inspired by Reveille's answer on https://stackoverflow.com/questions/53308674/audio-frequencies-in-python
    freq_array = np.arange(n) * (float(sample.frame_rate) / n)  # two sides frequency range
    freq_array = freq_array[: (n // 2)]  # one side frequency range

    raw_audio_data = raw_audio_data - np.average(raw_audio_data)  # zero-centering
    freq_magnitude = scipy.fft.fft(raw_audio_data)  # fft computing and normalization
    freq_magnitude = freq_magnitude[: (n // 2)]  # one side

    if max_frequency:
        max_index = int(max_frequency * n / sample.frame_rate) + 1
        freq_array = freq_array[:max_index]
        freq_magnitude = freq_magnitude[:max_index]

    freq_magnitude = abs(freq_magnitude)
    freq_magnitude = freq_magnitude / np.sum(freq_magnitude)
    return freq_array, freq_magnitude


def classify_note_attempt_1(freq_array, freq_magnitude):
    i = np.argmax(freq_magnitude)
    f = freq_array[i]
    print("frequency {}".format(f))
    print("magnitude {}".format(freq_magnitude[i]))
    return get_note_for_freq(f)


def classify_note_attempt_2(freq_array, freq_magnitude):
    note_counter = Counter()
    for i in range(len(freq_magnitude)):
        if freq_magnitude[i] < 0.01:
            continue
        note = get_note_for_freq(freq_array[i])
        if note:
            note_counter[note] += freq_magnitude[i]
    return note_counter.most_common(1)[0][0]


def classify_note_attempt_3(freq_array, freq_magnitude):
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
            note = get_note_for_freq(freq)
            if note:
                note_counter[note] += freq_magnitude[i] * credit_multiplier

    return note_counter.most_common(1)[0][0]


# If f is within tolerance of a note (measured in cents - 1/100th of a semitone)
# return that note, otherwise returns None
# We scale to the 440 octave to check
def get_note_for_freq(f, tolerance=33):
    # Calculate the range for each note
    tolerance_multiplier = 2 ** (tolerance / 1200)
    note_ranges = {
        k: (v / tolerance_multiplier, v * tolerance_multiplier) for (k, v) in NOTES.items()
    }

    # Get the frequence into the 440 octave
    range_min = note_ranges["A"][0]
    range_max = note_ranges["G#"][1]
    if f < range_min:
        while f < range_min:
            f *= 2
    else:
        while f > range_max:
            f /= 2

    # Check if any notes match
    for (note, note_range) in note_ranges.items():
        if f > note_range[0] and f < note_range[1]:
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

    starts = predict_note_starts(song, plot_starts, actual_starts)

    predicted_notes = predict_notes(song, starts, actual_notes, plot_fft_indices)

    print("")
    if actual_notes:
        print("Actual Notes")
        print(actual_notes)
    print("Predicted Notes")
    print(predicted_notes)

    if actual_notes:
        lev_distance = calculate_distance(predicted_notes, actual_notes)
        print("Levenshtein distance: {}/{}".format(lev_distance, len(actual_notes)))


# Very simple implementation, just requires a minimum volume and looks for left edges by
# comparing with the prior sample, also requires a minimum distance between starts
# Future improvements could include smoothing and/or comparing multiple samples
#
# song: pydub.AudioSegment
# plot: bool, whether to show a plot of start times
# actual_starts: []float, time into song of each actual note start (seconds)
#
# Returns perdicted starts in ms
def predict_note_starts(song, plot, actual_starts):
    # Size of segments to break song into for volume calculations
    SEGMENT_MS = 50
    # Minimum volume necessary to be considered a note
    VOLUME_THRESHOLD = -35
    # The increase from one sample to the next required to be considered a note
    EDGE_THRESHOLD = 5
    # Throw out any additional notes found in this window
    MIN_MS_BETWEEN = 100

    # Filter out lower frequencies to reduce noise
    song = song.high_pass_filter(80, order=4)
    # dBFS is decibels relative to the maximum possible loudness
    volume = [segment.dBFS for segment in song[::SEGMENT_MS]]

    predicted_starts = []
    for i in range(1, len(volume)):
        if volume[i] > VOLUME_THRESHOLD and volume[i] - volume[i - 1] > EDGE_THRESHOLD:
            ms = i * SEGMENT_MS
            # Ignore any too close together
            if len(predicted_starts) == 0 or ms - predicted_starts[-1] >= MIN_MS_BETWEEN:
                predicted_starts.append(ms)

    # If actual note start times are provided print a comparison
    if len(actual_starts) > 0:
        print("Approximate actual note start times ({})".format(len(actual_starts)))
        print(" ".join(["{:5.2f}".format(s) for s in actual_starts]))
        print("Predicted note start times ({})".format(len(predicted_starts)))
        print(" ".join(["{:5.2f}".format(ms / 1000) for ms in predicted_starts]))

    # Plot the volume over time (sec)
    x_axis = np.arange(len(volume)) * (SEGMENT_MS / 1000)
    graph_plotter(x_axis, volume, "dBFS vs Time", "Time(in seconds)", "dBFS(Decibels relative to full scale)")
    if plot:
        x_axis = np.arange(len(volume)) * (SEGMENT_MS / 1000)
        plt.plot(x_axis, volume)

        # Add vertical lines for predicted note starts and actual note starts
        for s in actual_starts:
            plt.axvline(x=s, color="r", linewidth=0.5, linestyle="-")
        for ms in predicted_starts:
            plt.axvline(x=(ms / 1000), color="g", linewidth=0.5, linestyle=":")

        plt.show()

    return predicted_starts

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
    
def predict_notes(song, starts, actual_notes, plot_fft_indices):
    predicted_notes = []
    for i, start in enumerate(starts):
        sample_from = start + 50
        sample_to = start + 550
        if i < len(starts) - 1:
            sample_to = min(starts[i + 1], sample_to)
        segment = song[sample_from:sample_to]
        freqs, freq_magnitudes = frequency_spectrum(segment)

        predicted = classify_note_attempt_3(freqs, freq_magnitudes)
        predicted_notes.append(predicted or "U")

        # Print general info
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
        plot_fft_indices = [0, 4, 5, 19]
        if i in plot_fft_indices:
            graph_plotter(freqs, freq_magnitudes, "Magnitude of the frequency response", "Frequency(in Hertz)", "|X(omega)|")
    return predicted_notes
from tkinter import filedialog
from tkinter import *
import tkinter as tk

root = Tk()

def myClick():

    file_path = filedialog.askopenfilename()
    main(file_path)


text2 = tk.Text(root, height=10, width=79)
scroll = tk.Scrollbar(root, command=text2.yview)
text2.configure(yscrollcommand=scroll.set)
text2.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
text2.tag_configure('big', font=('Verdana', 20, 'bold'))
text2.tag_configure('color',
                    foreground='#90EE90',
                    font=('Tempus Sans ITC', 12, 'bold'))

text2.insert(tk.END,'\nWelcome to note recognition with python\n', 'big')
text2.pack()
myButton = Button(root, text="Click to upload MP3 file", command = myClick, fg = "blue", bg="#90EE90", font = "sans 16 bold", padx = 20, pady = 20)
myButton.pack()

root.mainloop()

"""if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--note-file", type=str)
    parser.add_argument("--note-starts-file", type=str)
    parser.add_argument("--plot-starts", action="store_true")
    parser.add_argument(
        "--plot-fft-index",
        type=int,
        nargs="*",
        help="Index of detected note to plot graph of FFT for",
    )
    args = parser.parse_args()"""
"""main(
        args.file,
        note_file=args.note_file,
        note_starts_file=args.note_starts_file,
        plot_starts=args.plot_starts,
        plot_fft_indices=(args.plot_fft_index or []),
    )"""
