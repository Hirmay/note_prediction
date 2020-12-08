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
from PIL import Image

count = [0]
counter = 0
simp = open("Simple.txt", "w")
det = open("Note.txt","w")
comp = open("Compare.txt", "w")
NOTES = { #dictionary to map notes to their corresponding frequencies
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
Color_Notes = {
    "A": "red",
    "A#": "coral",
    "B": "brown",
    "C": "darkorange",
    "C#": "orange",
    "D": "lawngreen",
    "D#": "gold",
    "E": "slategray",
    "F": "darkgreen",
    "F#": "lime",
    "G": "navy",
    "G#": "skyblue",
    "U": "black"
}

def frequency_func(sample, max_frequency=800):
    """
    Derive frequency spectrum of a signal pydub.AudioSample
    Returns an array of frequencies and an array of how prevelant that frequency is in the sample
    """
    # Convert pydub.AudioSample to raw audio data
    # Copied from Jiaaro's answer on https://stackoverflow.com/questions/32373996/pydub-raw-audio-data
    bit_depth = sample.sample_width * 8
    array_type = get_array_type(bit_depth)
    numeric_audio_data = array.array(array_type, sample._data)
    n = len(numeric_audio_data)

    # Compute FFT and frequency value for each index in FFT array
    # Inspired by Reveille's answer on https://stackoverflow.com/questions/53308674/audio-frequencies-in-python
    freq_array = np.arange(n) * (float(sample.frame_rate) / n)  # two sides frequency range
    freq_array = freq_array[: (n // 2)]  # one side frequency range

    numeric_audio_data = numeric_audio_data - np.average(numeric_audio_data)  # zero-centering
    freq_magnitude = scipy.fft.fft(numeric_audio_data)  # fft computing and normalization
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
    det.write("frequency {}\n".format(f))
    print("magnitude {}".format(freq_magnitude[i]))
    det.write("magnitude {}\n".format(freq_magnitude[i]))
    return freq_to_note(f)


def classify_note_attempt_2(freq_array, freq_magnitude):
    note_counter = Counter()
    for i in range(len(freq_magnitude)):
        if freq_magnitude[i] < 0.01:
            continue
        note = freq_to_note(freq_array[i])
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
            note = freq_to_note(freq)
            if note:
                note_counter[note] += freq_magnitude[i] * credit_multiplier

    return note_counter.most_common(1)[0][0]


# If f is within tolerance of a note (measured in cents - 1/100th of a semitone)
# return that note, otherwise returns None
# We scale to the 440 octave to check
def freq_to_note(current_freq, tolerance=33):
    # Calculate the range for each note
    tolerance_multiplier = 2 ** (tolerance / 1200)
    note_ranges = {
        k: (v / tolerance_multiplier, v * tolerance_multiplier) for (k, v) in NOTES.items()
    }

    # Get the frequence into the 440 octave
    range_min = note_ranges["A"][0]
    range_max = note_ranges["G#"][1]
    if current_freq < range_min:
        while current_freq < range_min:
            current_freq *= 2
    else:
        while current_freq > range_max:
            current_freq /= 2

    # Check if any notes match
    for (note, note_range) in note_ranges.items():
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


actual_starts = []    
def main(file, note_file=None, note_starts_file=None, plot_starts=False, plot_fft_indices=[]):
    # If a note file and/or actual start times are supplied read them in
    global actual_starts
    if note_starts_file:
        with open(note_starts_file) as f:
            for line in f:
                actual_starts.append(line.strip())
    actual_notes = []
    if note_file:
        with open(note_file) as f:
            for line in f:
                actual_notes.append(line.strip())

    song = AudioSegment.from_file(file)
    song = song.high_pass_filter(80, order=4)

    starts = note_start_detector(song, plot_starts, actual_starts)

    predicted_notes = predict_notes(song, starts, actual_notes, plot_fft_indices)

    print("")
    det.write("\n")
    if actual_notes:
        print("Actual Notes")
        print(actual_notes)
    if len(actual_starts) > 0:
        difference = calculate_distance(predicted_notes, actual_starts)
        print("Actual Notes ({})".format(len(actual_starts)))
        det.write("Actual Notes ({})\n".format(len(actual_starts)))
        print(actual_starts)
        det.write(f"{actual_starts}\n")
        print("Predicted Notes ({})".format(len(predicted_notes)))
        det.write("Predicted Notes ({})\n".format(len(predicted_notes)))
        print(predicted_notes)
        det.write(f"{predicted_notes}\n")
        print("Difference")
        print(difference)
        det.write("Difference\n")
        det.write(f"{difference}\n")
        
    #print("Predicted Notes")
    #det.write("Predicted Notes\n")
    simp.write("Predicted Notes\n")
    #print(predicted_notes)
    #det.write(f"{predicted_notes}\n")
    simp.write(f"{predicted_notes}\n")
    global x_axis, volume, count
    count[0] = "Main"
    graph_plotter(x_axis, volume, "dBFS vs Time", "Time(in seconds)", "dBFS(Decibels relative to full scale)", predicted_notes, actual_starts, starts)
    if actual_notes:
        lev_distance = calculate_distance(predicted_notes, actual_notes)
        print("Levenshtein distance: {}/{}".format(lev_distance, len(actual_notes)))
        det.write("Levenshtein distance: {}/{}\n".format(lev_distance, len(actual_notes)))
    det.close()
    simp.close()
# Very simple implementation, just requires a minimum volume and looks for left edges by
# comparing with the prior sample, also requires a minimum distance between starts
# Future improvements could include smoothing and/or comparing multiple samples
#
# song: pydub.AudioSegment
# plot: bool, whether to show a plot of start times
# actual_starts: []float, time into song of each actual note start (seconds)
#
# Returns perdicted starts in ms
def note_start_detector(song, plot, actual_starts):
    # Size of segments to break song into for volume calculations
    segment_ms = 50
    # Minimum volume necessary to be considered a note
    vol_thresh = -35
    # The increase from one sample to the next required to be considered a note
    edge_thresh = 5
    # Throw out any additional notes found in this window
    min_ms_between = 100

    # Filter out lower frequencies to reduce noise
    song = song.high_pass_filter(80, order=4)
    # dBFS is decibels relative to the maximum possible loudness
    global volume
    volume = [segment.dBFS for segment in song[::segment_ms]]

    note_start_times = []
    for i in range(1, len(volume)):
        if volume[i] > vol_thresh and volume[i] - volume[i - 1] > edge_thresh:
            ms = i * segment_ms
            # Ignore any too close together
            if len(note_start_times) == 0 or ms - note_start_times[-1] >= min_ms_between:
                note_start_times.append(ms)


    # Plot the volume over time (sec)
    global x_axis
    x_axis = np.arange(len(volume)) * (segment_ms / 1000)
    if plot:
        x_axis = np.arange(len(volume)) * (segment_ms / 1000)
        plt.plot(x_axis, volume)

        # Add vertical lines for predicted note starts and actual note starts
        for s in actual_starts:
            plt.axvline(x=s, color="r", linewidth=0.5, linestyle="-")
        for ms in note_start_times:
            plt.axvline(x=(ms / 1000), color="g", linewidth=0.5, linestyle=":")

        plt.show()

    return note_start_times

def graph_plotter(x_axis, y_axis, title, xlabel, ylabel, list_notes=[], actual_starts=[], note_start_times =[], text=-0.8, pos=-1):
    global Color_Notes
    figure = plt.figure(figsize=(20,15))
    axes = figure.add_subplot()
    axes.plot(x_axis, y_axis, linewidth=2.5, color='blue')
    if len(actual_starts)!= 0 or len(note_start_times)!= 0:
        new_pred_starts = [x / 1000 for x in note_start_times]
        labels = list(set(list_notes))
        j = 0
        for i, ms in enumerate(new_pred_starts):
            if j < len(labels):
                for i in range(len(list_notes)):
                    if j < len(labels):
                        if list_notes[i] == labels[j]:
                            plt.axvline(ms, color=Color_Notes[list_notes[i]], linewidth=2, linestyle='--',     label=f"Note {labels[j]}")
                            del labels[j]
                            continue
                else:
                    plt.axvline(ms, color=Color_Notes[list_notes[i]], linewidth=2, linestyle='--')
            else:
                plt.axvline(ms, color=Color_Notes[list_notes[i]], linewidth=2, linestyle='--')
    axes.tick_params(which='minor', length=3, color='black')
    axes.tick_params(which='major', length=5) 
    axes.tick_params(which='both', width=2) 
    axes.tick_params(labelcolor='black', labelsize=15, width=3.5)
    if len(list_notes) != 0:
        axes.legend(prop={'size': 18})
    plt.ylabel(ylabel, {'fontsize': 21, 'color': 'y'})
    plt.xlabel(xlabel,  {'fontsize': 21, 'color': 'y'})
    global count
    plt.title(title, {'color': 'y', 'fontsize': 45})
    plt.savefig(f'Graph {count[0]}.png')
    
def predict_notes(song, starts, actual_notes, plot_fft_indices):
    global predicted_notes
    predicted_notes = []
    for i, start in enumerate(starts):
        sample_from = start + 50
        sample_to = start + 550
        if i < len(starts) - 1:
            sample_to = min(starts[i + 1], sample_to)
        segment = song[sample_from:sample_to]
        freqs, freq_magnitudes = frequency_func(segment)

        predicted = classify_note_attempt_3(freqs, freq_magnitudes)
        predicted_notes.append(predicted or "U")

        # Print general info
        print("")
        det.write("\n")
        print("Note: {}".format(i))
        det.write("Note: {}\n".format(i))
        if i < len(actual_notes):
            print("Predicted: {} Actual: {}".format(predicted, actual_notes[i]))
            det.write("Predicted: {} Actual: {}\n".format(predicted, actual_notes[i]))
        else:
            print("Predicted: {}".format(predicted))
            det.write("Predicted: {}\n".format(predicted))
        print("Predicted start: {}".format(start))
        det.write("Predicted start: {}\n".format(start))
        length = sample_to - sample_from
        print("Sampled from {} to {} ({} ms)".format(sample_from, sample_to, length))
        det.write("Sampled from {} to {} ({} ms)\n".format(sample_from, sample_to, length))
        print("Frequency sample period: {}hz".format(freqs[1]))
        det.write("Frequency sample period: {}hz\n".format(freqs[1]))
        # Print peak info
        peak_indicies, props = scipy.signal.find_peaks(freq_magnitudes, height=0.015)
        print("Peaks of more than 1.5 percent of total frequency contribution:")
        det.write("Peaks of more than 1.5 percent of total frequency contribution:\n")
        for j, peak in enumerate(peak_indicies):
            freq = freqs[peak]
            magnitude = props["peak_heights"][j]
            print("{:.1f}hz with magnitude {:.3f}".format(freq, magnitude))
            det.write("{:.1f}hz with magnitude {:.3f}\n".format(freq, magnitude))
        plot_fft_indices = [4]
        global count
        if i in plot_fft_indices:
            graph_plotter(freqs, freq_magnitudes, "Magnitude of the frequency response", "Frequency(in Hertz)", "|X(omega)|")
            count[0] += 1
    return predicted_notes

from tkinter import filedialog
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk

root = Tk()

text11 = "hey there" 



def openNewWindow():
    main(file_path_m4a, None, file_path_text)
    newWindow = Toplevel(root)
    newWindow.title("Select the result")
    newWindow.geometry("1720x972")
    text3 = tk.Text(newWindow, height=10, width=40)
    scroll = tk.Scrollbar(root, command=text2.yview)
    text3.configure(yscrollcommand=scroll.set)
    text3.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
    text3.tag_configure('big', font=('Verdana', 20, 'bold'))
    text3.tag_configure('color',
                        foreground='#90EE90',
                        font=('Tempus Sans ITC', 12, 'bold'))

    text3.insert(tk.END,'\nSelect Output Result\n', 'big')
    text3.pack()
    
    def showText():
        global f, Str, counter, text4
        if counter != 0:
            text4.delete("1.0", END)
        else:
            text4 = tk.Text(newWindow, height=30, width=120)
            text4.delete("1.0", "end")
            scroll = tk.Scrollbar(root, command=text2.yview)
            text4.configure(yscrollcommand=scroll.set)
            text4.tag_configure('color',
                                foreground='#90EE90',
                                font=('Tempus Sans ITC', 8, 'bold'))
        f = open("Simple.txt", "r")
        Str = f.read()
        text4.insert(tk.END, Str, 'big')
        text4.pack(side = 'right')
        counter += 1
    def detailedText():
        global f, Str, counter, text4
        if counter != 0:
            text4.delete("1.0", END)
        else:
            text4 = tk.Text(newWindow, height=30, width=120)
            text4.delete("1.0", "end")
            scroll = tk.Scrollbar(root, command=text2.yview)
            text4.configure(yscrollcommand=scroll.set)
            text4.tag_configure('color',
                                foreground='#90EE90',
                                font=('Tempus Sans ITC', 8, 'bold'))
        f = open("Note.txt", "r")
        Str = f.read()
        text4.insert(tk.END, Str, 'big')
        text4.pack(side = 'right')
        counter += 1
    def showCompare():
        global f, Str, counter, text4
        if counter != 0:
            text4.delete("1.0", END)
        else:
            text4 = tk.Text(newWindow, height=30, width=120)
            text4.delete("1.0", "end")
            scroll = tk.Scrollbar(root, command=text2.yview)
            text4.configure(yscrollcommand=scroll.set)
            text4.tag_configure('color',
                                foreground='#90EE90',
                                font=('Tempus Sans ITC', 8, 'bold'))
        f = open("Compare.txt", "r")
        Str = f.read()
        text4.insert(tk.END, Str, 'big')
        text4.pack(side = 'right')
        counter += 1
    def showImage():

        # creating a object  
        im = Image.open(r"C:\Users\Admin_2020\Desktop\Note_Prediction\Graph Main.png")  

        im.show()

    def Compare():
        global file_path_text, predicted_notes, actual_starts
        actual_notes1 = actual_starts
        difference = calculate_distance(predicted_notes, actual_notes1)
        comp.write("Predicted Notes\n")                             
        comp.write(f"{predicted_notes}\n")                             
        comp.write("Actual Notes\n")                             
        comp.write(f"{actual_notes1}\n")
        comp.write("Difference\n")
        comp.write(f"{difference}\n")
        comp.close()
        showCompare()
        
    simpleButton = Button(newWindow, text="Simple Output", command = showText, fg = "blue", bg="#90EE90", font = "sans 12 ", padx = 50, pady = 20, bd = 5).place(x = 10, y = 200)

    detailedButton = Button(newWindow, text="Detailed Output", command = detailedText, fg = "blue", bg="#90EE90", font = "sans 12 ", padx = 45, pady = 20, bd = 5).place(x = 10, y = 300)
    
    volumeButton = Button(newWindow, text="Volume vs Time Graph", command = showImage, fg = "blue", bg="#90EE90", font = "sans 12 ", padx = 20, pady = 20, bd = 5).place(x = 10, y = 400)

    compareButton = Button(newWindow, text="Comparison", command = Compare, fg = "blue", bg="#90EE90", font = "sans 12 ", padx = 55, pady = 20, bd = 5).place(x = 10, y = 500)

    newWindow.mainloop()

file_path_m4a = ""
file_path_text = ""

def myClick_m4a():
    global file_path_m4a
    file_path_m4a = filedialog.askopenfilename()
    
    
def myClick_text():
    global file_path_text
    file_path_text = filedialog.askopenfilename()

def openWindow():

    openNewWindow()


root.geometry("650x500")
text2 = tk.Text(root, height=10, width=80)
scroll = tk.Scrollbar(root, command=text2.yview)
text2.configure(yscrollcommand=scroll.set)
text2.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
text2.tag_configure('big', font=('Verdana', 20, 'bold'))
text2.tag_configure('color',
                    foreground='#90EE90',
                    font=('Tempus Sans ITC', 12, 'bold'))

text2.insert(tk.END,'\nWelcome to note recognition with python\n', 'big')
text2.pack()
from PIL import Image
from PIL import ImageTk
width = 200
height = 150
img = Image.open("musical_note.png")
img = img.resize((width,height), Image.ANTIALIAS)
photoImg =  ImageTk.PhotoImage(img)
img1 = Image.open("text_upload.png")
img1 = img1.resize((width-50,height), Image.ANTIALIAS)
photoImg1 =  ImageTk.PhotoImage(img1)
uploadm4a = Button(root, image= photoImg, command = myClick_m4a, fg = "blue", bg="#90EE90", font = "sans 12 ", padx = 20, pady = 20).place (x = 35, y = 200)

uploadText = Button(root, image=photoImg1, command = myClick_text, fg = "blue", bg="#90EE90", font = "sans 12 ", padx = 20, pady = 20).place (x = 400, y = 200)

myButton3 = Button(root, text="Submit", command = openWindow, fg = "blue", bg="#90EE90", font = "sans 14 ", padx = 20, pady = 20).place(x = 270, y = 400)


root.mainloop()

