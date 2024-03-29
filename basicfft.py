import librosa
import numpy as np
from numpy.fft import rfft, rfftfreq
import random
from math import log2, pow
import matplotlib.pyplot as plt
import soundfile as sf
from glob import glob
import scipy.io.wavfile as wavfile
import sys
import os

# See user manual for more details on these controls

# input controls
default_name = 'fft_test.wav' # located inside "soundfiles" folder
default_fps = 24
default_n_partials = 15

# Select type of output to write: 
create_csv = True # write as a csv with time on left axis
create_houdini_arrays = True # write as a series of arrays for each data field, separated by line
create_harmonics_list = True # write a list n_partials long of loudest harmonics in the entire sample

# Select data type for write:
write_frequencies = True # write frequency for harmonics
write_amplitudes = True # write amplitudes corresponding to harmonics
write_notes = True # write note values for selected harmonics

# Controls for data display
max_output_amplitude = 100 # value to scale maximum output amplitude to
transcribe_note = True # if true, write as note (CDEFGAB). if false, write as midi number value
round_midi = False # round midi values from decimal to integer if transcribed_note = False
filter_notes = True # only write the loudest harmonic detected for each individual midi note
exclude_below_threshold = True # exclude harmonics with amplitudes less than exclusion_threshold
exclusion_threshold = 20 # minimum amplitude value to include.

# constants for pitch detection
A4 = 440
C0 = A4*pow(2, -4.75)
note_vals = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def main(filename = default_name, fps = default_fps, n = default_n_partials):
    
    # load signal(s) from default audio directory 'soundfiles/'
    signal_lib, sample_rate_lib = load_audio_librosa(filename) # in stereo?
    # signal_scip, sample_rate_scip = load_audio_scipy('fft_test.wav')

    if(fps == "os"):
        times, frequencies, amplitudes = onset_fft_analysis(signal = signal_lib, sample_rate=sample_rate_lib, n=n)
        file_write(filename, True, 0, n, times, frequencies, amplitudes)
    else:
        # times, frequencies, amplitudes = librosa_fft_analysis( signal=signal_lib, sample_rate=sample_rate_lib, fps=int(fps), n=n)
        centroid_calculation(signal = signal_lib, sample_rate=sample_rate_lib, fps = int(fps))
        # print(librosa.feature.spectral_centroid(y= signal_lib, sr= sample_rate_lib))
        times, frequencies, amplitudes, max_read_amplitude = n_loudest_peaks( signal=signal_lib, sample_rate=sample_rate_lib, fps=int(fps), n=n)
        if filter_notes:
            filter_partials_by_note(times, frequencies, amplitudes)
        file_write(filename, False, int(fps), n, times, frequencies, amplitudes, max_read_amplitude)
        
        # filename, fps, n, times, frequencies, amplitudes 
    # fft loudest partial collections
    # # experimental specgram plotting ----------------------------
    # # mono signal
    # mono_lib = make_mono(signal_lib)
    
    # # plot testing
    # plot_values(mono_lib, sample_rate_lib)
    # # ------------------------------------------------------------

# load given file name from directory at default sample rate of 22050 using librosa
def load_audio_librosa(file_name, data_dir = 'soundfiles/', sr = 22050):
    file_name = data_dir + file_name
    print("File: %s" % file_name)
    signal, sample_rate = librosa.load(file_name, sr=sr)
    print("Librosa Sample Rate: %d" % sample_rate)
    return signal, sample_rate

# load given file name from directory at default sample rate of 22050 using librosa
def load_audio_scipy(file_name, data_dir = 'soundfiles/'):
    file_name = data_dir + file_name
    print("File: %s" % file_name)
    sample_rate, signal = wavfile.read(file_name)
    print("scipy Sample Rate: %d samples/sec" % sample_rate)
    return signal, sample_rate

# plot spectrogram of audio file using matplotlib
def plot_values(mono_signal, sample_rate):
    # create numpy linspace for plotting using sample rate and length of signal
    xaxis = np.linspace(0, len(mono_signal)/sample_rate, num=len(mono_signal))
    ax1 = plt.subplot(211)
    plt.plot(xaxis, mono_signal)
    plt.subplot(212, sharex = ax1)
    spec, f, t, i = plt.specgram(mono_signal, NFFT =256, Fs = sample_rate, noverlap=128, mode = 'psd', sides = 'default')
    plt.show()

# if signal is stereo, convert to mono
def make_mono(signal, channel = 0):
    if signal.ndim == 2:
        return signal[:,channel]
    else:
        print("Signal not stereo, has %d dimension " % signal.ndim)
        return signal

# converts a frequency value to midi note
# TODO uses class C4 centering for midi
def convert_to_pitch(frequency):
    
    # check if frequency is 0 (likely error)
    if(frequency != 0):
        if(transcribe_note):
            midi_num = round(12*log2(frequency/C0))
            octave = midi_num//12 # floor division for octave number
            note_index = midi_num % 12 
            return note_vals[note_index] + str(octave)
        else:
            if(round_midi):
                midi_num = round(12*log2(frequency/C0))
            else:
                midi_num = round(12*log2(frequency/C0), 2)
            return midi_num+12

    # write error character if frequency is 0
    else:
        return "!"


# write results to csv
def file_write(filename, os_mode, fps, n, times, frequencies, amplitudes, max_read_amplitude):
    if(create_csv):
        # split file extension and append fps/bucket info
        if(os_mode == False):
            out_name = filename.split('.')[0] + "_fps%d_n%d.csv" %(fps, n)
        else:
            out_name = filename.split('.')[0] + "_os_n%d.csv" % n
        file = open("outputfiles/%s" %out_name, "w")
        print("Writing to csv at outputfiles/%s"%out_name)

        # write csv headers
        i = 1
        file.write("time")
        while i < n + 1:
            if(write_frequencies):
                file.write(", frequency %d"% i) 
            if(write_notes):
                file.write(", note %d" % i)
            if(write_amplitudes):
                file.write(", amplitude %d" %i)
            i+=1
        file.write("\n")

        # write data
        i = 0
        while i < len(times):
            file.write("%.3f" % (times[i]))
            j = 0
            while j < len(frequencies[i]):
                if((exclude_below_threshold == False) or 
                    ((amplitudes[i][j]/max_read_amplitude)*max_output_amplitude) > exclusion_threshold):
                    if(write_frequencies):
                        file.write(", %.4f" %frequencies[i][j])
                    if(write_notes):
                        if(transcribe_note):
                            file.write(", %s" % convert_to_pitch(frequencies[i][j]))
                        else:
                            if(round_midi):
                                file.write(", %.0f" % convert_to_pitch(frequencies[i][j]))
                            else:
                                file.write(", %.2f" % convert_to_pitch(frequencies[i][j]))
                    if(write_amplitudes):
                        # file.write(", %.4f" %amplitudes[i][j]) # unscaled
                        file.write(", %.2f" %((amplitudes[i][j]/max_read_amplitude)*max_output_amplitude))
                j+=1
            i+=1
            file.write("\n")

        # close file
        file.close()
    if(create_houdini_arrays):
        out_name = filename.split('.')[0] + "_houdini_fps%d_n%d.txt" %(fps, n)
        file = open("outputfiles/%s"%out_name, "w")
        print("Writing to csv at outputfiles/%s"%out_name)
        file.write("times = {")
        # file.write("%.2f" % (times[0]))
        i = 0
        while i < len(times):
            j = 0
            while (j < len(frequencies[i])):
                if(i == 0 and j == 0):
                    file.write("%.2f" %times[i])
                else:
                    file.write(", %.2f" %times[i])
                j+=1
            i+=1
        file.write("};\n\n")
        file.write("midi = {")
        i = 0
        while i < len(frequencies):
            j = 0
            while (j < len(frequencies[i])):
                if(i == 0 and j == 0):
                    if(transcribe_note is False):
                        if(round_midi):
                            file.write("%.0f" %convert_to_pitch(frequencies[i][j]))
                        else:
                            file.write("%.2f" %convert_to_pitch(frequencies[i][j]))
                    else:
                        file.write("%s" %convert_to_pitch(frequencies[i][j]))
                else:
                    if(transcribe_note is False):
                        if(round_midi):
                            file.write(", %.0f" %convert_to_pitch(frequencies[i][j]))
                        else:
                            file.write(", %.2f" %convert_to_pitch(frequencies[i][j]))
                    else:
                        file.write(", %s" %convert_to_pitch(frequencies[i][j]))
                j+=1
            i+=1
        file.write("};\n\n")
        file.write("amplitudes = {")
        i = 0
        while i < len(times):
            j = 0
            while (j < len(amplitudes[i])):
                if(i == 0 and j == 0):
                    file.write("%.2f" %amplitudes[i][j])
                else:
                    file.write(", %.2f" %amplitudes[i][j])
                j+=1
            i+=1
        file.write("};\n")
        file.close()
    if(create_harmonics_list):
        out_name = filename.split('.')[0] + "_partial_list_n%d.txt" %(n)
        file = open("outputfiles/%s"%out_name, "w")
        print("Writing to csv at outputfiles/%s"%out_name)
        i = 0
        notes = []
        while(i < len(frequencies)):
            j = 0
            while(j < len(frequencies[i])):
                note_tuple = []
                note_tuple.append(convert_to_pitch(frequencies[i][j]))
                note_tuple.append(amplitudes[i][j])
                note_tuple.append(frequencies[i][j])
                found = False
                k = 0
                while(k < len(notes)):
                    if(notes[k][0] == note_tuple[0]):
                        # found = True
                        if(notes[k][1] < note_tuple[1]):
                            # elem[1] = note_tuple[1]
                            notes.pop(k)
                        else:
                            found = True
                    k+=1
                k = 0
                # while(k < len(notes) and found == False):
                while(k < len(notes) and found == False):
                    if(note_tuple[1] > notes[k][1]):
                        notes.insert(k, note_tuple)
                        found = True
                        if(len(notes) >= n):
                            notes.pop(len(notes)-1)
                    k+=1
                if (found == False and len(notes) < n):
                    notes.append(note_tuple)
                j+=1
            i+=1
        for note in notes:
            if(transcribe_note):
                file.write("\n%s" %note[0])
            if(write_amplitudes):
                file.write(", %.2f" %((note[1]/max_read_amplitude)*max_output_amplitude))
            if(write_frequencies):
                file.write(", %.2f" %note[2])
        file.close()
            
# acquires audio onset times using librosa and smallest interval
def get_onset_times(signal, sample_rate):
    o_env = librosa.onset.onset_strength(y=signal, sr = sample_rate)
    times = librosa.times_like(o_env, sr=sample_rate)
    onset_times = librosa.onset.onset_detect(onset_envelope = o_env, sr=sample_rate, units='time')
    
    return onset_times
    # # iterate over onset times and find the smallest gap between them for fft analysis
    # smallest_interval = 100.0
    # i = 0
    # while i < len(onset_times)-1:
    #     time_interval = onset_times[i+1] - onset_times[i]
    #     if time_interval < smallest_interval:
    #         smallest_interval = time_interval
    #     i += 1

    # return onset_times, smallest_interval

# performs similar fft analysis but with higher resolution, snapping to times determined by get_onset_times 
def onset_fft_analysis(signal, sample_rate, n_fft = 2048, n = 2):
    onset_times = get_onset_times(signal, sample_rate)
    print(onset_times)
    fps = 50
    times, frequencies, amplitudes = n_loudest_peaks(signal, sample_rate, n_fft, fps, n)  
    
    onset_frame_times = []
    onset_frequencies = []
    onset_amplitudes = []

    for onset_time in onset_times:
        i = 0
        while i < len(times)-1:
            if(onset_time <= times[i]):
                onset_frame_times.append(times[i])
                onset_frequencies.append(frequencies[i])
                onset_amplitudes.append(amplitudes[i])
                i = len(times)
            i+=1    

    return onset_frame_times, onset_frequencies, onset_amplitudes


def centroid_calculation(signal, sample_rate, n_fft=4096, fps=3, n=2):
    hop_length = int(sample_rate/fps)
    fft_set = np.abs(librosa.stft(signal, center=False, n_fft=n_fft, hop_length = hop_length))
    fft_set = np.abs(librosa.stft(signal, center=False, n_fft=n_fft, hop_length = hop_length))
    fft_freqs = librosa.fft_frequencies(sr = sample_rate, n_fft = n_fft)
    time_index = 0
    file = open("outputfiles/centroid_log.csv", "w")
    file.write("Time, Centroid\n")
    while time_index < len(fft_set[0])-1:
        
        # Sum of weighted amplitudes
        # Sum of amplitudes
        k = 0 
        j = 0
        max_amp = 0
        freq_index = 0
        while freq_index < (n_fft/2)-1:
            k += fft_set[freq_index, time_index]*fft_freqs[freq_index]     
            j += fft_set[freq_index, time_index]
            freq_index += 1
            if max_amp < fft_set[freq_index, time_index]:
                max_amp = fft_set[freq_index, time_index]
        centroid = k/j
        # print("Time: %.4f centroid: %.6f" %(time_index*hop_length/sample_rate, centroid))
        if max_amp > 1:
            # print("Time: %.6f centroid: %.6f" %(time_index*hop_length/sample_rate, centroid))
            # file.write("Time: %")
            file.write("%.6f, %.6f\n" %(time_index*hop_length/sample_rate, centroid))
        else:
            file.write("\n")
        time_index +=1

def n_loudest_peaks(signal, sample_rate, n_fft=4096, fps=3, n=2):
    hop_length = int(sample_rate/fps)
    fft_set = np.abs(librosa.stft(signal, center=False, n_fft=n_fft, hop_length = hop_length))
    # fft_set = np.abs(librosa.stft(signal, center=False, n_fft=n_fft, hop_length = hop_length))
    time_index = 0 
    frequencies = []
    amplitudes = []
    times = []
    max_read_amplitude = 0
    # Reference array containing corresponding frequency values relative to bin indexes
    fft_freqs = librosa.fft_frequencies(sr = sample_rate, n_fft = n_fft)
    # print(fft_set.shape)

    while time_index < len(fft_set[0]):
    
        # Grab peak bin indexes ----------------------------------------------------------
        # parse vertically through amplitude at frequency bins at time time_index
        freq_index = 1
        # if(time_index == 4):
        peak_indexes = []
        

        while (freq_index < (n_fft/2)):
            if(fft_set[freq_index, time_index] > fft_set[freq_index-1, time_index]):
                if (fft_set[freq_index, time_index] > fft_set[freq_index+1, time_index]):
                    if(fft_set[freq_index, time_index] > 1):
                        peak_indexes.append(freq_index)
                        # print("time %.4f index %d freq %.6f amp %.6f" %(time_index*hop_length/sample_rate, freq_index, fft_freqs[freq_index], fft_set[freq_index, time_index]))
            freq_index += 1
        # print("Number Peaks: %d" %len(peak_indexes))
        # -------------------------------------------------------------------------

        # Set non peak bins to 0 --------------------------------------------------
        freq_index = 0
        while(freq_index < (n_fft/2)+1):
            is_peak = False
            for index in peak_indexes:
                if freq_index == index:
                    is_peak = True
            if is_peak == False:
                fft_set[freq_index, time_index] = 0
            freq_index += 1

        # -------------------------------------------------------------------------

        # Grab loudest of peak bins -----------------------------------------------
        if(len(peak_indexes) < n):
            n_loudest_indexes = np.argsort(-fft_set[:,time_index])[:len(peak_indexes)]
        else:    
            n_loudest_indexes = np.argsort(-fft_set[:,time_index])[:n]
        n_loudest_amplitudes = fft_set[n_loudest_indexes, time_index]
        # if(n_loudest_amplitudes[0] > max_read_amplitude):
        #     max_read_amplitude = n_loudest_amplitudes[0]
        
        frequency_array = fft_freqs[n_loudest_indexes]
        frequencies.append(frequency_array)
        amplitudes.append(n_loudest_amplitudes)
        times.append((time_index*hop_length/sample_rate))
        # -------------------------------------------------------------------------

        time_index+=1
    for amp in amplitudes:
        if(len(amp) > 0):
            if(amp[0] > max_read_amplitude):
                max_read_amplitude = amp[0]   
    return times, frequencies, amplitudes, max_read_amplitude


# acquire n loudest partials from sample using librosa fft
# Parameters:
#   signal <- librosa audio file 
#   int sample rate <- sample rate of signal 
#   int n_fft <- number of frequency bins to analyze per frame
#   int fps <- number of frames to collect data at per second
#   int n <- number of loudest partials to collect  
# Returns:
#   1D array times <- times belonging to data capture
#   2D array frequencies <- n loudest frequencies for len(time) times
#   2D array amplitudes <- corresponding n loudest amplitudes for len(time) times
def librosa_fft_analysis(signal, sample_rate, n_fft=4096, fps=3, n=2):
    
    # TODO is the fft exclusive or inclusive time hop?
    # distance in samples between each fft analysis
    hop_length = int(sample_rate/fps) 
    
    # perform librosa short-time fourier transform on signal 
    fft_set = np.abs(librosa.stft(signal, center=False, n_fft=n_fft, hop_length = hop_length))
    
    # print relevant information on analysis
    fft_set = np.abs(librosa.stft(signal, center=False, n_fft=n_fft, hop_length = hop_length))
    # print(fft_set.shape)

    # time index for parsing horizonatally
    time_index = 0 

    # list of n loudest frequencies  
    frequencies = []
    amplitudes = []
    times = []
    # peak_array = [][]
    # EXAMPLE STFT ARRAY:
        # ff = final frame = song_len {in samples} / hop_length = len(fft_set[0])
        # bin frequency = determine with librosa.fft_frequencies
        #
        # time(index*hop_len/sr)        :     0       1       2    ...,  ff   
        # 
        # amplitude at freq bin 0       : [amp0_0, amp0_1, amp0_2, ..., amp0_ff]
        # amplitude at freq bin 1       : [amp1_0, amp1_1, amp1_2, ..., amp1_ff]
        # amplitude at freq bin ...     : [...]
        # amplitude at freq bin n_fft/2 : [ampn_0, ampn_1, ampn_2, ..., ampn_ff]
        #
        # tldr: horizontal = time, vertical = frequency 

    # reference with bin index for frequency val
    fft_freqs = librosa.fft_frequencies(sr = sample_rate, n_fft = n_fft)

    # parse through time horizontally and get loudest frequencies at each time
    while(time_index < len(fft_set[0])):

        
        # Grab loudest bins -------------------------------------------------------
        # obtain sorted array of n bin indexes by amplitude value at time index
        n_loudest_indexes = np.argsort(-fft_set[:,time_index])[:n]
        # sorted array of n loudest amplitudes at time index 
        n_loudest_amplitudes = fft_set[n_loudest_indexes, time_index]
        
        # # check if loudest partial meets the amplitude threshold and print
        # if(n_loudest_amplitudes[0] > 1):
        #     print("Time %.3f" %(time_index*hop_length/sample_rate) )
        #     print(n_loudest_indexes)
        #     print(n_loudest_amplitudes)
        
        # collect all data into larger dataset
        frequency_array = fft_freqs[n_loudest_indexes]
        frequencies.append(frequency_array)
        amplitudes.append(n_loudest_amplitudes)
        times.append((time_index*hop_length/sample_rate))
        # -------------------------------------------------------------------------

        # Increment time index horizontally
        time_index += 1
    
    # # print in order
    # i = 0
    # while i < len(times):
    #     print(times[i])
    #     print(frequencies[i])
    #     print(amplitudes[i])
    #     i += 1
    
    # # print as lists
    # print(amplitudes)    
    # print(frequencies)
    # print(times)
    return times, frequencies, amplitudes 


def filter_partials_by_note(times, frequencies, amplitudes):
    i = 0
    while i < len(times):
        notes = []
        j = 0
        while j < len(frequencies[i]):
            notes.append(convert_to_pitch(frequencies[i][j]))
            k = 0
            while k < j:
                if(transcribe_note):
                    if(notes[k] == notes[j]):
                        # print("deleting duplicate note %s at i: %d j: %d\n"%(notes[k], i, j))
                        # print(frequencies[i])
                        # print(amplitudes[i])
                        # print(notes)
                        frequencies[i] = np.delete(frequencies[i], j, 0)
                        amplitudes[i] = np.delete(amplitudes[i], j, 0)
                        # frequencies[i].pop(j)

                        # print(frequencies[i])
                        # print(amplitudes[i])
                        notes.pop(j)
                        # print(notes)                    
                        j = j - 1
                    # elif(write_amplitudes):
                else:
                    if(round(notes[k]) == round(notes[j])):
                        frequencies[i] = np.delete(frequencies[i], j, 0)
                        amplitudes[i] = np.delete(amplitudes[i], j, 0)
                        notes.pop(j)
                        j = j - 1
                k+=1
            j+=1
        i+=1
    

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(sys.argv[1]) 
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2]) 
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]))     
    else:
        print("Usage: basicfft.py <filename> <frames per second> <n loudest partials>")