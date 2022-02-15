import librosa
import numpy as np
from numpy.fft import rfft, rfftfreq
import random
import matplotlib.pyplot as plt
import soundfile as sf
from glob import glob
import scipy.io.wavfile as wavfile
import sys

def main(filename = 'fft_test.wav'):
    signal_lib, sample_rate_lib = load_audio_librosa(filename)
    # signal_scip, sample_rate_scip = load_audio_scipy('fft_test.wav')

    # mono signals
    # mono_lib = make_mono(signal_lib)
    # mono_scip = make_mono(signal_scip)

    # numpy_fft_analysis(signal_scip, sample_rate_scip)
    librosa_fft_analysis(signal=signal_lib, sample_rate=sample_rate_lib, fps=3, n=3)


    # # plot testing
    # plot_values(mono_lib, sample_rate_lib)
    # plot_values(mono_scip, sample_rate_scip)

    # # # compare data
    # i = 0
    # while (i < len(mono_lib)):
    #     if(mono_lib[i] != mono_scip[i]):
    #         print("lib: " + str(mono_lib[i]) + " scip: " + str(mono_scip[i]))
    #     i +=1

def load_audio_librosa(file_name, data_dir = 'soundfiles/', sr = 22050):
    file_name = data_dir + file_name
    print("File: %s" % file_name)
    signal, sample_rate = librosa.load(file_name, sr=sr)
    print("Librosa Sample Rate: %d" % sample_rate)
    return signal, sample_rate

def load_audio_scipy(file_name, data_dir = 'soundfiles/'):
    file_name = data_dir + file_name
    print("File: %s" % file_name)
    sample_rate, signal = wavfile.read(file_name)
    print("scipy Sample Rate: %d samples/sec" % sample_rate)
    return signal, sample_rate

def plot_values(mono_signal, sample_rate):
    # create numpy linspace for plotting using sample rate and length of signal
    xaxis = np.linspace(0, len(mono_signal)/sample_rate, num=len(mono_signal))
    ax1 = plt.subplot(211)
    plt.plot(xaxis, mono_signal)
    plt.subplot(212, sharex = ax1)
    spec, f, t, i = plt.specgram(mono_signal, NFFT =256, Fs = sample_rate, noverlap=128, mode = 'psd', sides = 'default')
    plt.show()

def make_mono(signal, channel = 0):
  # if signal is stereo, convert to mono
    if signal.ndim == 2:
        return signal[:,channel]
    else:
        print("Signal not stereo, has %d dimension " % signal.ndim)
        return signal

# def schumaker_fft(signal, sample_rate, window_size = 2**14):

def numpy_fft_analysis(signal, sample_rate, window_size = 2**14):
    print("FFT window size: %d samples"  % window_size) 
    # print("Sample Rate: " + str(sample_rate))
    print("Window Duration: %.4f seconds" % (window_size/sample_rate))
    
    window_start = 0
    while(window_start < len(signal)):
        window = signal[window_start: window_start + window_size]

        fft_out = rfft(window)
        frequencies = rfftfreq(window_size, 1/sample_rate)

        magnitudes = np.abs(fft_out) 
        magnitudes = (magnitudes * 2)/float(window_size)

        window_start = len(signal)
    print(len(magnitudes))
    print(magnitudes.ndim)

    print(len(frequencies))
    print(frequencies.ndim)

def librosa_fft_analysis(signal, sample_rate, n_fft=2048, fps=3, n = 3):
    # test_file_path = resources.sample_wav_file('wav_1.wav')
    # y, sr = librosa.load(test_file_path, sr=None)
    # frames = librosa.util.frame(signal, frame_length=2048, hop_length=1024)
    hop_length = int(sample_rate/fps) # distance in samples between each fft (inclusive?) = samples per second/frames per second
    print(signal.shape)
    S = np.abs(librosa.stft(signal, center=False, n_fft=n_fft, hop_length = hop_length))
    print("interval = %.4f seconds" % hop_length)
    print(S.shape)
    time = 0
    # max_frequency[2]
    while(time < len(S[0])):
        freq = 0
        # max_freqs = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0] # creates a 2d array of num_loudest number of loudest bins
        max_freqs = [[0 for x in range(4)] for y in range(n)] # creates a 2d array of num_loudest number of loudest bins
        print(max_freqs)
        while(freq < n_fft/2):
            # print(freq)
            i = 0
            empty = False
            while i < n: # check if max_frequency array has empty entries
                # print("CHECKING EMPTY LOOP {} {} {:.4f} {}".format(time, freq,S[freq][time], i))
                if max_freqs[i][1] == float(0.0000): # does this convert to int?
                    max_freqs[i][0] = time*hop_length/sample_rate
                    max_freqs[i][1] = S[freq][time]
                    max_freqs[i][2] = freq*(sample_rate/n_fft)
                    max_freqs[i][3] = freq

                    empty = True
                    i = n
                i = i + 1
            smallest = float(100) # smallest amplitude
            smallest_index = 0 # index of max_frequency with smallest amplitude
            i = 0
            if(empty == False): # if not false replace smallest amplitude to replace
                # print("HERE {}".format(freq))

                while i < n: # find smallest bin
                    if smallest > max_freqs[i][1]:
                        smallest = max_freqs[i][1]
                        smallest_index = i                
                    i = i + 1
                # print("Smallest: {}".format(smallest))
                if(smallest < S[freq][time]): # is smallest bin less than current amplitude
                    max_freqs[smallest_index][0] = time*hop_length/sample_rate
                    max_freqs[smallest_index][1] = S[freq][time]
                    max_freqs[smallest_index][2] = freq*(sample_rate/n_fft)
                    max_freqs[smallest_index][3] = freq
            # print(max_freqs)
            freq += 1
        # print(max_freqs)
        for max_freq in max_freqs:
            # print("time: {:2.4f} bin_number: {} frequency: {:.4f} amplitude: {:.4f}".format(max_freq[0], max_freq[3], max_freq[2], max_freq[1]))
            print("time: {:2.4f} bin_number: {} frequency range: {:.4f}-{:.4f} amplitude: {:.4f}".format(max_freq[0], max_freq[3], max_freq[2], (max_freq[3]+1)*sample_rate/n_fft, max_freq[1]))
        time += 1
        # if(time == 19):
        #     time = len(S[0])

    # parse all frequencies in X dimension at Y time, get loudest
    
    # for x in S:
    #     count_outer += 1
    #     count_inner = 0
    #     for val in x:

    #         print("frequency (Hz): {} amplitude (dBFS?): {:4f} time (seconds): {:4f}s".format(count_outer*(sample_rate/n_fft), val, count_inner*hop_length/sample_rate ))
    # for bin 
    



if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1]) 
    else:
        main()