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
    # signal_lib, sample_rate_lib = load_audio_librosa('fft_test.wav')
    signal_scip, sample_rate_scip = load_audio_scipy('fft_test.wav')

    # mono signals
    # mono_lib = make_mono(signal_lib)
    mono_scip = make_mono(signal_scip)

    fft_analysis(signal_scip, sample_rate_scip)

    # # plot testing
    # plot_values(mono_lib, sample_rate_lib)
    # plot_values(mono_scip, sample_rate_scip)

    # # # compare data
    # i = 0
    # while (i < len(mono_lib)):
    #     if(mono_lib[i] != mono_scip[i]):
    #         print("lib: " + str(mono_lib[i]) + " scip: " + str(mono_scip[i]))
    #     i +=1

def load_audio_librosa(file_name, data_dir = 'soundfiles/', sr = 44100):
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

def fft_analysis(signal, sample_rate, window_size = 2**14):
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
    

if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1]) 
    else:
        main()