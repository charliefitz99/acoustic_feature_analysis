import scipy.io.wavfile as wavfile
from matplotlib import pyplot as plt
from numpy.fft import rfft, rfftfreq
import numpy as np
from scamp_extensions.pitch import hertz_to_midi
from scamp import *


NUM_TOP_FREQS = 50

s = Session()

s.tempo = 240

clarinet = s.new_part("clarinet")

#Only  takes in .wav files
sample_rate, signal = wavfile.read("Robeson-long-ways-from-home-mono_01.wav")

if len(signal.shape) > 1:
    # extract first channel if it's a stereo file
    signal = signal[:, 0]

#window_size = 2**14  # 16384 samples = 0.3715 seconds for each analysis window

window_size = 2**14  # 16384 samples = 0.3715 seconds for each analysis window
#window_size = int(len(signal) / 10)  # 16384 samples = 0.3715 seconds for each analysis window


#s.fast_forward_in_beats(1000)

step_size = window_size // 2

s.start_transcribing()

start_sample = 0

held_pitches = []

i = 1

while start_sample < len(signal):
    
    window = signal[start_sample: start_sample + window_size]

    amplitude_spectrum = np.abs(rfft(window))
    frequencies = rfftfreq(len(window), 1/sample_rate)
    loud_bins = np.argsort(amplitude_spectrum)[::-1]
    top_frequencies = frequencies[loud_bins][:NUM_TOP_FREQS]
    top_amplitudes = amplitude_spectrum[loud_bins][:NUM_TOP_FREQS]
    
    
    #for freq, amplitude in zip(top_frequencies, top_amplitudes):
        #print("midi:", hertz_to_midi(freq), "Amp", amplitude/top_amplitudes[0])
    
    #"Resynthesize the sound analysis with clarinet sounds, each analysis frame lasting 0.15 of a beat
    for freq, amplitude in zip(top_frequencies, top_amplitudes):
        #clarinet.play_note(hertz_to_midi(freq), amplitude/top_amplitudes[0], 0.15, blocking=False)
        #clarinet.play_note(hertz_to_midi(freq), amplitude/top_amplitudes[0], 0.125)
        if freq > 0:
            held_pitches.append(hertz_to_midi(freq))
    #round frequencies to make them equal temperament
    held_pitches = [round(num) for num in held_pitches]
    #remove duplicates
    held_pitches = list(set(held_pitches))
    #sort pitches
    held_pitches.sort()
    print("pitches_"+str(i)+" = ", held_pitches)
    #print("length: ", len(held_pitches))
    i = i+1
    
    for pitch in held_pitches:
        clarinet.play_note(pitch, 1, 0.125)
    
    held_pitches = []
    
    wait(1)
   

    #print("BREAK")
    #num_skip = N times the step size to limit output analysis frames
    num_skip = 7
    start_sample += step_size * num_skip

performance = s.stop_transcribing()
performance.to_score(composer = "matt schumaker", title = "robeson-analysis").show()

