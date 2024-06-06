import numpy as np


def CalcFC(signal,signal_duration, sampling_rate, frequencies):
    number_of_samples = int(signal_duration * sampling_rate)
    fmax = frequencies[-1]

    # Step 1: DFT to get frequency-domain representation
    frequency_domain_signal = np.fft.fft(signal)
    frequency_bins = np.fft.fftfreq(number_of_samples, d=1/sampling_rate) # with sample spacing d=1.0, replace with actual sampling rate if needed

    # Step 2: Filter and extract amplitude at our frequencies of interest
    # Since FFT output is symmetrical, consider only the positive half of the frequencies
    half_n = number_of_samples // 2
    amplitudes = dict()
    for f in frequencies:
        # Find the index closest to the frequency of interest
        index = np.argmin(np.abs(frequency_bins[:half_n] - f))
        # Store the amplitude at that frequency
        amplitudes[f] = np.abs(frequency_domain_signal[index])
    total_energy = sum((f/fmax * amplitudes[f]) ** 2 for f in frequencies)
    
    # Adding a small number to avoid division by zero
    epsilon = 1e-10
    total_energy += epsilon
    
    # Step 3: Calculate energy proportions (pf) and weights (wf)
    energy_proportions = {f: (f/2000 * amplitudes[f]) ** 2 / total_energy for f in frequencies}
    
    # Step 4: Calculate FC
    FC = -sum(pf * np.log(pf) for pf in energy_proportions.values() if pf > 0) # The if condition handles the log(0) case
    return FC