import numpy as np

def CalcEDR(current_signal, template_signal, sampling_rate):
    
    energy_bands = [(400, 800), (800, 1200), (1200, 1600), (1600, 2000)]
    band_energies = np.zeros(len(energy_bands))

    #Calculating the template means (packets of sequential 3000 samples)
    packet_size = 3000
    num_packets = len(template_signal) // packet_size  # Calculate the number of full packets
    mean_band_energies = np.zeros((num_packets, len(energy_bands)))
    
    # Process each packet of 3000 samples
    for i in range(num_packets):
        # Extract the current packet
        packet = template_signal[i * packet_size:(i + 1) * packet_size]
    
        # Step 2: Convert packet to frequency domain using DFT
        frequency_domain_packet = np.fft.fft(packet)
        frequency_bins = np.fft.fftfreq(packet_size, d=1/sampling_rate)  # Assuming sample rate is 1
    
        # Step 3 & 4: Calculate energies and normalize within each band
        for n, (f_start, f_end) in enumerate(energy_bands):
            band_frequencies = (frequency_bins >= f_start) & (frequency_bins < f_end)
            band_energies = np.sum(np.abs(frequency_domain_packet[band_frequencies])**2)
            mean_band_energies[i, n] = band_energies
    
    # Define an epsilon
    epsilon = 1e-8

    # Normalize the band energies for each packet
    frequency_template = mean_band_energies / (np.sum(mean_band_energies, axis=1, keepdims=True) + epsilon)

    
    # Find the average value across packets for each energy band
    frequency_template = np.mean(frequency_template, axis=0)
    
    
    #Calculating for current sample
    
    # Step 1: DFT to get frequency-domain representation
    frequency_domain_signal = np.fft.fft(current_signal)
    frequency_bins = np.fft.fftfreq(len(current_signal), d=1/sampling_rate)
    
    # Step 3: Calculate the energy for each band
    band_energies = np.zeros(len(energy_bands))
    for n, (f_start, f_end) in enumerate(energy_bands):
        band_frequencies = (frequency_bins >= f_start) & (frequency_bins < f_end)
        band_energies[n] = np.sum(np.abs(frequency_domain_signal[band_frequencies])**2)
    
    # Step 4: Normalize band energies
    normalized_band_energies = band_energies / np.sum(band_energies)
    
    #Calculate the distances between each frequency band
    E_diffs = np.zeros(len(frequency_template))
    for i in range(len(frequency_template)):
        E_diffs[i]=np.linalg.norm(normalized_band_energies[i]-frequency_template[i])
    
    EDR = E_diffs.mean()*100
    return EDR
