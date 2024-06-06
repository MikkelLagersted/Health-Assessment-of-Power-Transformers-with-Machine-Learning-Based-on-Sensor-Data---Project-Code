import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

##phase difference##
def phase_difference(signal1, signal2):
    # Apply FFT
    fft1 = np.fft.fft(signal1)
    fft2 = np.fft.fft(signal2)

    # Get the phase angle of each FFT
    phase1 = np.angle(fft1)
    phase2 = np.angle(fft2)

    # Assume you want to compare the phase of the first peak frequency
    dominant_frequency_index = np.argmax(np.abs(fft1))
    phase_difference = phase2[dominant_frequency_index] - phase1[dominant_frequency_index]

    return phase_difference % (2 * np.pi) #in radians

def CalcMPC(freq_temp_signal1,freq_temp_signal2,freq_temp_signal3):
    #Constructing X matrix
    wi1 = np.zeros(len(freq_temp_signal1)//3000)
    wi2 = np.zeros(len(freq_temp_signal1)//3000)
    wi3 = np.zeros(len(freq_temp_signal1)//3000)
    
    for i in range(len(freq_temp_signal1)//3000-1):
        phase_sig1 = freq_temp_signal1[i*3000:(i+1)*3000]
        phase_sig2 = freq_temp_signal1[(i+1)*3000:(i+2)*3000]
        wi1[i] = np.sqrt(phase_sig1[0]**2 + phase_sig2[0]**2 - 2*phase_sig1[0]*phase_sig2[0]*np.cos(phase_difference(phase_sig1, phase_sig2)))
        
        phase_sig1 = freq_temp_signal2[i*3000:(i+1)*3000]
        phase_sig2 = freq_temp_signal2[(i+1)*3000:(i+2)*3000]
        wi2[i] = np.sqrt(phase_sig1[0]**2 + phase_sig2[0]**2 - 2*phase_sig1[0]*phase_sig2[0]*np.cos(phase_difference(phase_sig1, phase_sig2)))
    
        phase_sig1 = freq_temp_signal3[i*3000:(i+1)*3000]
        phase_sig2 = freq_temp_signal3[(i+1)*3000:(i+2)*3000]
        wi3[i] = np.sqrt(phase_sig1[0]**2 + phase_sig2[0]**2 - 2*phase_sig1[0]*phase_sig2[0]*np.cos(phase_difference(phase_sig1, phase_sig2)))
    
    X = np.column_stack((wi1, wi2, wi3))
    
    ### Applying PCA to X matrix ###
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    # Eigenvalues
    eigenvalues = pca.explained_variance_
    
    # Compute MPC
    MPC = eigenvalues[0] / np.sum(eigenvalues)
    
    return MPC
