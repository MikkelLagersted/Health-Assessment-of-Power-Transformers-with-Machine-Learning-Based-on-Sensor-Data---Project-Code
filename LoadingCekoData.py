import json
import pandas as pd
import numpy as np
import os
from scipy.signal import detrend
from scipy.signal import firwin, lfilter

from CalcFC import CalcFC
from CalcDET2 import CalcDET2
from CalcEDR2 import CalcEDR
from CalcMPC import CalcMPC


##### Importing #####
# Directory containing your JSON files
desired_columns = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6']
directory = 'D:/Ishoj/Extracted'
# Initialize an empty list to store dataframes
dfs_sensor = []
FCs_sensor1 = np.array([])
FCs_sensor2 = np.array([])
FCs_sensor3 = np.array([])

template_signal1 = np.array([])
template_signal2 = np.array([])
template_signal3 = np.array([])

avg_FCs_s1 = np.array([])
avg_FCs_s2 = np.array([])
avg_FCs_s3 = np.array([])

EDRs_s1 = np.array([])
EDRs_s2 = np.array([])
EDRs_s3 = np.array([])

DETs_s1 = np.array([])
DETs_s2 = np.array([])
DETs_s3 = np.array([])

MPCs = np.array([])

date_times = pd.DataFrame()

file_times = np.array([])

#Initialize counters and arrays
counter = 0

#Times
h24_us = 86400000000

#Only look at nth file
nth_file = 1
file_in_row = 0
n_files_avg = 12

##### Constants #####
#FFT Constants
samples_used = 3000
sampling_rate = 3000
signal_duration = samples_used/sampling_rate

frequencies = np.arange(100, 2100, 100) # 100, 200, ..., 2000 Hz

#Sensor sensitivities
sensor1_sensitivity = 0.497
sensor2_sensitivity = 0.228
sensor3_sensitivity = 0.168

sensor4_sensitivity = 0.01005
sensor5_sensitivity = 0.00975
sensor6_sensitivity = 0.01009

#Center wavelengths
cw_sensor4 = 1554.36284
cw_sensor5 = 1549.72909
cw_sensor6 = 1545.46436

#Filter constants
cutoff_frequency = 100  # Cutoff frequency for high-pass filter
numtaps = 101  # or any other suitable value
fir_coefficients = firwin(numtaps, cutoff=cutoff_frequency, pass_zero=False, fs=sampling_rate)



#Determinism constants
embed_dim = 3 #As per paper
delay_time = 3 #Calculated from correlation
threshold = 0.2 #As per paper
min_l = 2 #As per paper


#Loading of data, and calculating parameters for every n'th file
for filename in sorted(os.listdir(directory)):
    if filename.endswith('.json'):
        # Process only every 20th file
        if counter % nth_file == 0:
            filepath = os.path.join(directory, filename)
            # Load JSON data into a DataFrame
            with open(filepath, 'r') as file:
                data = json.load(file)
            file_in_row = file_in_row+1
            
            # Sensor data 
            selected_data = {column: data[column] for column in desired_columns}
            df_sensor = pd.DataFrame(selected_data)
            df_sensor = df_sensor.iloc[:samples_used,:]

            # Applying detrend to raw data
            detrended_values = detrend(df_sensor.iloc[:, :3], axis=0)
            detrended_df = pd.DataFrame(detrended_values, columns=df_sensor.columns[:3])

            df_sensor["sensor1"] = detrended_df["sensor1"]/sensor1_sensitivity
            df_sensor["sensor2"] = detrended_df["sensor2"]/sensor2_sensitivity
            df_sensor["sensor3"] = detrended_df["sensor3"]/sensor3_sensitivity

            # Apply temperature recalculation
            df_sensor["sensor4"] = (df_sensor["sensor4"]-cw_sensor4)/sensor4_sensitivity
            df_sensor["sensor5"] = (df_sensor["sensor5"]-cw_sensor5)/sensor5_sensitivity
            df_sensor["sensor6"] = (df_sensor["sensor6"]-cw_sensor6)/sensor6_sensitivity
            
            #Applying filter to the data
            df_sensor["sensor1"] = lfilter(fir_coefficients, 1.0, df_sensor["sensor1"])
            df_sensor["sensor2"] = lfilter(fir_coefficients, 1.0, df_sensor["sensor2"])
            df_sensor["sensor3"] = lfilter(fir_coefficients, 1.0, df_sensor["sensor3"])
            signal1 = df_sensor["sensor1"].to_numpy()
            signal2 = df_sensor["sensor2"].to_numpy()
            signal3 = df_sensor["sensor3"].to_numpy()
            
            #Creating 24 hour signal for template
            template_signal1 = np.append(template_signal1, signal1)
            template_signal2 = np.append(template_signal2, signal2) 
            template_signal3 = np.append(template_signal3, signal3) 
            
            #Calculate FC for current sample
            FCs_sensor1 = np.append(FCs_sensor1, CalcFC(signal1, signal_duration, sampling_rate, frequencies))
            FCs_sensor2 = np.append(FCs_sensor2, CalcFC(signal2, signal_duration, sampling_rate, frequencies))
            FCs_sensor3 = np.append(FCs_sensor3, CalcFC(signal3, signal_duration, sampling_rate, frequencies))
            
            file_times = np.append(file_times, data["startTime"])
            
            if (file_times[-1]-file_times[0]) > h24_us: #24 hourly data
                #Mean FCs:
                avg_FCs_s1 = np.append(avg_FCs_s1, FCs_sensor1.mean())
                avg_FCs_s2 = np.append(avg_FCs_s2, FCs_sensor2.mean())
                avg_FCs_s3 = np.append(avg_FCs_s3, FCs_sensor3.mean())
                #Re-empty arrays
                FCs_sensor1 = np.array([])
                FCs_sensor2 = np.array([])
                FCs_sensor3 = np.array([])
                
                #Calculate DETs
                #DETs_s1 = np.append(DETs_s1, CalcDET2(signal1, embed_dim, delay_time, threshold, min_l))
                #DETs_s2 = np.append(DETs_s2, CalcDET2(signal2, embed_dim, delay_time, threshold, min_l))
                #DETs_s3 = np.append(DETs_s3, CalcDET2(signal3, embed_dim, delay_time, threshold, min_l))
                
                #EDRs
                EDRs_s1 = np.append(EDRs_s1, CalcEDR(signal1, template_signal1, sampling_rate))
                EDRs_s2 = np.append(EDRs_s2, CalcEDR(signal2, template_signal2, sampling_rate))
                EDRs_s3 = np.append(EDRs_s3, CalcEDR(signal3, template_signal3, sampling_rate))
                
                #MPC
                MPCs = np.append(MPCs, CalcMPC(template_signal1, template_signal2, template_signal3))
                
                #Datetime
                date_time = pd.DataFrame([pd.to_datetime(data["startTime"], unit="us")])
                date_times = pd.concat([date_times, date_time],ignore_index=True)
                
                #Resetting template signal
                template_signal1 = np.array([])
                template_signal2 = np.array([])
                template_signal3 = np.array([])
                
                #Reset timesteps
                file_times = np.array([])

        # Increment counter
        counter += 1

# Concatenate all DataFrames in the list into a single DataFrame
combined_df_sensor = pd.DataFrame()
combined_df_sensor["FC sensor1"] = avg_FCs_s1
combined_df_sensor["FC sensor2"] = avg_FCs_s2
combined_df_sensor["FC sensor3"] = avg_FCs_s3

#combined_df_sensor["DET sensor1"] = DETs_s1
#combined_df_sensor["DET sensor2"] = DETs_s2
#combined_df_sensor["DET sensor3"] = DETs_s3

combined_df_sensor["EDR sensor1"] = EDRs_s1
combined_df_sensor["EDR sensor2"] = EDRs_s2
combined_df_sensor["EDR sensor3"] = EDRs_s3

combined_df_sensor["MPC"] = MPCs

combined_df_sensor["Datetime"] = date_times

combined_df_sensor.to_csv("D:/ISH Health Metrics.csv", index=False)
