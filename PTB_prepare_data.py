import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wfdb
import os


def is_healthy_control(hea_file_path):
    record = wfdb.rdheader(hea_file_path[0:-4])
    comments = record.comments
    # Check for keywords indicating absence of diagnosed cardiac conditions
    for comment in comments:
        if ": " in comment:
            # Split the string into key and value
            key, value = comment.split(": ", 1)
        else:
            # If separator doesn't exist, treat the whole string as value
            key = "Unknown"
            value = comment

        if "Healthy control".lower() in value.lower():
            return True
    return False


MainFolder = r"F:\ptb-diagnostic-ecg-database-1.0.0"

patient_files = [file for file in os.listdir(MainFolder) if 'patient' in file]

n = 50000  # number of samples taken from each channel
counter = 4  # number of needed persons
data_files = []
participant_data = {}
data_array = np.empty((counter, n, 15))
channel_names = ['i', 'vx', 'vy']
for index in range(len(patient_files)):

    if counter == 0:
        # random_ = np.random.randint(200, 250)
        # participant = patient_files[random_]
        # path = os.path.join(MainFolder, participant)
        # heaFile = [v for v in os.listdir(path) if ".hea" in v][0]
        # filename = [i for i in os.listdir(path) if ".dat" in i][0]
        # file = path + "\\" + filename
        #
        # data = pd.DataFrame()
        # for i in range(15):
        #     signal_name = wfdb.rdsamp(record_name=file[:-4])[1]['sig_name'][i]
        #     if channel_names.__contains__(signal_name):
        #         signal = wfdb.rdsamp(record_name=file[:-4])[0][n:n+5000, i]
        #         # Reshape the signal to have 'n' samples
        #         channel = pd.DataFrame({str(signal_name): signal})
        #         data = pd.concat([data, channel], axis=1)
        #
        # # can add more descriptive columns from the data
        #
        # # data.to_csv(fr'newData\unknown.csv', index=False)
        # data["Participant"] = 'unknown'
        # # append
        # data_files.append(data)
        break
    else:
        participant = patient_files[index]
        path = os.path.join(MainFolder, participant)
        heaFile = [v for v in os.listdir(path) if ".hea" in v][0]
        datFile = [i for i in os.listdir(path) if ".dat" in i][0]
        file = path + "\\" + datFile

    if is_healthy_control(path + r"\\" + heaFile):
        counter -= 1
        # get signal
        data = pd.DataFrame()
        for i in range(15):
            signal_name = wfdb.rdsamp(record_name=file[:-4])[1]['sig_name'][i]
            if channel_names.__contains__(signal_name):
                signal = wfdb.rdsamp(record_name=file[:-4])[0][n+1:n+5000, i]
                # Reshape the signal to have 'n' samples
                channel = pd.DataFrame({str(signal_name): signal})
                data = pd.concat([data, channel], axis=1)

        # can add more descriptive columns from the data
        data.to_csv(fr'TestData\{participant}.csv', index=False)
        data["Participant"] = participant
        # append
        data_files.append(data)

df = pd.concat(data_files)

# visualize data Examples
participants = df['Participant'].unique()
for participant in participants[:2]:
    participant_data = df[df['Participant'] == participant]
    num_channels = participant_data.shape[1] - 1  # Exclude the 'Participant' column
    num_plots = num_channels // 2 + num_channels % 2  # Calculate the number of subplots
    plt.figure(figsize=(12, 6))
    plt.title(f"{participant}")
    for i in range(num_channels):
        plt.subplot(num_plots, 2, i + 1)
        channel_name = participant_data.columns[i]
        # Convert index and signal data to numpy arrays
        index = participant_data.index.to_numpy()
        signal = participant_data[channel_name].to_numpy()
        plt.plot(index[0:3000], signal[0:3000])
        plt.title(f'{channel_name}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True)
    plt.rcParams['figure.constrained_layout.use'] = False
    plt.tight_layout()
    plt.show()
print('done')
