import os
import pickle
from matplotlib import pyplot as plt
from ECG_preprocessing import *

participant_data = {}
MainFolder = r"F:\8th semester\HCI\Labs\project\newData"
patients = []
patient_files = [file for file in os.listdir(MainFolder)]
for participant in patient_files:
    patients.append(participant.split('.')[0])
    participant_data[participant.split('.')[0]] = pd.read_csv(os.path.join(MainFolder, participant))

# Define the Butterworth low-pass filter parameters
order = 5
fs = 1000  # sampling rate of 1000 Hz
cutoff = 50  # Cutoff frequency in Hz
cutoff_low = 1

# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# channel_name = participant_data[patients[0]].columns[0]
# # Convert index and signal data to numpy arrays
# index = participant_data[patients[0]].index.to_numpy()
# signal = participant_data[patients[0]][channel_name].to_numpy()
# plt.plot(index[0:3000], signal[0:3000])
# plt.title(f'{channel_name}')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.rcParams['figure.constrained_layout.use'] = False
# plt.tight_layout()
# plt.show()

participant_data, segments = preprocessing(participant_data, cutoff_low, cutoff, fs, order)


print("segments are done")
x = pd.DataFrame(segments[patients[1]])

plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    # Convert index and signal data to numpy arrays
    index = x.columns.to_numpy()
    signal = x.iloc[0].to_numpy()
    plt.plot(index[:], signal[:])
    plt.title(f'segment:{i+1}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
plt.rcParams['figure.constrained_layout.use'] = False
plt.tight_layout()
plt.show()

# plot after preprocessing
plt.figure(figsize=(12, 6))
i = 1
for col in participant_data[patients[0]].columns:

    plt.subplot(3, 1, i)
    i += 1
    channel_name = col
    # Convert index and signal data to numpy arrays
    X = []
    Y = []
    index = participant_data[patients[0]].index.to_numpy()
    signal = participant_data[patients[0]][channel_name].to_numpy()
    # data_peaks[f"{patients[0]}-{channel_name}"] = data_peaks[f"{patients[0]}-{channel_name}"][
    #     (data_peaks[f"{patients[0]}-{channel_name}"] > 0) & (data_peaks[f"{patients[0]}-{channel_name}"] < 3000)]
    # for i in range(len(data_peaks[f"{patients[0]}-{channel_name}"]) - 1):
    #     L = (data_peaks[f"{patients[0]}-{channel_name}"][i])
    #     # if signal[L] > 0.2 or signal[L] < -0.3:
    #     Y.append(signal[L])
    #     X.append(data_peaks[f"{patients[0]}-{channel_name}"][i])
    plt.plot(index[0:3000], signal[0:3000])
    plt.title(f' ({patients[0]}) filtered signal of channel {channel_name}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.rcParams['figure.constrained_layout.use'] = False
    plt.tight_layout()
    # plt.plot(X, Y, "x")
plt.show()

print("done")


segments_array, Labels = prepare_segments_array(segments)

# with open('segments.pkl', 'wb') as f:
#     pickle.dump(segments_array, f)
#
# with open('Labels.pkl', 'wb') as f:
#     pickle.dump(Labels, f)

print("finish")
