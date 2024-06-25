import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import StandardScaler


# at first from data visualization we find that channel i and vx is the most real signals ,so we will
# use these two channels and drop others


# Function to apply Butterworth low-pass filter to ECG signal
def butter_lowpass_filter(data, cutoff_, fs_, order_):
    nyquist = 0.5 * fs_
    normal_cutoff = cutoff_ / nyquist
    b, a = butter(order_, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def butter_bandpass_filter(data, low_cut, high_cut, fs_, order_):
    nyquist = 0.5 * fs_
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = butter(order_, [low, high], btype='band', analog=False)
    y = filtfilt(b, a, data)
    return y


def Normalize(df):
    # Initialize StandardScaler
    scaler = StandardScaler()
    # Fit and transform the data
    normalized_data = scaler.fit_transform(df)

    # Convert the normalized data back to a DataFrame
    df_normalized = pd.DataFrame(normalized_data, columns=df.columns)

    # Replace the original DataFrame with the normalized one
    df[df.columns] = df_normalized

    return df


def my_findpeaks(column, threshold_low=None, threshold_high=None):
    result = []
    m = len(column)
    for indexer in range(4, m - 4):
        if threshold_low > column[indexer] or column[indexer] > threshold_high:
            if (column[indexer - 1] < column[indexer] > column[indexer + 1]) or (
                    column[indexer - 1] > column[indexer] < column[indexer + 1]):
                result.append(indexer)
    return result


def moving_average(signal_, window_size):
    """
    Calculates the moving average of a signal using convolution.

    Parameters:
    - signal (numpy array): Input signal for which the moving average is calculated.
    - window_size (int): Size of the moving average window.

    Returns:
    - ma_signal (numpy array): Signal after applying the moving average.
    """
    weights = np.ones(window_size) / window_size  # Create a weight array for the moving average
    ma_signal = np.convolve(signal_, weights, mode='same')  # Perform convolution to calculate moving average

    return ma_signal  # Return the signal after applying the moving average


def pan_and_tompkins(channel_record, sampling_rate):
    """
    Detects R-peaks in an ECG signal using the Pan-Tompkins algorithm.

    Parameters:

    - ecg_signal (numpy array): ECG signal to detect R-peaks from.
    - sampling_rate (float): Sampling rate of the ECG signal in Hz.

    Returns:
    - r_peaks (list): List of indices corresponding to detected R-peaks.
    """
    # Constants for Pan-Tompkins algorithm
    window_size = int(0.1 * sampling_rate)  # Window size for moving average integration
    refractory_period = int(0.2 * sampling_rate)  # Refractory period for R-peaks
    threshold_factor = 0.6  # Factor for setting R-peaks detection threshold

    # Preprocessing: Bandpass filter, differentiation, and squaring
    filtered_signal = channel_record  # Apply Bandpass filter
    differentiated_signal = np.gradient(filtered_signal)  # Differentiation
    squared_signal = differentiated_signal ** 2  # Squaring

    # Moving average integration
    ma_signal = moving_average(squared_signal, window_size)

    # Find the maximum value as the threshold for R-peaks
    threshold = threshold_factor * np.max(ma_signal)

    # Initialize variables
    r_peaks = []  # List to store R-peak indices
    last_r_peak = 0  # Variable to track last detected R-peak index

    # Detection of R-peaks
    for i in range(window_size, len(ma_signal) - refractory_period):
        if ma_signal[i] > threshold and i - last_r_peak > refractory_period:
            # Check if it is the maximum in the local window
            if ma_signal[i] >= np.max(ma_signal[i - window_size: i + window_size + 1]):
                r_peaks.append(i)
                last_r_peak = i

    return r_peaks  # Return the list of detected R-peaks


def Segmentation(channel_record):
    """
    Segments an ECG signal around R-peaks.

    Parameters:
    - signal (numpy array): ECG signal to be segmented.

    Returns:
    - segmented_signal (numpy array): Segmented signal around R-peaks.
    - beforeR (int): Length of the segment before R-peak.
    - afterR (int): Length of the segment after R-peak.
    """
    peaks = pan_and_tompkins(channel_record, 1000)  # Detect R-peaks in the signal
    # Calculate average heartbeat length based on the first three R-peaks
    heartbeatLength = ((peaks[1] - peaks[0]) + (peaks[2] - peaks[1])) / 2
    beforeR = int((1 / 3) * heartbeatLength)  # Length of segment before R-peak
    afterR = int((2 / 3) * heartbeatLength)  # Length of segment after R-peak
    segmented_signal = []
    for j in range(len(peaks) - 1):
        # Segment the signal around the third R-peak
        temp = channel_record[peaks[j] - beforeR: peaks[j] + afterR]
        segmented_signal.append(temp[0:600])

    return np.array(segmented_signal)  # Return the segmented signal


def preprocessing(data, cutoff_low_, cutoff_high, fs_, order_):
    new_data = {}
    segments_dict = {}
    for participant_, data_ in data.items():
        segment = object
        # Apply filter to each column (ECG signal) except the last one
        preprocessed_data = data_.copy()
        iteration = 0
        for column in preprocessed_data.columns[:]:
            preprocessed_data[column] = butter_bandpass_filter(preprocessed_data[column],
                                                               cutoff_low_, cutoff_high, fs_, order_)
            if iteration == 0:
                segment = Segmentation(preprocessed_data[column])
                iteration += 1
            else:
                segment = np.vstack((segment, Segmentation(preprocessed_data[column])))
            preprocessed_data[column] = preprocessed_data[column] - np.mean(preprocessed_data[column])

        segments_dict[participant_] = segment
        new_data[participant_] = preprocessed_data

    return new_data, segments_dict


def prepare_segments_array(segments_):
    Labels_ = []
    segments_array_ = []
    for patient in segments_:
        cur_patient = segments_[patient]
        for c in range(cur_patient.shape[0]):
            g = np.array(cur_patient[c])
            segments_array_.append(g)
            Labels_.append(patient)
    segments_array_ = np.array(segments_array_)
    Labels_ = np.array(Labels_)
    return segments_array_, Labels_


def minimumRadiusOfCurvature(signal, peakIndex, windowSize, isOnset):
    """
    Calculates the minimum radius of curvature in a signal segment around a peak index.

    Parameters:
    - signal (numpy array): Input signal.
    - peakIndex (int): Index of the peak around which the radius of curvature is calculated.
    - windowSize (int): Size of the window around the peak index.
    - isOnset (bool): Flag indicating whether the peak is an onset or not.

    Returns:
    - minCurveIndex (int): Index corresponding to the minimum radius of curvature.
    """
    t = np.arange(0, len(signal), 1)  # Time array
    X = peakIndex  # Peak index around which the radius of curvature is calculated
    if isOnset:
        Y = peakIndex - windowSize  # Start index of the window
        if Y < 0:
            Y = 0  # Ensure Y is within signal range
    else:
        Y = peakIndex + windowSize  # End index of the window
        if Y >= len(signal):
            Y = len(signal) - 1  # Ensure Y is within signal range

    a = [t[Y] - t[X], signal[Y] - signal[X]]  # Vector 'a' representing the window direction
    normA = math.sqrt(a[0] ** 2 + a[1] ** 2)  # Norm of vector 'a'
    C = X  # Initialize current index 'C' to the peak index
    allSigma = []  # List to store all curvature values
    allSigmaIndex = []  # List to store corresponding indices for curvature values

    # Calculate curvature for each point in the window
    while C != Y:
        c = [t[C] - t[X], signal[C] - signal[X]]  # Vector 'c' from peak to current point
        sigma = abs(np.cross(a, c)) / normA  # Curvature calculation
        allSigma.append(sigma)  # Append curvature value to list
        allSigmaIndex.append(C)  # Append corresponding index to list
        if isOnset:
            C -= 1  # Move to the previous point in the window
        else:
            C += 1  # Move to the next point in the window

    if len(allSigma) == 0:
        return Y  # Return the end index of the window if no curvature values are calculated
    else:
        finalSigmaIndex = np.argmax(allSigma)  # Find the index of maximum curvature
        return allSigmaIndex[finalSigmaIndex]  # Return the index corresponding to minimum radius of curvature


def findP(segment_, qrs_onset):
    """
    Finds the P-wave index in a signal given the QRS onset index.

    Parameters:
    - signal (numpy array): Input signal.
    - qrs_onset (int): Index of the QRS onset in the signal.

    Returns:
    - p_peak (int): Index of the P-wave peak in the signal.
    """
    j = qrs_onset - 200  # Start searching 200 samples before QRS onset
    if j < 0:
        j = 0  # Ensure j is within signal range

    max_val = -np.inf  # Initialize maximum value as negative infinity
    index = 0  # Initialize index variable

    # Search for the P-wave peak index
    while j < qrs_onset:
        if segment_[j] > max_val:
            max_val = segment_[j]
            index = j
        j += 1

    p_peak = index  # P-wave peak index is the index of the maximum value before QRS onset
    return p_peak  # Return the P-wave peak index


def findqrsOffset(segment_, R):
    """
    Finds the QRS offset index in a signal given the R-peak index.

    Parameters:
    - signal (numpy array): Input signal.
    - R (int): Index of the R-peak in the signal.

    Returns:
    - qrs_offset (int): Index of the QRS offset in the signal.
    """
    j = R + 50  # Start searching 50 samples after R-peak
    if j > len(segment_) - 1:
        j = len(segment_) - 1  # Ensure j is within signal range

    # Search for the QRS offset index
    while j > R and segment_[j] > segment_[j + 1]:
        j -= 1

    qrs_offset = j  # QRS offset index is the index where signal[j] <= signal[j + 1]
    return qrs_offset  # Return the QRS offset index


def findqrsOnset(segment_, R):
    """
    Finds the QRS onset index in a signal given the R-peak index.

    Parameters:
    - signal (numpy array): Input signal.
    - R (int): Index of the R-peak in the signal.

    Returns:
    - qrs_onset (int): Index of the QRS onset in the signal.
    """
    j = R - 50  # Start searching 50 samples before R-peak
    if j < 0:
        j = 0  # Ensure j is within signal range

    # Search for the QRS onset index
    while j < R and segment_[j] > segment_[j - 1]:
        j += 1

    qrs_onset = j  # QRS onset index is the index where signal[j] <= signal[j - 1]
    return qrs_onset  # Return the QRS onset index


def findS(segment_, R):
    """
    Finds the S-wave index in a signal given the R-peak index.

    Parameters:
    - signal (numpy array): Input signal.
    - R (int): Index of the R-peak in the signal.

    Returns:
    - S (int): Index of the S-wave in the signal.
    """
    j = R + 100  # Start searching 100 samples after R-peak
    if j > len(segment_) - 1:
        j = len(segment_) - 1  # Ensure j is within signal range
    min_val = float('inf')  # Initialize minimum value as positive infinity
    index = 0  # Initialize index variable

    # Search for the minimum value after the R-peak
    while j > R:
        if segment_[j] < min_val:
            min_val = segment_[j]
            index = j
        j -= 1

    S = index  # S-wave index is the index of the minimum value after the R-peak
    return S  # Return the S-wave index


def findQ(segment_, R):
    """
    Finds the Q-wave index in a signal given the R-peak index.

    Parameters:
    - signal (numpy array): Input signal.
    - R (int): Index of the R-peak in the signal.

    Returns:
    - Q (int): Index of the Q-wave in the signal.
    """
    j = R - 100  # Start searching 100 samples before R-peak
    if j < 0:
        j = 0  # Ensure j is within signal range
    min_val = float('inf')  # Initialize minimum value as positive infinity
    index = 0  # Initialize index variable

    # Search for the minimum value before the R-peak
    while j < R:
        if segment_[j] < min_val:
            min_val = segment_[j]
            index = j
        j += 1

    Q = index  # Q-wave index is the index of the minimum value before the R-peak
    return Q  # Return the Q-wave index


def findR(segment_):
    """
    Finds the peak (R-peak) index in a signal.

    Parameters:
    - signal (numpy array): Input signal.

    Returns:
    - R (int): Index of the peak (R-peak) in the signal.
    """
    max_val = -float('inf')  # Initialize maximum value as negative infinity
    index = 0  # Initialize index variable

    # Iterate through the signal to find the maximum value and its index
    for j in range(len(segment_)):
        if segment_[j] > max_val:
            max_val = segment_[j]
            index = j

    R = index  # Peak (R-peak) index is the index of the maximum value in the signal
    return R  # Return the R-peak index


def findPon(signal, p_peak, qrs_onset, isOnset):
    """
    Finds the P-wave onset or offset index in a signal given the P-wave peak index,
    QRS onset index, and a flag indicating whether it's an onset or offset.

    Parameters:
    - signal (numpy array): Input signal.
    - p_peak (int): Index of the P-wave peak in the signal.
    - qrs_onset (int): Index of the QRS onset in the signal.
    - isOnset (bool): Flag indicating whether it's the P-wave onset (True) or offset (False).

    Returns:
    - p_onset_offset (int): Index of the P-wave onset or offset in the signal.
    """
    window = signal[p_peak - 100:p_peak + 100]  # Create a window around the P-wave peak
    z = find_peaks(window)  # Find peaks in the window
    peaks = []
    peaksValue = []

    # Extract peak indices and values from the window
    for k in range(len(z[0])):
        index = np.where(signal == window[z[0][k]])[0][0]  # Get the index of the peak in the original signal
        peaks.append(index)
        peaksValue.append(signal[index])

    peaksValue.sort()  # Sort peak values in ascending order
    threshold = 0.002  # Threshold for distinguishing peaks

    p_onset_offset = 0  # Initialize P-wave onset or offset index

    if len(z[0]) == 0:
        # If no peaks are detected in the window, estimate onset or offset using minimum radius of curvature
        if isOnset:
            p_onset_offset = minimumRadiusOfCurvature(signal, p_peak, 100, True)
        else:
            p_onset_offset = minimumRadiusOfCurvature(signal, p_peak, qrs_onset - p_peak, False)
    else:
        p_peak1 = np.where(signal == peaksValue[-1])  # Index of the highest peak in the window

        # Check if the difference between the highest and ,second-highest peaks is below the threshold

        if round(peaksValue[-1] - peaksValue[-2], 3) <= threshold:
            p_peak2 = np.where(signal == peaksValue[-2])  # Index of the second-highest peak in the window

            if isOnset:
                p_onset_offset = minimumRadiusOfCurvature(signal, p_peak1, 100, True)
            else:
                if p_peak2 < qrs_onset:
                    p_onset_offset = minimumRadiusOfCurvature(signal, p_peak2, qrs_onset - p_peak2, False)
                else:
                    p_onset_offset = minimumRadiusOfCurvature(signal, p_peak2, qrs_onset, False)
        else:
            # If peaks are distinct, estimate onset or offset using minimum radius of curvature
            if isOnset:
                p_onset_offset = minimumRadiusOfCurvature(signal, p_peak, 100, True)
            else:
                p_onset_offset = minimumRadiusOfCurvature(signal, p_peak, qrs_onset - p_peak, False)

    return p_onset_offset  # Return the P-wave onset or offset index


def findT(signal, qrs_offset):
    """
    Finds the T-wave index in a signal given the QRS offset index.

    Parameters:
    - signal (numpy array): Input signal.
    - qrs_offset (int): Index of the QRS offset in the signal.

    Returns:
    - T_peak (int): Index of the T-wave peak in the signal.
    """
    j = qrs_offset + 100  # Start searching 100 samples after QRS offset
    if j > len(signal) - 1:
        j = len(signal) - 1  # Ensure j is within signal range

    max_val = -np.inf  # Initialize maximum value as negative infinity
    index = 0  # Initialize index variable

    # Search for the T-wave peak index
    while j < len(signal):
        if signal[j] > max_val:
            max_val = signal[j]
            index = j
        j += 1

    # Check if the T-wave peak is downward
    if signal[index] == 0:
        j = qrs_offset + 400  # Start searching 400 samples after QRS offset for downward T-wave
        min_val = np.inf  # Initialize minimum value as positive infinity

        # Search for the downward T-wave peak index
        while j < len(signal):
            if signal[j] < min_val:
                min_val = signal[j]
                index = j
            j += 1

    T_peak = index  # T-wave peak index is the index of the maximum value after QRS offset
    return T_peak  # Return the T-wave peak index


def findFiducialPoints(segment_):
    """
    Finds multiple fiducial points in an ECG signal.

    Parameters:
    - signal (numpy array): Input ECG signal.

    Returns:
    - fiducialPoints (list): List of fiducial points including R, Q, S peaks,
                             QRS onset and offset, P peak, P onset and offset,
                             T peak, T onset and offset.
    """
    try:
        fiducialPoints = []  # Initialize list to store fiducial points

        # Find R peak and append to fiducial points list
        R = findR(segment_)
        fiducialPoints.append(R)

        # Find Q peak and append to fiducial points list
        Q = findQ(segment_, R)
        fiducialPoints.append(Q)

        # Find S peak and append to fiducial points list
        S = findS(segment_, R)
        fiducialPoints.append(S)

        # Find QRS onset and append to fiducial points list
        qrs_onset = findqrsOnset(segment_, R)
        fiducialPoints.append(qrs_onset)

        # Find QRS offset and append to fiducial points list
        qrs_offset = findqrsOffset(segment_, R)
        fiducialPoints.append(qrs_offset)

        # Find P peak and append to fiducial points list
        p_peak = findP(segment_, qrs_onset)
        fiducialPoints.append(p_peak)

        # Find P onset and append to fiducial points list
        p_onset = findPon(segment_, p_peak, qrs_onset, True)
        fiducialPoints.append(p_onset)

        # Find P offset and append to fiducial points list
        p_offset = findPon(segment_, p_peak, qrs_onset, False)
        fiducialPoints.append(p_offset)

        # Find T peak and append to fiducial points list
        T_peak = findT(segment_, qrs_offset)
        fiducialPoints.append(T_peak)

        # Find T onset and append to fiducial points list
        t_onset = minimumRadiusOfCurvature(segment_, T_peak, 200, True)
        fiducialPoints.append(t_onset)

        # Find T offset and append to fiducial points list
        t_offset = minimumRadiusOfCurvature(segment_, T_peak, 300, False)
        fiducialPoints.append(t_offset)
        return fiducialPoints  # Return the list of fiducial points

    except Exception as e:
        return None


def findFeatures(fiducialPoints, signal):

    features = [fiducialPoints[0] - fiducialPoints[1], fiducialPoints[2] - fiducialPoints[0],
                fiducialPoints[7] - fiducialPoints[6], fiducialPoints[10] - fiducialPoints[9],
                fiducialPoints[5] - fiducialPoints[1], fiducialPoints[1] - fiducialPoints[2],
                fiducialPoints[4] - fiducialPoints[8], fiducialPoints[6] - fiducialPoints[0],
                fiducialPoints[0] - fiducialPoints[9], fiducialPoints[5] - fiducialPoints[0],
                fiducialPoints[0] - fiducialPoints[8], fiducialPoints[5] - fiducialPoints[2],
                fiducialPoints[1] - fiducialPoints[8], fiducialPoints[6] - fiducialPoints[0],
                fiducialPoints[0] - fiducialPoints[10], fiducialPoints[6] - fiducialPoints[1],
                fiducialPoints[2] - fiducialPoints[10], fiducialPoints[7] - fiducialPoints[3],
                fiducialPoints[4] - fiducialPoints[9], fiducialPoints[7] - fiducialPoints[10],
                fiducialPoints[5] - fiducialPoints[8], signal[fiducialPoints[0]] - signal[fiducialPoints[1]],
                signal[fiducialPoints[8]] - signal[fiducialPoints[5]],
                signal[fiducialPoints[5]] - signal[fiducialPoints[2]],
                signal[fiducialPoints[8]] - signal[fiducialPoints[1]],
                signal[fiducialPoints[0]] - signal[fiducialPoints[5]],
                signal[fiducialPoints[0]] - signal[fiducialPoints[8]],
                signal[fiducialPoints[1]] - signal[fiducialPoints[2]],
                signal[fiducialPoints[8]] - signal[fiducialPoints[1]],
                signal[fiducialPoints[8]] - signal[fiducialPoints[2]],
                signal[fiducialPoints[0]] - signal[fiducialPoints[2]]]  # Initialize list to store features

    # Calculate features based on fiducial points and signal

    return features  # Return the list of calculated features


def feature_extraction(segments_, segments_labels):
    counter = 1
    Labels = []
    Features = []
    X = []
    Y = []
    for c in range(segments_.shape[0]):
        g = np.array(segments_[c, :])
        points = findFiducialPoints(g)
        if points is not None:
            if counter == 1:
                plt.figure(figsize=(12, 6))
                x = pd.DataFrame(g)
                index_ = x.index.to_numpy()
                signal_ = g
                # plt.plot(index_[:], signal_[:])
                # plt.title(f'segment:')
                # plt.xlabel('Sample')
                # plt.ylabel('Amplitude')
                # for i in range(len(points) - 1):
                #     L = (points[i])
                #     # if signal[L] > 0.2 or signal[L] < -0.3:
                #     Y.append(signal_[L])
                #     X.append(L)
                # plt.grid(True)
                # plt.rcParams['figure.constrained_layout.use'] = False
                # plt.tight_layout()
                # plt.plot(X, Y, "x")
                # plt.plot(index_[:], signal_[:])
                # plt.xlabel('Sample')
                # plt.ylabel('Amplitude')
                # plt.grid(True)
                # plt.rcParams['figure.constrained_layout.use'] = False
                # plt.tight_layout()
                # plt.show()
                counter -= 1
            features = findFeatures(points, g)
            Features.append()
            Labels.append(segments_labels[c])
    Features = np.array(Features)
    Labels = np.array(Labels)
    return Features, Labels



