import os
import tkinter as tk
import pickle
from tkinter import messagebox
import wfdb
from ECG_preprocessing import *

# Load the trained model
# with open(r'F:\8th semester\HCI\Labs\project\Models\X_test.pkl', 'rb') as f:
#     X = pickle.load(f)
# with open(r'F:\8th semester\HCI\Labs\project\Models\y_test.pkl', 'rb') as f:
#     Y = pickle.load(f)

# participant_data = {}
MainFolder = r"F:\8th semester\HCI\Labs\project\TestData"
# patients = []
# patient_files = [file for file in os.listdir(MainFolder)]
# for participant in patient_files:
#     patients.append(participant.split('.')[0])
#     participant_data[participant.split('.')[0]] = pd.read_csv(os.path.join(MainFolder, participant))
#
# order = 5
# fs = 1000  # sampling rate of 1000 Hz
# cutoff = 50  # Cutoff frequency in Hz
# cutoff_low = 1
#
# participant_data, segments = preprocessing(participant_data, cutoff_low, cutoff, fs, order)
#
# segments_array, Labels = prepare_segments_array(segments)
#
# X, Y = feature_extraction(segments_array, Labels)


# Function to perform login action based on model prediction

with open(r'F:\8th semester\HCI\Labs\project\Models\SVM.pkl', 'rb') as file:
    svm_classifier = pickle.load(file)


def read_personal_info(filePath):
    try:
        heaFile = [v for v in os.listdir(filePath) if ".hea" in v][0]
        record = wfdb.rdheader(os.path.join(filePath, heaFile[0:-4]))
        comments = record.comments
        # Check for keywords indicating absence of diagnosed cardiac conditions
        messagebox.showinfo("personal info", comments[0:10])
        print("done")
    except Exception as e:
        print("error")


def perform_login():
    file_read = False
    signal = {}
    username = username_entry.get()
    try:
        signal[username] = pd.read_csv(os.path.join(MainFolder, f"patient116.csv"))
        file_read = True
    except Exception as e:
        messagebox.showinfo("wrong username", "please check your user name")

    if file_read:
        # random = np.random.randint(0, X.shape[0])
        order = 5
        fs = 1000  # sampling rate of 1000 Hz
        cutoff = 50  # Cutoff frequency in Hz
        cutoff_low = 1

        signal, segments = preprocessing(signal, cutoff_low, cutoff, fs, order)

        segments_array, Labels = prepare_segments_array(segments)

        X, _ = feature_extraction(segments_array, Labels)

        y_pred = svm_classifier.predict(X)

        right = len(y_pred[y_pred == username])
        acc = right / len(y_pred)
        # Perform login action based on the predicted class
        if acc < 0.9:
            login_result_label.config(text="Login Failed. Unknown person.\n Please try again.", fg='red')
        else:
            PTB_folder = r"F:\ptb-diagnostic-ecg-database-1.0.0"
            login_result_label.config(text=f"Login Success. Welcome, {y_pred[0]}", fg='green')
            read_personal_info(os.path.join(PTB_folder, username))


# Create the Tkinter window
window = tk.Tk()
window.title("Login Form")

# Set window size and background color
window.geometry("500x250")
window.configure(bg='#f0f0f0')

# Configure colors and font
bg_color = "#98F5FF"  # Light gray background color
text_color = "#333333"  # Dark gray text color
button_color = "#4CAF50"  # Green button color
button_text_color = "white"  # White button text color

# Login button

# Create the login button with the custom style
username_label = tk.Label(window, text="Username:", bg='#f0f0f0')
username_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
username_entry = tk.Entry(window)
username_entry.grid(row=5, column=1, padx=10, pady=5)

login_result_label = tk.Label(window, text="", bg='#f0f0f0', font=('Helvetica', 16))
login_result_label.grid(row=0, column=0, columnspan=2, padx=10, pady=5)

login_button = tk.Button(window, text="Test", command=perform_login, bg=button_color, fg=button_text_color,
                         font=('Helvetica', 10, 'bold'))
login_button.grid(row=10, column=3, padx=10, pady=5)

# Styling the button


# Run the Tkinter event loop
window.mainloop()
