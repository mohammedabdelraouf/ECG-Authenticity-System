import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score

from ECG_preprocessing import *

# Load features from the pickle file
with open(r'F:\8th semester\HCI\Labs\project\segments.pkl', 'rb') as file:
    X = pickle.load(file)

# Load labels from the pickle file
with open(r'F:\8th semester\HCI\Labs\project\Labels.pkl', 'rb') as file:
    y = pickle.load(file)

print(X.shape)
print(y.shape)

X_train_seg, X_test_seg, y_train_seg, y_test_seg = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train, y_train = feature_extraction(X_train_seg, y_train_seg)
print(X_train.shape)
print(y_train.shape)

# Create an SVM classifier
svm_classifier = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)

# Train the classifier on the training data
svm_classifier.fit(X_train, y_train)

X_test, y_test = feature_extraction(X_test_seg, y_test_seg)

# Make predictions on the testing data
y_pred = svm_classifier.predict(X_test)

y_pred_prob = svm_classifier.predict_proba(X_test)

for i in range(y_pred_prob.shape[0]):
    if not y_pred_prob[i].any() > 0.9:
        y_pred[i] = 'unknown'

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = dt_classifier.predict(X_test)

# Evaluate the classifier
Rate = accuracy_score(y_test, y_pred)
print(f"Accuracy: {Rate}")

# from sklearn.neighbors import KNeighborsClassifier
# knn_classifier = KNeighborsClassifier(n_neighbors=4)
#
# # Train the classifier on the training data
# knn_classifier.fit(X_train, y_train)
#
# # Make predictions on the testing data
# y_pred = knn_classifier.predict(X_test)
#
# # Evaluate the classifier
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")


# with open('/content/drive/MyDrive/ECG_Project/Models/SVM.pkl', 'wb') as f:
#     pickle.dump(svm_classifier, f)
# with open('/content/drive/MyDrive/ECG_Project/Models/knn_classifier.pkl', 'wb') as f:
#     pickle.dump(knn_classifier, f)
# with open('/content/drive/MyDrive/ECG_Project/Models/dt_classifier.pkl', 'wb') as f:
#     pickle.dump(dt_classifier, f)
#
#
# with open('/content/drive/MyDrive/ECG_Project/Models/X_test.pkl', 'wb') as f:
#     pickle.dump(X_test_seg, f)
# with open('/content/drive/MyDrive/ECG_Project/Models/y_test.pkl', 'wb') as f:
#     pickle.dump(y_test_seg, f)
