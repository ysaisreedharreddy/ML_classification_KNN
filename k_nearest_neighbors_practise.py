# Import necessary libraries for data handling and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset from the specified path into a DataFrame
ds = pd.read_csv(r"C:\Users\prasu\DS2\git\classification\3. KNN\Social_Network_Ads.csv")

# Extract features (Age and Estimated Salary) from the dataset to form the feature matrix X
# and the target variable (Purchased) to form the target vector y
X = ds.iloc[:, 2:4].values  # Age and Estimated Salary are typically found at these indexes
y = ds.iloc[:, -1].values   # The last column is often the target

# Split the dataset into training and testing sets using sklearn's train_test_split function
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the feature data to normalize distributions and improve the performance of the algorithm
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # Fit and transform the training data
X_test_s = scaler.transform(X_test)        # Transform the test data based on the training fit

# Initialize and train the K-Nearest Neighbors classifier with the training data
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=4, p=1)  # p=1 indicates Manhattan distance
classifier.fit(X_train_s, y_train)

# Predict the test set results and evaluate the model
y_pred = classifier.predict(X_test_s)

# Calculate and print the bias (training accuracy)
bias = classifier.score(X_train_s, y_train)
print('Bias (Training Accuracy):', bias)

# Calculate and print the variance (test accuracy)
variance = classifier.score(X_test_s, y_test)
print('Variance (Test Accuracy):', variance)

# Calculate the overall accuracy of the model on the test set
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print('Accuracy:', ac)

# Generate and display the confusion matrix to evaluate the classifier's performance
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)

# Generate a classification report to assess the model's performance in more detail
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print('Classification Report:\n', report)
