import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


iris_data = "IRIS.csv"

# Load the dataset provided by the user
df = pd.read_csv(iris_data)



# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display the last 5 rows
print("\nLast 5 rows of the dataset:")
print(df.tail())


# Encode the target variable

le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])

# Data visualisation

scatter_matrix = pd.plotting.scatter_matrix(df.iloc[:, :-2], c=df['species_encoded'], figsize=(12, 12), marker='o', hist_kwds={'bins': 20}, cmap='viridis')
plt.suptitle('Pairplot-like Visualization of Iris Features', fontsize=16)
plt.show()


# Split the data into training and testing sets
X = df.iloc[:, :-2]
y = df['species_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=df['species'].unique())

print("\nAccuracy:", accuracy*100)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)


