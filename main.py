# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
from sklearn.datasets import load_iris
data = load_iris()

# Create DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Show dataset preview
print("Dataset Preview:\n")
print(df.head())

# Split data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Split into training and testing sets (fixed random_state for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model (KNN with better k value)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print results
print("\nModel Accuracy:", accuracy)

# Confusion Matrix (bonus for marks)
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))