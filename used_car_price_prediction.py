# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from google.colab import files

# Load dataset
# Ensure 'car data.csv' is uploaded to the Colab environment or specify the correct path.
# For example, if you mount Google Drive, the path might be '/content/drive/MyDrive/car data.csv'

try:
    df = pd.read_csv('car data.csv')
except FileNotFoundError:
    print("The file 'car data.csv' was not found. Please upload it.")
    uploaded = files.upload()
    if 'car data.csv' in uploaded:
        df = pd.read_csv('car data.csv')
        print("File uploaded successfully!")
    else:
        print("Upload failed or incorrect file name. Please ensure 'car data.csv' is uploaded.")
        raise FileNotFoundError("Could not load 'car data.csv' after upload attempt.")

# Feature Engineering
df["CarAge"] = 2024 - df["Year"]

# Convert text to numbers
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print("Predicted Prices:", predictions[:5])
