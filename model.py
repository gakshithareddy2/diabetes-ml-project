import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset (make sure you have this CSV file)
df = pd.read_csv("diabetes.csv")

print("Dataset Loaded Successfully!")
print(f"Shape: {df.shape}")

# Split features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model Training Completed!")

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# -------- USER INPUT --------
print("\n---- Enter Patient Medical Details ----")

pregnancies = int(input("Enter Pregnancies: "))
glucose = int(input("Enter Glucose: "))
blood_pressure = int(input("Enter BloodPressure: "))
skin_thickness = int(input("Enter SkinThickness: "))
insulin = int(input("Enter Insulin: "))
bmi = float(input("Enter BMI: "))
dpf = float(input("Enter DiabetesPedigreeFunction: "))
age = int(input("Enter Age: "))

# Create input dataframe
user_data = pd.DataFrame([[
    pregnancies, glucose, blood_pressure,
    skin_thickness, insulin, bmi, dpf, age
]], columns=X.columns)

# Prediction
prediction = model.predict(user_data)

print("\n---- Prediction Result ----")

if prediction[0] == 1:
    print("⚠️ Diabetes Detected")
else:
    print("✅ No Diabetes Detected")