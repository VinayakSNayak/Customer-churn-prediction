import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = pd.read_csv("churn.csv")

# Preview
print("\nDataset Preview:\n", data.head())

# ✅ Convert Churn (Yes/No → 1/0)
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# ✅ Fix TotalCharges (string → float)
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')

# ✅ Drop missing values
data = data.dropna()

# ✅ Select features
X = data[["tenure", "MonthlyCharges", "TotalCharges"]]
y = data["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model (Pro level)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Feature Importance
print("\nFeature Importance:")
for i, col in enumerate(X.columns):
    print(f"{col}: {model.feature_importances_[i]}")

# 📊 Visualization
plt.figure()
plt.scatter(data["tenure"], data["MonthlyCharges"])
plt.xlabel("Tenure")
plt.ylabel("Monthly Charges")
plt.title("Customer Distribution")
plt.show()

# 🔥 User Input
print("\n--- Predict New Customer ---")
tenure = int(input("Enter Tenure (months): "))
charges = float(input("Enter Monthly Charges: "))
total = float(input("Enter Total Charges: "))

input_data = pd.DataFrame([[tenure, charges, total]],
                          columns=["tenure", "MonthlyCharges", "TotalCharges"])

prediction = model.predict(input_data)

if prediction[0] == 1:
    print("Prediction: Customer may leave ❌")
else:
    print("Prediction: Customer will stay ✅")