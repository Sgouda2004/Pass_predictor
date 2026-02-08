import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(r"E:\data\Desktop\python\ML\students.csv")

# Split into input (X) and output (y)
X = data[["Hours_Studied", "Attendance (%)"]]
y = data["Pass"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Make a new prediction
new_data = [[5, 75]]  # 5 hours studied, 75% attendance
print("Predicted:", model.predict(new_data))

# Visualization
plt.scatter(data["Hours_Studied"], data["Attendance (%)"], c=data["Pass"], cmap='coolwarm', edgecolors='k')
plt.xlabel("Hours Studied")
plt.ylabel("Attendance (%)")
plt.title("Student Pass Prediction")
plt.colorbar(label='0 = Fail, 1 = Pass')  # ✅ fixed colorbar
plt.show()
import joblib
joblib.dump(model, "E:\data\Desktop\python\ML\student_model.pkl")
