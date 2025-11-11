import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load your dataset
df = pd.read_csv("studper.csv")

# Show first few rows
print(df.head())
print(df.info())

# Separate features (inputs) and target (output)
X = df.drop(columns=["OUTCOME"])   # All columns except outcome
y = df["OUTCOME"]                  # Target column


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top Features Influencing Student Outcome")
plt.show()

import pickle

with open("student_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("student_model.pkl", "rb") as f:
    model = pickle.load(f)

# Suppose a new student record
new_data = pd.DataFrame([{
    "gender": 0,          # 0=male, 1=female
    "age": 17,
    "attendance": 4,
    "school_support": 5,
    "family_support": 4,
    "study_time": 3,
    "use_of_time": 5,
    "free_time": 2
}])

# One-hot encode new data (match training columns)
new_data = pd.get_dummies(new_data)
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# Predict outcome
pred = model.predict(new_data)
print("Predicted outcome:", "Good" if pred[0] == 1 else "Poor")
