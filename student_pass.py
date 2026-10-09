import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = {
    "study_hours": [2, 3, 5, 6, 4, 1, 7, 5],
    "attendance": [55, 65, 80, 90, 75, 40, 95, 85],
    "previous_marks": [35, 42, 65, 75, 55, 20, 88, 70],
    "result": [
        "Fail", "Fail", "Pass", "Pass",
        "Pass", "Fail", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

print("\n--- Student Pass Prediction ---")

hours = float(input("Study hours per day: "))
attendance = float(input("Attendance percentage: "))
marks = float(input("Previous marks: "))

student = pd.DataFrame([{
    "study_hours": hours,
    "attendance": attendance,
    "previous_marks": marks
}])

prediction = model.predict(student)

print("\nPredicted Result:", prediction[0])