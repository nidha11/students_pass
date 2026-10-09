from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Historical student data
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

# Features and target
X = df[["study_hours", "attendance", "previous_marks"]]
y = df["result"]

# Train machine learning model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    student_name = ""

    if request.method == "POST":
        try:
            student_name = request.form["student_name"].strip()

            if not student_name:
                raise ValueError("Please enter the student's name.")

            hours = float(request.form["study_hours"])
            attendance = float(request.form["attendance"])
            marks = float(request.form["previous_marks"])

            if not 0 <= hours <= 24:
                raise ValueError(
                    "Study hours must be between 0 and 24."
                )

            if not 0 <= attendance <= 100:
                raise ValueError(
                    "Attendance must be between 0 and 100."
                )

            if not 0 <= marks <= 100:
                raise ValueError(
                    "Previous marks must be between 0 and 100."
                )

            # New student data
            student = pd.DataFrame([{
                "study_hours": hours,
                "attendance": attendance,
                "previous_marks": marks
            }])

            # Predict Pass or Fail
            prediction = model.predict(student)[0]

        except ValueError as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        student_name=student_name
    )


if __name__ == "__main__":
    app.run(debug=True)