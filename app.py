from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    attendance = float(request.form['attendance'])
    score = float(request.form['score'])
    homework = float(request.form['homework'])
    hours = float(request.form['hours'])
    discipline = float(request.form['discipline'])

    features = np.array([[attendance, score, homework, hours, discipline]])
    output = model.predict(features)[0]

    label_map = {2: "High", 1: "Medium", 0: "Low"}
    result = label_map[output]

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
