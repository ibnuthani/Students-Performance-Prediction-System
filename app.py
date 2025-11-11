from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("student_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    values = [float(x) for x in request.form.values()]
    final_input = np.array(values).reshape(1, -1)
    prediction = model.predict(final_input)
    result = 'Good' if prediction[0] == 1 else 'Poor'
    return render_template('index.html', prediction_text=f"Student performance: {result}")

if __name__ == "__main__":
    app.run(debug=True)
