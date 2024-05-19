import flask
from flask import request, render_template, Flask
import pickle

# Import saved files
model = pickle.load(open('logreg.pkl', 'rb'))
scaler = pickle.load(open('scaler (1).sav', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        # Get the input values
        pl = float(request.form.get('petal_length'))
        sl = float(request.form.get('sepal_length'))
        pw = float(request.form.get('petal_width'))
        sw = float(request.form.get('sepal_width'))

        # Preprocess the input values
        scaled = scaler.transform([[sl, sw, pl, pw]])

        # Make the prediction
        pred = model.predict(scaled)

        # Return the prediction
        if pred==0:
            prediction='setosa'
        elif pred==1:
            prediction='versicolor'
        else:
            prediction='virginica'
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)