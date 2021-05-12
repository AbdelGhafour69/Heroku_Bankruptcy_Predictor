import pickle

import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    pour l'affichage sur html
    '''

    features = request.form.to_dict()
    features = list(features.values())
    features = list(map(float, features))
    print(features)
    final_features = np.array(features).reshape(1, 10)
    prediction = model.predict(final_features)

    #select = request.form.get('category')
    output = round(prediction[0], 2)
    output = 'Not bankrupt' if output == 0 else 'Bankrupt'
    return render_template('index.html', prediction_text='Prediction: '+output)


if __name__ == "__main__":
    app.run(debug=True)
