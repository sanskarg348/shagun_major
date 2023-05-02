import joblib
from flask import request
from flask import Flask
import numpy as np

app = Flask(__name__)
dic = {20: 'rice',
 11: 'maize',
 3: 'chickpea',
 9: 'kidneybeans',
 18: 'pigeonpeas',
 13: 'mothbeans',
 14: 'mungbean',
 2: 'blackgram',
 10: 'lentil',
 19: 'pomegranate',
 1: 'banana',
 12: 'mango',
 7: 'grapes',
 21: 'watermelon',
 15: 'muskmelon',
 0: 'apple',
 16: 'orange',
 17: 'papaya',
 4: 'coconut',
 6: 'cotton',
 8: 'jute',
 5: 'coffee'}

@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/forest', methods=['GET', 'POST'])
def random_forest():
    d = np.array(list(map(float,request.form['data'].split(',')))).reshape(1,-1)
    print(d)
    loaded_rf = joblib.load("random_forest.joblib")

    return dic[int(loaded_rf.predict(d)[0])]


@app.route('/linear', methods=['GET', 'POST'])
def regression():
    d = list(map(int,request.form['data'].split(',')))
    loaded_lr = joblib.load("linear_regression.joblib")

    return dic[int(loaded_lr.predict(d)[0])]


if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()

