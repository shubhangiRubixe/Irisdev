import pickle
import sklearn
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app=Flask(__name__)
model=pickle.load(open('lr.pkl','rb'))

@app.route('/')
def homepage():
    return render_template('home.html')

## from postman
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(np.array(list(data.values())).reshape(1,-1))
    return jsonify(output[0])


@app.route('/predict',methods=['POST'])

def predict():
    data=[float(x)for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output1=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The predicted iris species is {}".format(output1))

if __name__=='__main__':
    app.run(debug=True)