import json

import numpy as np
from flask import Flask, request, jsonify, render_template
import requests
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)
filename = './static/GenderLogistic.sav'
model = pickle.load(open(filename, 'rb'))
# print(model)


def labelencoder(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df


# 处理GET请求，返回表单以输入X
@app.route('/')
def home():
    return render_template('DataInput.html')


@app.route("/api/predict", methods=["POST"])
def api_predict():
    # api token TODO
    data = request.get_json()
    df = pd.DataFrame(data)
    # print(df)
    df = labelencoder(df.copy())
    # X_test = df.drop(columns=' Gender', axis=1)
    X_test = df
    y_pred = model.predict(X_test)
    # print(y_pred)
    res = y_pred.tolist()

    return jsonify({'result': res})


@app.route("/pred", methods=["POST"])
def pred():
    # gender = request.form["gender"]
    age = request.form["age"]
    height = request.form["height"]
    weight = request.form["weight"]
    occupation = request.form["occupation"]
    education = request.form["education"]
    marital = request.form["marital"]
    income = request.form["income"]
    color = request.form["color"]

    # Do something with the data
    data = {
        # " Gender": gender,
        " Age": age,
        " Height (cm)": height,
        " Weight (kg)": weight,
        " Occupation": ' ' + occupation,
        " Education Level": ' ' + education,
        " Marital Status": ' ' + marital,
        " Income (USD)": income,
        " Favorite Color": ' ' + color
    }
    df = pd.DataFrame(data, index=[0])
    print(df)
    df.to_csv('output.csv', index=False)

    data2 = pd.read_csv('output.csv')
    # del data2['Unnamed: 8']
    # print("input data2:")
    # print(data2)
    df = data2
    url = 'http://127.0.0.1:5000/api/predict'
    response = requests.post(url, json=df.to_dict(orient='records'))
    res = response.json()['result']
    print(res[0])
    # 将所有0替换为'female'，将所有1替换为'male'
    gender_map = {0: 'female', 1: 'male'}
    pretty_res = np.vectorize(gender_map.get)(res)

    return render_template('result.html',
                           prediction=pretty_res[0],
                           age=age,
                           height=height,
                           weight=weight,
                           occupation=occupation,
                           education=education,
                           marital_status=marital,
                           income=income,
                           color=color)


# 启动应用程序
if __name__ == '__main__':
    app.run(debug=True)
