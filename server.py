import json

from flask import Flask, request, jsonify, render_template
import requests
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)
filename = './static/GenderLogistic.sav'
model = pickle.load(open(filename, 'rb'))
print(model)


def classify(model, X_test, y_test):
    print('Accuracy:', model.score(X_test, y_test))


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


# @app.route('/api', methods=['POST'])
# def api():
#     # api token TODO
#     input_x = request.json['input_X']
#     # 对输入参数 input_x 进行处理，并返回预测结果 pred
#     pred = process_input(input_x)
#     response = {'output_Y': pred}
#     return jsonify(response)
#     # # 进行模型预测
#     # output_Y = model.predict(input_X)
#     # return jsonify({'output_Y': output_Y.tolist()})
#
#
# def process_input(input_x):
#     # 在这里编写处理输入参数的代码
#     flag = input_x[0].upper()
#     if flag == 'X':
#         return 'X gender'
#     elif flag > 'M':
#         return 'male'
#     else:
#         return 'female'


# @app.route('/predict', methods=['POST'])
# def predict():
#     # 获取输入X
#     input_X = request.form.get('input_X')
#
#     # 发送POST请求到API端点，获取预测值Y
#     response = requests.post('http://localhost:5000/api', json={'input_X': input_X})
#     output_Y = response.json()['output_Y']
#     # print("response:")
#     # print(response)
#     # print("output_Y")
#     # print(output_Y)
#     # 返回预测值Y
#     return render_template('index.html', input_X=input_X, output_Y=output_Y)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    print(df)
    df = labelencoder(df.copy())
    # X_test = df.drop(columns=' Gender', axis=1)
    X_test = df
    y_pred = model.predict(X_test)
    print(y_pred)
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
    # print(df)
    df = labelencoder(df.copy())
    # X_test = df.drop(columns=' Gender', axis=1)
    X_test = df
    y_pred = model.predict(X_test)
    print(y_pred[0])

    return "Prediction: <insert prediction here>"


# 启动应用程序
if __name__ == '__main__':
    app.run(debug=True)
