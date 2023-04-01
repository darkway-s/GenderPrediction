from flask import Flask, request, jsonify, render_template
import requests
import pickle
from sklearn.preprocessing import LabelEncoder

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
    return render_template('index.html')


@app.route('/api', methods=['POST'])
def api():
    # api token TODO
    input_x = request.json['input_X']
    # 对输入参数 input_x 进行处理，并返回预测结果 pred
    pred = process_input(input_x)
    response = {'output_Y': pred}
    return jsonify(response)
    # # 进行模型预测
    # output_Y = model.predict(input_X)
    # return jsonify({'output_Y': output_Y.tolist()})


def process_input(input_x):
    # 在这里编写处理输入参数的代码
    flag = input_x[0].upper()
    if flag == 'X':
        return 'X gender'
    elif flag > 'M':
        return 'male'
    else:
        return 'female'


@app.route('/predict', methods=['POST'])
def predict():
    # 获取输入X
    input_X = request.form.get('input_X')

    # 发送POST请求到API端点，获取预测值Y
    response = requests.post('http://localhost:5000/api', json={'input_X': input_X})
    output_Y = response.json()['output_Y']
    # print("response:")
    # print(response)
    # print("output_Y")
    # print(output_Y)
    # 返回预测值Y
    return render_template('index.html', input_X=input_X, output_Y=output_Y)


# 启动应用程序
if __name__ == '__main__':
    app.run(debug=True)
