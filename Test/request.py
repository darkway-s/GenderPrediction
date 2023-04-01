import requests
import pandas as pd
import numpy as np

url = 'http://127.0.0.1:5000/api/predict'
data = pd.read_csv('test_data.csv')
del data['Unnamed: 8']
print("input data:")
print(data)
response = requests.post(url, json=data.to_dict(orient='records'))
print("raw_response:")
print(response.json())

res = response.json()['result']
# 将所有0替换为'female'，将所有1替换为'male'
gender_map = {0: 'female', 1: 'male'}
pretty_res = np.vectorize(gender_map.get)(res)
print('pretty res:')
print(pretty_res)
