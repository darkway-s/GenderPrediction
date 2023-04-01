# Gender Prediction

该项目是一个性别预测模型的交互API。

## 如何运行该程序

1. 安装依赖库：

   ```
   pip install -r requirements.txt
   ```

2. 启动程序：

   ```
   python server.py
   ```

   然后在浏览器中打开 http://localhost:5000/ 即可访问。
   
## 如何测试程序

1. 进入Test文件夹

   ```
   cd Test
   ```

2. 启动测试程序:

   ```
   python request.py
   ```

## 文件说明

- `app.py`：程序入口文件，包含Web框架Flask的初始化和路由定义。
- `templates/index.html`：前端页面模板，使用Jinja2模板引擎渲染。
- `static/css/style.css`：前端页面的样式表。TODO
- `static/js/main.js`：前端页面的JavaScript脚本。TODO

## 主要函数说明

- `predict(data)`：

  描述：对输入数据进行预测。

  输入参数：

  - `data`：待预测的数据，类型为`numpy.ndarray`，形状为`(n_samples, n_features)`。

  输出：

  - 预测结果，类型为`numpy.ndarray`，形状为`(n_samples,)`。

- `load_model(model_path)`：

  描述：从指定路径加载训练好的模型。

  输入参数：

  - `model_path`：模型文件的路径，类型为字符串。

  输出：

  - 训练好的模型，类型为`sklearn.linear_model.LogisticRegression`。

## 参考资料

- Flask官方文档：http://flask.pocoo.org/docs/
- Scikit-learn官方文档：https://scikit-learn.org/stable/documentation.html