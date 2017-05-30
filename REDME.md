参考博客内容，写了2个用tensorflow 解析手写数字图片的示例。体验了一下tensorflow框架。

example1：
    1.先用example1_create_model1.py使用MNIST数据集训练并保存模型。
    2.在用example1_predict_num1.py 解析手写数字图片并调用模型文件识别图片中的数字。

example2：
    1.先用example2_create_model2.py使用MNIST数据集训练并保存模型。
    2.在用example2_predict_num2.py 解析手写数字图片并调用模型文件识别图片中的数字。
    
调试中遇到的主要问题：
    使用Spyder IDE调试代码是遇到“python kernel died”问题，导致掉用模型式一直打印找不到参数的error，后在CMD中直接敲python命令运行代码ok。

