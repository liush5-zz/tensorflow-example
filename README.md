# tensorflow-example-by-MNIST
Two simple tensorflow examples:Predict handwriting images by using MNIST

#### 参考博客内容，尝试写了2个用tensorflow 解析手写数字图片的示例。体验了一下**tensorflow**框架。

* example1：   
    1.先用model1.py使用MNIST数据集训练并保存模型。

    2.在用predict1.py 解析手写数字图片并调用模型文件识别图片中的数字。

* example2：  
    1.先用model2.py使用MNIST数据集训练并保存模型。

    2.在用predict2.py 解析手写数字图片并调用模型文件识别图片中的数字。
    
#### 调试中遇到的主要问题：

>使用Spyder IDE调试代码是遇到“python kernel died”问题，导致调用模型时一直打印找不到参数的error，后在CMD中直接敲python命令运行代码ok。
