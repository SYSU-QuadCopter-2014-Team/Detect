# Detect - An object detecting and tracking algorithm

本算法基于边缘查找，SVM和KCF跟踪算法。

本版本依赖于OpenCV 3

运行测试代码前准备：
准备样本集，提取hog特征并训练模型。
准备images.txt，样本集文件名列表，并置于./RunTracker 同一目录下。

本测试SVMModel基于voc2014 ball样本集。

编译及运行：
`cmake .`
`make`
`./RunTracker show`


