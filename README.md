# 计算语言学大作业：探索大语言模型的知识边界

## 环境配置
我们小组参考了rtune的微调方式选择使用LMFlow，安装方式如下：
```sh
git clone -b v0.0.9 https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n kb-lmflow python=3.9 -y
conda activate kb-lmflow
conda install mpi4py -y
pip install -e .
```
## 目录结构
* data目录下存放了训练和测试用的数据集，2.1中的结果，还有微调所需要的数据。目录中代码文件是用于分割和改变格式用的。
* result目录下存放了2.2的结果
* 主目录下的文件是用于推理的文件

## 2.1
选择Qwen2-7B和Qwen2-7B-Instruct作为本实验的模型进行实验。
分别执行 2_1.py 可以得到两个模型分别在MMLU数据集中的表现。

## 2.2.1
使用2.1中生成的结果，分割数据集为 in-domain_train,in-domain_test,out-of-domain_test。

各个方法均使用in-domain_train作为训练集微调，并在in-domain和out-of-domain的测试集进行评估。

## 2.2.2
采用CRAFT方式解决模型过度拒绝的问题
