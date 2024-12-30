# 2.1 探知知识边界
* data/MMLU 是原始的数据集 其中dev是用于样例的prompt，test是测试数据集
* 执行2.1.py生成对所有测试数据集的评估，保存在data/model_name/2.1下
* 在2.1生成的初步评估数据一分为3，id_test，ood_test，id_train,用于2.2
* 3个traindata文件用于生成符合格式的训练数据，用于LMFLOW微调
# 2.2.1 学会拒绝

# 2.2.2 避免过度拒绝
