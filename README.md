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
* data目录下存放了数据和格式转换用的文件(用于LMflow进行微调)
* result目录下存放了2.1，2.2.1，2.2.2的结果和评估结果用的文件
* src目录下存放了各个方法测试所用的代码

# 2.1 探查⼤模型的“知识边界”
选择Qwen2-7B和Qwen2-7B-Instruct作为本实验的模型进行实验。
分别执行 2_1.py 可以得到两个模型分别在MMLU数据集中的表现。
```sh
cd src
python 2_1.py
```

结果存放在result/2.1

# 2.2 提升和应⽤⼤模型⾃我知识边界的探索能⼒
2.2.1和2.2.2实验我们选择使用Qwen2-7B进行。

## 2.2.1 让⼤模型在不知道的时候回答“不知道”
使用2.1中生成的结果，分割数据集为 in-domain_train,in-domain_test,out-of-domain_test存放在split_data文件夹下。
```sh
cd data
python split.py
```

各个方法均使用in-domain_train作为训练集微调，具体是通过各自的trandata_method.py文件讲训练集转化为符合LMFlow的格式保存在traindata文件夹里。

```sh
python traindata_vanilla.py
python traindata_rtune.py
python traindata_sft.py
```

后续使用LMflow进行微调，按方法更改训练集路径和模型路径。
```sh
cd ..
cd LMFlow
./scripts/run_finetune_with_lora.sh    --model_name_or_path /path/to/Qwen2-7B   --dataset_path  ../data/Qwen2-7B/traindata/sft    --output_lora_path  ../output_models/Qwen2-7B-sft/lora 

bash ./scripts/run_merge_lora.sh --model_name_or_path /path/to/Qwen2-7B --lora_model_path ../output_models/Qwen2-7B-sft/lora --output_model_path ../output_models/Qwen2-7B-sft/merge --device cpu
```

之后使用src目录下的评估文件在in-domain和out-of-domain的测试集进行测试。
```sh
cd ..
cd src
python evaluate_vanilla.py
python evaluate_vanilla.py --domain ood
python evaluate_rtune.py
python evaluate_rtune.py --domain ood
python evaluate_sft.py
python evaluate_sft.py --domain ood
```

## 2.2.2 避免知道的知识被错回复为“不知道”
采用RAIT方式解决模型过度拒绝的问题

通过下面方式生成rait训练数据集
```sh
cd src
uncorrectness.py 

cd ..
cd data
python cor_rait_dataset_selection.py 
python traindata_cor_rait.py
```
与2.2.1相同，使用LMFlow微调
```sh
cd ..
cd LMFlow
./scripts/run_finetune_with_lora.sh    --model_name_or_path /path/to/Qwen2-7B   --dataset_path  ../data/Qwen2-7B/traindata/cor-rait/train/rait_indomain_train.json    --output_lora_path  ../output_models/Qwen2-7B-cor-rait/lora 

bash ./scripts/run_merge_lora.sh --model_name_or_path /path/to/Qwen2-7B --lora_model_path ../output_models/Qwen2-7B-cor-rait/lora --output_model_path ../output_models/Qwen2-7B-cor-rait/merge --device cpu

```
随后进行测试
```sh
cd ..
cd src
python evaluate_cor_rait.py
```