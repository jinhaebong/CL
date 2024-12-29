import json
import os

# 输入文件路径
input_file = "Qwen2-7B-Instruct/2.1/evaluated_MMLU.json"  
output_dir = "Qwen2-7B-Instruct/split_data"  # 输出目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载数据
with open(input_file, "r", encoding="utf-8") as f:
    evaluate_data = json.load(f)

# 处理 kb 字段，只保留 "Known" 或 "Unknown" 根据 Correctness
for subject, questions in evaluate_data.items():
    for question in questions:
        correctness_result = question["kb"]["Correctness"]["result"]
        question["kb"] = correctness_result

# 获取所有 subject
subjects = list(evaluate_data.keys())

# 前28个 subject 作为 in-domain，其余作为 out-of-domain
indomain_subjects = subjects[:28]
outofdomain_subjects = subjects[28:]

# 将数据按 subject 分类
indomain_data = {subject: evaluate_data[subject] for subject in indomain_subjects}
outofdomain_data = {subject: evaluate_data[subject] for subject in outofdomain_subjects}

# 按 subject 顺序分割 in-domain 数据
def split_data_by_subject(data):
    train_data = {}
    test_data = {}

    for subject, questions in data.items():
        midpoint = len(questions) // 2  # 找到一半的位置
        train_data[subject] = questions[:midpoint]  # 前一半作为训练集
        test_data[subject] = questions[midpoint:]   # 后一半作为测试集

    return train_data, test_data

indomain_train_data, indomain_test_data = split_data_by_subject(indomain_data)

# 保存数据
indomain_train_file = os.path.join(output_dir, "indomain_train.json")
indomain_test_file = os.path.join(output_dir, "indomain_test.json")
outofdomain_file = os.path.join(output_dir, "outofdomain.json")

with open(indomain_train_file, "w", encoding="utf-8") as f:
    json.dump(indomain_train_data, f, ensure_ascii=False, indent=4)
with open(indomain_test_file, "w", encoding="utf-8") as f:
    json.dump(indomain_test_data, f, ensure_ascii=False, indent=4)
with open(outofdomain_file, "w", encoding="utf-8") as f:
    json.dump(outofdomain_data, f, ensure_ascii=False, indent=4)

print(f"Saved in-domain train data to {indomain_train_file}")
print(f"Saved in-domain test data to {indomain_test_file}")
print(f"Saved out-of-domain data to {outofdomain_file}")
