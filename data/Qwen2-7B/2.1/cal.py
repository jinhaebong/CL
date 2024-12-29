import json
# 输入文件路径
input_file = "evaluated_MMLU.json"  # 假设生成的文件路径

# 加载数据
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)
correct = 0
total = 0

for subject in data.keys():
    subject_samples = data[subject]
    total += len(subject_samples)
    for sample in subject_samples:
        correctness_result = sample["kb"]["Correctness"]["result"]
        if correctness_result == "Known":
            correct +=1
print(f"Overall accuracy: {correct/total:.4f}")
