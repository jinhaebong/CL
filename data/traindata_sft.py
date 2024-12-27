import json
import os
import random

input_file = "split_data/indomain_train.json"  
output_file = "traindata/sft/sft_indomain_train.json"  

# 加载推理完成的数据
with open(input_file, "r", encoding="utf-8") as f:
    inferred_data = json.load(f)

# 初始化输出数据结构
formatted_data = {"type": "text_only", "instances": []}

# 遍历推理完成的数据
for subject, questions in inferred_data.items():
    for question in questions:
        # 构建基础问题格式
        full_input = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
        full_input += f"Question: {question['question']}\n"
        full_input += "Choices: " + " ".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(question['choices'])]) + "\n"

        # 根据 kb 的值添加 sure/unsure
        if question["kb"] == "Known":
            full_input += f"Answer: {question['answer']}. "
        else:
            full_input += "Answer: N. "

        # 添加到实例中
        formatted_data["instances"].append({"text": full_input})

random.shuffle(formatted_data["instances"])

# 保存到输出文件
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=4)

print(f"Formatted data has been saved to {output_file}")