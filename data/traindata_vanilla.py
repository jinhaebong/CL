import json
import os
import random

input_file = "Qwen2-7B-Instruct/split_data/indomain_train.json"  
prompt_file = "MMLU/dev.json"
output_file = "Qwen2-7B-Instruct/traindata/vanilla/vanilla_indomain_train.json"  

# 加载推理完成的数据
with open(input_file, "r", encoding="utf-8") as f:
    inferred_data = json.load(f)

with open(prompt_file, "r", encoding="utf-8") as f:
    prompt_data = json.load(f)

# 初始化输出数据结构
formatted_data = {"type": "text_only", "instances": []}

# 遍历推理完成的数据
for subject, questions in inferred_data.items():
    # 获取当前 subject 的 few-shot 示例
    subject_prompt_data = prompt_data.get(subject, [])
    
    for question in questions:
        # 构建基础问题格式
        full_input = f"The following are multiple choice questions (with answers) about {subject}.\n\n"

        # 加入 few-shot 示例
        for prompt_sample in subject_prompt_data:
            full_input += f"Question: {prompt_sample['question']}\n"
            full_input += "Choices: " + " ".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(prompt_sample['choices'])]) + "\n"
            full_input += f"Answer: {prompt_sample['answer']}.\n\n"
            
        full_input += f"Question: {question['question']}\n"
        full_input += "Choices: " + " ".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(question['choices'])]) + "\n"
        full_input += f"Answer: {question['answer']}. "

        # 添加到实例中
        formatted_data["instances"].append({"text": full_input})

random.shuffle(formatted_data["instances"])

# 保存到输出文件
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=4)

print(f"Formatted data has been saved to {output_file}")