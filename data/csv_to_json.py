import os
import pandas as pd
import json

folder_path = "data_csv/dev"  
output_file = "original/dev.json" 

all_data = {}

for file_name in os.listdir(folder_path):
    subject = os.path.splitext(file_name)[0]
    subject = subject.replace('_dev', '').replace('_test', '')
    file_path = os.path.join(folder_path, file_name)

    df = pd.read_csv(file_path, header=None)

    subject_data = []
    for index, row in df.iterrows():
        question = row[0]  # 问题
        choices = [row[1], row[2], row[3], row[4]]  # 选项
        answer = row[5]  # 正确答案
        question_data = {
            "question": question,
            "choices": choices,
            "answer": answer,
            # "kb": {}  # 这里用于存储后续知识边界信息
        }
        subject_data.append(question_data)

    all_data[subject] = subject_data
sorted_data = {subject: all_data[subject] for subject in sorted(all_data.keys())}
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=4)

print(f"数据已合并并写入到 {output_file}")
