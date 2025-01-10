import matplotlib.pyplot as plt
import json
import numpy as np
from collections import defaultdict
import os


with open("../result/2.1/Qwen2-7B/evaluated_MMLU.json", "r", encoding="utf-8") as f:
# with open("../result/2.1/Qwen2-7B-Instruct/evaluated_MMLU.json", "r", encoding="utf-8") as f:
    data = json.load(f)

result = "Qwen2-7B"
output_dir = "Qwen2-7B" 

# result = "Qwen2-7B-Instruct"
# output_dir = "Qwen2-7B-Instruct" 

output_file = f"{output_dir}/{result}_subcategories.png"  
os.makedirs(output_dir, exist_ok=True)



summary = defaultdict(lambda: {"Known": 0, "Unknown": 0})
for subject, questions in data.items():
    for question in questions:
        correctness = question["kb"]["Correctness"]["result"]
        summary[subject][correctness] += 1

keys = list(summary.keys())
values = [[summary[subject]["Known"], summary[subject]["Unknown"]] for subject in keys]
value_labels = ["Known", "Unknown"]


x = np.arange(len(keys))
width = 0.8  

fig, ax = plt.subplots(figsize=(12, 8))

colors = ['#1f8142', '#FFA500']  
bottom = np.zeros(len(keys))
for i, label in enumerate(value_labels):
    ax.bar(x, [v[i] for v in values], width, label=label, bottom=bottom, color=colors[i])
    bottom += [v[i] for v in values]


ax.set_xlabel('Subjects')
ax.set_ylabel('Count')
ax.set_title(f'{result}')  
ax.set_xticks(x)  
ax.set_xticklabels(keys, rotation=270) 

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig(output_file)
plt.show()
