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

output_file = f"{output_dir}/{result}_categories.png"
os.makedirs(output_dir, exist_ok=True)

# 定义子类别和大类别映射
subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"]
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "Humanities": ["history", "philosophy", "law"],
    "Social Sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "Other (Business, Health, Misc.)": ["other", "business", "health"]
}

summary = defaultdict(lambda: {"Known": 0, "Unknown": 0})
for subject, questions in data.items():
    for question in questions:
        correctness = question["kb"]["Correctness"]["result"]
        for subcat in subcategories.get(subject, ["other"]):
            for cat, subcat_list in categories.items():
                if subcat in subcat_list:
                    summary[cat][correctness] += 1

keys = list(summary.keys())
values = [[summary[cat]["Known"], summary[cat]["Unknown"]] for cat in keys]
value_labels = ["Known", "Unknown"]

x = np.arange(len(keys))
width = 0.4

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#1f8142', '#FFA500']
bottom = np.zeros(len(keys))
for i, label in enumerate(value_labels):
    ax.bar(x, [v[i] for v in values], width, label=label, bottom=bottom, color=colors[i])
    bottom += [v[i] for v in values]

ax.set_xlabel('Categories')
ax.set_ylabel('Count')
ax.set_title(f'{result} - Category Summary')
ax.set_xticks(x)
ax.set_xticklabels(keys, rotation=45)
ax.legend()

plt.tight_layout()
plt.savefig(output_file)
plt.show()
