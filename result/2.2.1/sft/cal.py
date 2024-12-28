import os
import json

with open(f"result_id.json", 'r') as f:
    data = json.load(f)

correct = 0
refuse = 0
overrefuse=0
total = 0
count= 0
for subject in data.keys():
    subject_samples = data[subject]
    total += len(subject_samples)
    for sample in subject_samples:
        if(sample[0]==1):
            correct +=1
            count+=1
        elif(sample[0]==2):
            refuse+=1
        else:
            count+=1

print(f"correct: {correct}")
print(f"accuracy: {correct/count}")
print(f"refuse: {refuse}")