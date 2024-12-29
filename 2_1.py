
import os
import json
import logging
import numpy as np
from argparse import ArgumentParser
from tqdm.auto import tqdm
from scipy.stats import entropy

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger("transformers").setLevel(logging.ERROR)

choices = ["A", "B", "C", "D"]

def gen_prompt(sample, subject, prompt_data):
    """
    sample: {
       "question": str,
       "choices": [str, str, str, str],
       "answer": "A"/"B"/"C"/"D"
    }
    """
    # 1) 先处理“shots”的示例拼接
    prompt_str = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    
    for item in prompt_data:
        prompt_str += item["question"]
        for idx, opt in enumerate(item["choices"]):
            prompt_str += f"\n{choices[idx]}. {opt}"
        prompt_str += f"\nAnswer: {item['answer']}\n\n"
    
    # 2) 拼接当前 sample
    prompt_str += sample["question"]
    for idx, opt in enumerate(sample["choices"]):
        prompt_str += f"\n{choices[idx]}. {opt}"
    prompt_str += "\nAnswer:"
    
    return prompt_str

def inference(tokenizer, model, sample, subject, prompt_data, device):
    """
    返回:
      pred_choice: 预测出的 'A'/'B'/'C'/'D'
      probs:       
    """
    # 构造 prompt
    full_input = gen_prompt(sample, subject, prompt_data)

    inputs = tokenizer(full_input, return_tensors="pt").to(device)
    ids = inputs['input_ids']
    length = ids.shape[1]

    # 生成
    outputs = model.generate(
        ids,
        max_new_tokens=1,
        output_scores=True,
        return_dict_in_generate=True
    )
    # logits shape: [vocab_size]
    logits = outputs["scores"][0][0]
    
    # 拿到 "A"/"B"/"C"/"D" 在词表里的token_id 
    a_id = tokenizer("A").input_ids[0]
    b_id = tokenizer("B").input_ids[0]
    c_id = tokenizer("C").input_ids[0]
    d_id = tokenizer("D").input_ids[0]
    
    # 只取这四个 logits 做 softmax
    selected_logits = torch.tensor([logits[a_id], logits[b_id], logits[c_id], logits[d_id]], device=device)
    probs_4 = torch.nn.functional.softmax(selected_logits, dim=0).detach().cpu().numpy()
    # 选出最大概率下标
    pred_idx = np.argmax(probs_4)
    pred_choice = {0:"A",1:"B",2:"C",3:"D"}[pred_idx]

    return pred_choice, probs_4

def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default="/home/LLMs/Qwen2-7B-Instruct")
    args = parser.parse_args()
    

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True,
        unk_token="<unk>", bos_token="<s>", eos_token="</s>", add_bos_token=False
    )
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    
    with open(f"data/MMLU/test.json", 'r') as f:
        data = json.load(f)
    
    with open(f"data/MMLU/dev.json", 'r') as f:
        prompt = json.load(f)


    # 准备存评估信息
    evaluate_data = {}
    correct_count = 0
    confidence_count = 0
    certainty_count = 0
    
    CONFIDENCE_THRESHOLD = 0.25
    ENTROPY_THRESHOLD = 1

    for subject in tqdm(data.keys(), desc="Subjects"):
        subject_samples = data[subject]  # 列表 of {question, choices, answer}
        
        # 这个 subject 的 few-shot prompt
        # 如果 prompt_data_dict.get(subject, []) 是空，说明没相应示例，也可以给个空list
        subject_prompt_data = prompt.get(subject, [])
        subject_data = []
        for sample in tqdm(subject_samples, desc="Samples", leave=False):
            # sample: { "question":..., "choices": [...], "answer": "A"/"B"/"C"/"D" }
            correct_answer = sample["answer"]

            # 1) 做推理
            pred_choice, probs_4 = inference(
                tokenizer=tokenizer,
                model=model,
                sample=sample,
                subject=subject,
                prompt_data=subject_prompt_data,
                device=device
            )

            # 2) Correctness
            correctness = "Known" if (pred_choice == correct_answer) else "Unknown"

            # 3) Confidence
            answer_prob = np.max(probs_4)
            confidence = "Known" if (answer_prob > CONFIDENCE_THRESHOLD and correctness=="Known") else "Unknown"

            # 4) Certainty (基于熵, 熵越小越确定)
            ent = entropy(probs_4 + 1e-12)
            certainty = "Known" if (ent < ENTROPY_THRESHOLD and correctness=="Known") else "Unknown"

            # 统计计数
            if correctness == "Known":
                correct_count += 1
            if confidence == "Known":
                confidence_count += 1
            if certainty == "Known":
                certainty_count += 1
            
            # 记录
            sample_eval = {
                # "subject": subject,
                "question": sample["question"],
                "choices": sample["choices"],
                "answer": correct_answer,
                "predicted_answer": pred_choice,
                "kb": {
                    "Correctness": {
                        "result": correctness,
                    },
                    "Confidence": {
                        "result": confidence,
                        "answer_prob": float(answer_prob),
                    },
                    "Certainty": {
                        "result": certainty,
                        "entropy": float(ent),
                    }
                }
            }
            subject_data.append(sample_eval)
        evaluate_data[subject] = subject_data
    
    # 打印统计
    print("=== Evaluation Result ===")
    total_samples = len(evaluate_data)
    print(f"Total samples: {total_samples}")
    print(f"Correctness (Known):  {correct_count}")
    print(f"Confidence (Known):   {confidence_count}")
    print(f"Certainty (Known):    {certainty_count}")

    # 保存评估结果
    save_dir = "data/Qwen2-7B-Instruct/2.1"
    save_file = "evaluated_MMLU.json"
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir,save_file )


    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluate_data, f, ensure_ascii=False, indent=4)
    print(f"Saved result => {output_file}")


if __name__ == "__main__":
    main()
