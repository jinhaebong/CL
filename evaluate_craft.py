
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
            prompt_str += " N. I don't know"
        prompt_str += f"\nAnswer: {item['answer']}\n\n"
    
    # 2) 拼接当前 sample
    prompt_str += "If you are not sure about the answer, respond with N.\n\n"
    prompt_str += sample["question"]
    for idx, opt in enumerate(sample["choices"]):
        prompt_str += f"\n{choices[idx]}. {opt}"
        prompt_str += " N. I don't know"
    prompt_str += "\nAnswer:"
    
    return prompt_str

def inference(tokenizer, model, sample, subject, prompt_data, device):
    """
    返回:
      pred_choice: 预测出的 'A'/'B'/'C'/'D'/'N'
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
    n_id = tokenizer("N").input_ids[0]

    # 只取这四个 logits 做 softmax
    selected_logits = torch.tensor([logits[a_id], logits[b_id], logits[c_id], logits[d_id], logits[n_id]], device=device)
    probs_5 = torch.softmax(selected_logits, dim=0).detach().cpu().numpy()
    # 选出最大概率下标
    pred_idx = np.argmax(probs_5)
    pred_choice = {0:"A",1:"B",2:"C",3:"D",4:"N"}[pred_idx]

    return pred_choice, probs_5, full_input

def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen2-7B")
    parser.add_argument('--domain', type=str, default="id",choices=["id","ood"])
    args = parser.parse_args()

    model_name = args.model
    model_pth = f"output_models/{model_name}-craft-2/merge"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_pth, use_fast=True,
        unk_token="<unk>", bos_token="<s>", eos_token="</s>", add_bos_token=False
    )
    model = AutoModelForCausalLM.from_pretrained(model_pth).to(device)

    with open(f"data/{model_name}/split_data/{args.domain}_test.json", 'r') as f:
        data = json.load(f)
    
    with open(f"data/MMLU/dev.json", 'r') as f:
        prompt = json.load(f)

    results = {}

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
            pred_choice, probs_4,full_input = inference(
                tokenizer=tokenizer,
                model=model,
                sample=sample,
                subject=subject,
                prompt_data=subject_prompt_data,
                device=device
            )

            
            # 2) correctness
            if pred_choice == correct_answer:
                correctness = 1
            elif pred_choice == "N":
                correctness = 2  # 拒绝
            else:
                correctness = 0  # 错误

            # 3) correctness
            correct_idx = {"A":0,"B":1,"C":2,"D":3}[correct_answer]
            answer_prob = probs_4[correct_idx]

            # 4) Certainty (基于熵, 熵越小越确定)
            ent = entropy(probs_4 + 1e-12)

            
            subject_data.append((correctness,float(answer_prob),float(ent)))
        results[subject] = subject_data
    

    # 保存评估结果
    save_dir = f"result/2.2.2/{model_name}/craft"
    save_file = f"result_{args.domain}.json"
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir,save_file )


    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Saved result => {output_file}")


if __name__ == "__main__":
    main()






