import json
import os
from tqdm.auto import tqdm

def main():
    base_results = "Qwen2-7B/split_data/indomain_train.json"  
    finetuned_results = "Qwen2-7B/rait_1_finetune_results/indomain_train.json"
    # 创建输出目录

    # 加载数据
    with open(base_results, "r", encoding="utf-8") as f:
        base = json.load(f)

    RAIT = base

    with open(finetuned_results, "r", encoding="utf-8") as f:
        finetuned = json.load(f)


    challenging_data = {}
    correctness_stat = {}
    for base_subject, finetuned_subject in tqdm(zip(base.keys(), finetuned.keys()), desc="Subjects"):
        base_subject_samples = base[base_subject]
        finetuned_subject_samples = finetuned[finetuned_subject]
        subject_data = []
        correct_count = 0
        wrong_count = 0
        for base_sample, finetuned_sample in tqdm(zip(base_subject_samples, finetuned_subject_samples),  desc="Samples", leave=False):
            if(base_sample["kb"]["Correctness"]["result"]=="Unknown" and finetuned_sample["kb"]["Correctness"]["result"]=="Known"):
                continue
            else:
                subject_data.append(base_sample)

            if (base_sample["kb"]["Correctness"]["result"]=="Known"):
                correct_count += 1
            else:
                wrong_count += 1
        correctness_stat[base_subject] = {
            "Correct":correct_count,
            "Wrong":wrong_count
        }
        challenging_data[base_subject] = subject_data



    filter_by_entropy = {}
    for subject in tqdm(base.keys(), desc="Subjects"):
        subject_data = base[subject]
        vanilla = [x for x in subject_data if x["kb"]["Correctness"]["result"] == "Known"]
        idk = [x for x in subject_data if x["kb"]["Correctness"]["result"] == "Unknown"]

        vanilla_sorted = sorted(vanilla, key=lambda x: x["kb"]["Certainty"]["entropy"])

        idk_sorted = sorted(idk, key=lambda x: x["kb"]["Certainty"]["entropy"], reverse=True)

        n_van = int(0.75 * len(vanilla_sorted))
        n_idk = int(0.25 * len(idk_sorted))

        vanilla_selected = vanilla_sorted[:n_van]
        idk_selected = idk_sorted[-n_idk:]

        rait_subject_data = vanilla_selected + idk_selected
        
        filter_by_entropy[subject] = rait_subject_data

    final_questions = {}

    for subject in challenging_data.keys():
        challenging_questions = challenging_data[subject]
        filtered_questions = filter_by_entropy.get(subject, [])
        
        # Extract question texts for comparison
        challenging_question_texts = {q["question"] for q in challenging_questions}
        
        # Store the full question object if it exists in both lists
        final_questions[subject] = [
            question for question in filtered_questions
            if question["question"] in challenging_question_texts
        ]
    
    
    save_dir = "Qwen2-7B/traindata/cor-rait"
    save_file = "final_questions.json"
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir,save_file )


    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_questions, f, ensure_ascii=False, indent=4)
    print(f"Saved result => {output_file}")


if __name__ == "__main__":
    main()


    

            
    

