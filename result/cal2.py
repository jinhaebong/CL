import os
import json

def classify_and_calculate_metrics(testdata_path, results_path):

    # 加载测试数据
    with open(testdata_path, 'r') as f:
        testdata = json.load(f)

    # 加载模型结果
    with open(results_path, 'r') as f:
        results = json.load(f)

    # 初始化分类统计
    classification_counts = {
        "correct": 0,
        "wrong": 0,
        "refuse": 0,
        "overrefuse": 0,
        "correct rate": 0,
        "wrong rate": 0,
        "refuse rate": 0,
        "overrefuse rate": 0
    }

    # 初始化基准模型正确回答计数
    base_correct_count = 0

    # 遍历每个学科并统计
    for subject in testdata.keys():
        test_samples = testdata[subject]
        result_samples = results.get(subject, [])

        for datum, result in zip(test_samples, result_samples):
            kb = datum.get("kb") # 获取知识边界，默认为 Unknown
            model_decision = result[0]  # 新模型预测的结果
            answer_sureness = result[3]

            if kb == "Known": 
                base_correct_count += 1


            if answer_sureness > 0.5:  # 新模型回答正确
                if model_decision == 1:
                    classification_counts["correct"] += 1
                else:
                    classification_counts["wrong"] += 1
            elif answer_sureness < 0.5:
                if kb == "Known":
                    classification_counts["overrefuse"] += 1
                else:
                    classification_counts["refuse"] += 1

    # 计算指标
    metrics = calculate_metrics(classification_counts, base_correct_count)

    # 打印分类统计
    print("分类统计:")
    for key, value in classification_counts.items():
        print(f"{key}: {value}")

    # 打印指标结果
    print("\n指标计算:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def calculate_metrics(classification_counts, base_correct_count):

    correct = classification_counts.get("correct", 0)
    wrong = classification_counts.get("wrong", 0)
    refuse = classification_counts.get("refuse", 0)
    overrefuse = classification_counts.get("overrefuse", 0)

    # 总样本数
    total_samples = correct + wrong + refuse + overrefuse

    classification_counts["correct rate"] = correct/total_samples
    classification_counts["wrong rate"] = wrong/total_samples
    classification_counts["refuse rate"] = refuse/total_samples
    classification_counts["overrefuse rate"] = overrefuse/total_samples
    # Accuracy
    accuracy_correct_answers_only = (correct)/ (correct+wrong)

    accuracy_with_refusal =  (correct + refuse)/ total_samples 

    # Precision
    precision = correct / (correct + wrong + overrefuse) 

    # Recall
    recall = correct / (correct + overrefuse)

    # F1-Score
    f1 = (2 * precision * recall) / (precision + recall) 

    # Overrefuse Rate
    overrefuse_rate = overrefuse / base_correct_count 

    return {
        "Accuracy": round(accuracy_correct_answers_only, 4),
        "Accuracy with Refusal": round(accuracy_with_refusal,4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
        "Overrefuse Rate": round(overrefuse_rate, 4)
    }

# 主程序入口
if __name__ == "__main__":
    # 数据文件路径
    testdata_path = "../data/Qwen2-7B/split_data/id_test.json"
    # results_path = "2.2.2/Qwen2-7B/rait/result_id.json"
    results_path = "2.2.1/Qwen2-7B/rtune/result_id.json"

    classify_and_calculate_metrics(testdata_path, results_path)



