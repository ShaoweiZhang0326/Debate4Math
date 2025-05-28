import json

def calculate_accuracy(jsonl_file):
    correct = 0
    total = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)  # 解析每一行的 JSON 数据
            answer = data.get('answer')  # 获取答案字段
            golden_answer = data.get('golden_answer')  # 获取标准答案字段
            
            if answer is not None and golden_answer is not None:
                total += 1
                if answer == golden_answer:  # 比较答案是否相等
                    correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

# 使用示例
jsonl_file = 'eval/eval.jsonl'  # 替换成实际文件路径
accuracy = calculate_accuracy(jsonl_file)
print(f'Accuracy: {accuracy:.4f}')