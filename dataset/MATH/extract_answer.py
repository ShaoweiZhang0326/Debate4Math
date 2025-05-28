import json
import re
import json

# 递归正则表达式提取 \boxed{...} 中的内容
def extract_boxed_content(latex_string):
    # 匹配 \boxed{...} 的内容，支持嵌套的 {}
    pattern = r'\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}'
    match = re.search(pattern, latex_string)
    if match:
        return match.group(1)
    return None

# 读取jsonl文件并处理每一行
def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)  # 解析每一行的JSON数据
            solution = data.get('solution', '')  # 获取solution字段
            subject = data.get('type', 'NO sub')
            answer = extract_boxed_content(solution)  # 提取boxed中的内容

            if answer:  # 如果成功提取到answer
                data['answer'] = answer  # 添加answer字段
            
            # data['subject'] = subject
            element = {
                'problem': data['problem'],
                'solution': solution,
                'answer': answer,
                'subject': subject
            }
            # 将处理后的数据写入新的文件
            outfile.write(json.dumps(element, ensure_ascii=False) + '\n')

input_file = 'dataset/MATH/MATH_train.jsonl'
output_file = 'dataset/MATH/MATH_train_has_ans_subject.jsonl'

process_jsonl(input_file, output_file)