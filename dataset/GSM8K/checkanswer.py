import json
import os

# 读取并处理jsonl文件
def process_jsonl(input_file):
    output_file = os.path.splitext(input_file)[0] + '_has_answer.jsonl'
    
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 读取每一行并解析为字典
            data = json.loads(line)
            
            # 获取问题和答案
            question = data.get('question')  # 'problem'改成 'question' 因为示例中是'question'
            answer = data.get('answer', '')
            
            # 通过'####'分割答案
            if '####' in answer:
                solution, answer_text = answer.split('####', 1)  # 分割为solution和answer
            else:
                solution, answer_text = answer, ''  # 如果没有'####'，直接处理
            
            # 创建新的字段
            data['solution'] = solution.strip()  # 去除多余的空格
            data['answer'] = answer_text.strip()  # 去除多余的空格
            
            # 保存修改后的数据到新的文件
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"处理完成，文件已保存为: {output_file}")

# 示例：调用处理函数
input_file = 'dataset/GSM8K/GSM8K_train.jsonl'  # 替换为你的文件路径
process_jsonl(input_file)
