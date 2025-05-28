import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import json
from tqdm import tqdm
import re

path = r'model path'
data_path = r'dataset/MATH/test_500_extract.jsonl'
pipeline = transformers.pipeline(
  "text-generation",
  model=path,
  model_kwargs={"torch_dtype": torch.bfloat16},
  device="auto",
)
ins_list = []
def extract_boxed_content(latex_string):
    # 匹配 \boxed{...} 的内容，支持嵌套的 {}
    pattern = r'\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}'
    match = re.findall(pattern, latex_string)
    if match:
        return match[-1]
    return None
with open(data_path, "r", encoding='utf-8') as f:
    for line in tqdm(f, desc="Reading dataset"):
        data = json.loads(line)
        instruction = data.get('problem', 'NONE')
        answer = data.get('answer')

        # Only include your final answer with \\boxed{{}}\n\n
        data_line = {'instruction': instruction, 'ground_truth': answer}
        ins_list.append(data_line)

new_data = []

output_path = r'eval/eval_500.jsonl'
with open(output_path, 'w', encoding='utf-8') as of:
    for data in tqdm(ins_list):
        Prompt_template = f"""
        Below is an instruction that describes a task. 
        Write a response that appropriately completes the request.
        Write your answer with format: ANSWER:\\boxed{{FINAL ANSWER}}
        ### Instruction:\n{data['instruction']}\n\n### Response: Let's think step by step.
        """
        outputs = pipeline(
            Prompt_template,
            max_new_tokens=256,
            temperature=0.1
        )
        answer = extract_boxed_content(outputs[0]["generated_text"])
        new_ = {
            "problem": data['instruction'],
            "solution": outputs[0]["generated_text"],
            "golden_answer": data['ground_truth'],
            "answer": answer
        }
        new_data.append(new_)
        of.write(json.dumps(new_, ensure_ascii=False) + '\n')

