from vllm import LLM, SamplingParams
from datasets import load_dataset
import util
import re
import json
invalid_outputs = []
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None
def extract_boxed_content(latex_string):
    pattern = r'\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}'
    match = re.findall(pattern, latex_string)
    if match:
        return match[-1]
    return None
def process_results(doc, completion, answer):
    split_ans = extract_boxed_content(completion)
    if split_ans is not None:
        # ans = split_ans[-1]
        # extract_ans_temp = ans.split('.\n')[0]
        # extract_ans_temp = extract_ans_temp.strip()
        # if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
        #     extract_ans = extract_ans_temp[0:-1]
        # else:
        #     extract_ans = extract_ans_temp
        extract_ans = split_ans.strip()
        print(extract_ans)
        if util.is_equiv(extract_ans, answer):
            # return (1, extract_ans)
            return True
        else:
            # return (0, extract_ans)
            return False 
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False
problem_prompt = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "Write your answer with format: ANSWER:\\boxed{{your answer here}}"
    "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
)
model = "your model path here"
dataset_path = r'AI-MO/aimo-validation-aime'
dataset = load_dataset(dataset_path, split='train')
problem = dataset['problem']

answer = dataset['answer']
problem_list = []
for p in problem:
    problem_prompt.format(instruction=p)
    problem_list.append(problem_prompt.format(instruction=p))
print(dataset)
print(problem_list)
sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=512)
llm = LLM(model=model,tensor_parallel_size=1)
completions = llm.generate(problem_list, sampling_params)
res_completions = []

for output in completions:
    prompt_temp = output.prompt
    generated_text = output.outputs[0].text
    res_completions.append(generated_text)
    
print(res_completions)
res_dict = []
tmp_answer_list = []

for idx, (problem, completion, prompt_answer) in enumerate(zip(problem_list, res_completions, answer)):
    res = process_results(problem, completion, prompt_answer)
    tmp_answer_list.append(res)
    ele = {
            # 'problem': problem,
            # 'subject': subject,
            'solution': completion,
            'answer': prompt_answer,
            'result': res
        }
    res_dict.append(ele)
# with open('output1.jsonl', 'w', encoding='utf-8') as f:
#     for item in res_dict:
#         json.dump(item, f, ensure_ascii=False)  # 将每个字典写入一行
#         f.write('\n')  # 每个 JSON 对象后加换行符
acc = sum(tmp_answer_list) / len(tmp_answer_list)
print(answer)
print('length====', len(tmp_answer_list), ', acc====', acc, 'correct_numbers===', sum(tmp_answer_list))



