from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os

from torch.nn import DataParallel


def load_debater(debate_path, name):
    print(f"Loading {name}")
    model_debate = AutoModelForCausalLM.from_pretrained(debate_path)
    model_debate.config.use_cache = False
    model_debate.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(debate_path, padding=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model_debate, tokenizer

# ['overall_system', 'debater_system', 'judge_system', 'pre_debate', 
# 'pre_debate_judge', 'pre_opening_speech', 'pre_speech', 'pre_opponent_speech', 
# 'pre_previous_speech', 'pre_debater_a_speech_judge', 'pre_debater_b_speech_judge', 
# 'post_round_judge', 'post_round_judge_without_reasoning', 'judge_question_instructions', 
# 'pre_judge_questions', 'judge_decision', 'debater_scratchpad', 'previous_debater_scratchpad', 
# 'judge_decision_for_debater']

DEBATE_A_INIT = DEBATE_B_INIT = [
    'overall_system', # 提示整个游戏的规则
    'debater_system', # 提示debate的规则
    'pre_debate', # 提示debate的背景以及二者的观点和立场
    'pre_opening_speech', # 提示其中一个debater要做的事、自己和对方的观点
    'pre_speech', # 提示开始debate
    'pre_opponent_speech', # 提示对方说了什么
    'pre_previous_speech', # 提示之前自己说了什么
    'pre_judge_questions', # 提示法官向谁提问了什么问题
    'debater_scratchpad', # 要求给出<quote>
    'previous_debater_scratchpad', # 之前轮次生成的<quote>
]

JUDGE_INIT = [
    'judge_system', # 提示法官的规则
    'pre_debate_judge', # 提示debate的背景以及两个辩论者的观点和立场
    'pre_debater_a_speech_judge', # 提示法官debater A 之前所有说的话
    'pre_debater_b_speech_judge', # 提示法官debater B 之前所有说的话
    'post_round_judge', # 提示法官作出决定
    'post_round_judge_without_reasoning', # 提示法官给出二者可靠的分数
    'judge_question_instructions', # 提示法官向二者进行提问
    'judge_decision', # 法官给出的最终结果
    'judge_decision_for_debater' # 法官给出的最终结果
]



def __prepare_debate_round(data_dict):
    # 定义每一轮的正方、反方和法官的输入
    round_data = {
        'debater_a_prompts': [],
        'debater_b_prompts': [],
        'judge_prompts': []
    }

    # 为 debater_a 和 debater_b 按照指定的键列表构建 prompts
    for key in DEBATE_A_INIT:
        if key in data_dict['debater_a_prompts']['Debate Prompt']:
            prompt = data_dict['debater_a_prompts']['Debate Prompt'][key]
            if isinstance(prompt['content'], list):
                prompt['content'] = ' '.join(prompt['content'])
            round_data['debater_a_prompts'].append(prompt)

        if key in data_dict['debater_b_prompts']['Debate Prompt']:
            prompt = data_dict['debater_b_prompts']['Debate Prompt'][key]
            if isinstance(prompt['content'], list):
                prompt['content'] = ' '.join(prompt['content'])
            round_data['debater_b_prompts'].append(prompt)

    for key in JUDGE_INIT:
        if key in data_dict['debater_a_prompts']['Debate Prompt']:
            prompt = data_dict['debater_a_prompts']['Debate Prompt'][key]
            if isinstance(prompt['content'], list):
                prompt['content'] = ' '.join(prompt['content'])
            round_data['judge_prompts'].append(prompt)
    assert all(
        prompt is not None 
        for prompt in [round_data['debater_a_prompts'], round_data['debater_b_prompts'], round_data['judge_prompts']]
        )
    return round_data
    

def generate_debate_data_list(data_dict_list):
    debate_data_list = []
    for data_dict in tqdm(data_dict_list, desc='Processing data'):
        round_data = __prepare_debate_round(data_dict)
        debate_data_list.append(round_data)
    return debate_data_list


def __prepare_debater_ab_prompts(prompts, mapping):
    results = {name: [] for name in mapping.keys()}

    for list_name, indices in mapping.items():
        for i in indices:
            results[list_name].append(prompts[i])
    return results


def __get_response(
    dialogue_A, 
    dialogue_B,
    debater_A, 
    debater_B, 
    tokenizer_A, 
    tokenizer_B,
    max_new_tokens=512, 
    temperature=1.2, 
    top_p=0.9
    ):
    device_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    device_ids = [f"cuda:{i}" for i in device_ids]
    
    
    debater_A.to(device_ids[0])
    debater_B.to(device_ids[1])
    debater_A.eval()
    debater_B.eval()
    def generate_response(dialogue, model, tokenizer, device):
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in dialogue])
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        # inputs = tokenizer.apply_chat_template(
        #     dialogue,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # model_inputs=tokenizer([inputs],return_tensors="pt").to(device)
        # output 
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
        torch.cuda.empty_cache()
        return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    response_a = generate_response(dialogue_A, debater_A, tokenizer_A, device_ids[0])
    dialogue_B.append({'role': 'system', 'content': response_a})
    response_b = generate_response(dialogue_B, debater_B, tokenizer_B, device_ids[1])

    return {"response_a": response_a, "response_b": response_b}

    
    


def debate(
    debater_a, 
    debater_b, 
    tokenizer_a, 
    tokenizer_b, 
    prompt_data
):
    '''
    [
        0: 'overall_system', # 提示整个游戏的规则
        1: 'debater_system', # 提示debate的规则
        2: 'pre_debate', # 提示debate的背景以及二者的观点和立场
        3: 'pre_opening_speech', # 提示其中一个debater要做的事、自己和对方的观点
        4: 'pre_speech', # 提示开始debate
        5: 'pre_opponent_speech', # 提示对方说了什么
        6: 'pre_previous_speech', # 提示之前自己说了什么
        7: 'pre_judge_questions', # 提示法官向谁提问了什么问题
        8: 'debater_scratchpad', # 要求给出<quote>
        9: 'previous_debater_scratchpad', # 之前轮次生成的<quote>
    ]
    '''
    
    print(f"Initialized {len(prompt_data)} data")
    
    for debate_data in tqdm(prompt_data):
        debater_a_prompt = debate_data['debater_a_prompts']
        debater_b_prompt = debate_data['debater_b_prompts']
        judge_prompt = debate_data['judge_prompts']
        prompt_mapping = {
            'init': [0, 1, 2, 3, 4],
            'opponent_speech': [5],
            'previous_speech': [6],
            'judge_asked_questions': [7],
            'previous_debater_scratchpad': [9]
        }
        
        # debater_a_init = [] # 0, 1, 2, 3, 4, 8
        # debater_a_opponent_speech = [] # 5
        # debater_a_previous_speech = [] # 6
        # debater_a_judge_asked_questions = [] # 7
        # debater_a_previous_debater_scratchpad = [] # 9
        
        debater_a_prompt = debate_data['debater_a_prompts']
        debater_b_prompt = debate_data['debater_b_prompts']

        debater_a_results = __prepare_debater_ab_prompts(debater_a_prompt, prompt_mapping)
        debater_b_results = __prepare_debater_ab_prompts(debater_b_prompt, prompt_mapping)

        # 提取生成的列表，分别用于 debater_a 和 debater_b
        debater_a_init = debater_a_results['init']
        debater_a_opponent_speech = debater_a_results['opponent_speech']
        debater_a_previous_speech = debater_a_results['previous_speech']
        debater_a_judge_asked_questions = debater_a_results['judge_asked_questions']
        debater_a_previous_debater_scratchpad = debater_a_results['previous_debater_scratchpad']
        
        debater_b_init = debater_b_results['init']
        debater_b_opponent_speech = debater_b_results['opponent_speech']
        debater_b_previous_speech = debater_b_results['previous_speech']
        debater_b_judge_asked_questions = debater_b_results['judge_asked_questions']
        debater_b_previous_debater_scratchpad = debater_b_results['previous_debater_scratchpad']
        
        b_init_oppo = debater_b_init
        b_init_oppo.append(debater_b_opponent_speech[0])
        res = __get_response(
            dialogue_A=debater_a_init,
            # dialogue_B=debater_b_init,
            dialogue_B=b_init_oppo,
            debater_A=debater_a,
            debater_B=debater_b,
            tokenizer_A=tokenizer_a,
            tokenizer_B=tokenizer_b,
            )
        
        
        