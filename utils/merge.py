import re
import itertools
from tqdm import tqdm
import copy
import nltk
nltk.download('punkt')  # 下载必要的模型
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize
nltk.data.path.append('add your nltk/tokenizer/punkt data path if needed')

    
def extract_sentence(text):

    sentences = sent_tokenize(text)
    
    # 返回拆分后的句子
    return sentences


def replace_placeholders(data, replacements):
    """
    递归替换字典或列表中的占位符。
    """
    if isinstance(data, dict):
        return {key: replace_placeholders(value, replacements) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_placeholders(item, replacements) for item in data]
    elif isinstance(data, str):
        # 用正则替换所有占位符
        pattern = re.compile('|'.join(map(re.escape, replacements.keys())))
        return pattern.sub(lambda match: replacements[match.group(0)], data)
    else:
        return data


def merge_data_and_prompt(data, prompt):
    
    prompts_ = []
    # idx = 1
    idx = 0
    for item in tqdm(data, desc="merging data and prompt"):
        idx += 1
        
        question = item.get("problem", "No question provided")
        if question == "No question provided":
            question = item.get("question", "No question provided")
        solution = item.get("solution", "No solution provided")
        answer = item.get("answer", "No answer")
        subject = item.get("subject", "No subject")
        solution_sentence = extract_sentence(text=solution)
        cumulative_solution = ""
        for i, sentence in enumerate(solution_sentence):
            # print(idx)
            if i == 0:
                cumulative_solution = sentence
            else:
                cumulative_solution += " " + sentence

            current_solution = cumulative_solution if i > 0 else ""
            replacements = {
                "<PROBLEM>": question,
                "<RS>": current_solution,
                "<Single Step>": sentence,
                "<ANSWER>": answer
            }
            updated_data = replace_placeholders(prompt, replacements)
            
            rules = "\n".join(updated_data["Debate_Prompt"]["Rules"])
            
            
            debater_a = updated_data["Debate_Prompt"]["Debater_A"]
            debater_b = updated_data["Debate_Prompt"]["Debater_B"]
            judge = updated_data["Debate_Prompt"]["Judge"]["description"]
            
            prompt_ = {
                "idx": idx,
                "problem": question,
                "solution": solution,
                "answer": answer,
                "debate_step": sentence,
                "subject": subject,
                "debater_A": debater_a['description'],
                "debater_B": debater_b['description'],
                "judge": judge
            }
            
            prompts_.append(prompt_)
        
    return prompts_, rules
    
    