from utils.read_model import *
from utils.read_prompt import *
from utils.call_GPT import *
from utils.read_dataset import *
from utils.merge import *
from debate.debate import *
import os
import transformers
from transformers import *
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['ROOT_PATH'] = 'your root path here'  # Set your root path here
DEBATE_ROUNDS = 1
debate_data_file = os.environ['ROOT_PATH'] + 'debate_data/'

model_path = r'your model path here'  # Set your model path here
prompt_path = r'Prompt/prompt_demo.yaml'
dataset_name = r'MATH'
# dataset_name = r'GSM8K'
input_dataset = r'debate/MATH_500_test_files/llama3_output_false.jsonl'
save_path = os.path.basename(input_dataset)
print(f"Your data will be saved at {save_path}")
print("Loading prompt")
yaml_data = read_yaml(prompt_path)  

print("Loading dataset")
train_data = read_dataset(dataset_name, input_dataset)
train_data, debate_rules = merge_data_and_prompt(data=train_data, prompt=yaml_data)


if not os.path.exists(debate_data_file):
    os.mkdir(debate_data_file)
debate = Debate(
    debater_A_model_path=model_path,
    debater_B_model_path=model_path,
    rules=debate_rules,
    # Judge="gpt-3.5-turbo"
    Judge=model_path
)

debate.init_agents()

for idx, data_ in enumerate(tqdm(train_data, desc="Start debate")):
    debater_A_speech = data_['debater_A']
    debater_B_speech = data_['debater_B']
    judge = data_['judge']
    debate_step = data_['debate_step']    
    history = debate.debate_round(
        debater_A_user=debater_A_speech,
        debater_B_user=debater_B_speech,
        debate_step = debate_step,
        rounds=DEBATE_ROUNDS
    )
    
    debate_data = {
        'idx': data_['idx'],
        'MATH_PROBLEM': data_['problem'],
        'REASONING_STEP': data_['solution'],
        'ANSWER': data_['answer'],
        'DEBATE_STEP': data_['debate_step'],
        'speech_history': history['speech_history']
    }
    
    speech_history = debate_data['speech_history']
    speech_A = "\n".join(speech_history['Debater_A'])
    speech_B = "\n".join(speech_history['Debater_B'])
    
    judge_data = debate.judge_decision(
        judge_template=judge,
        speech_history=speech_history
    )
    

    debate_data.update({'judge_decision': judge_data})
    # debate.save_to_jsonl(debate_data, "output_1000.jsonl")
    debate.save_to_jsonl(debate_data, save_path)
    
    
    
    

