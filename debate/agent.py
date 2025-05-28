import torch
from utils.call_GPT import *
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, AutoModelForSequenceClassification, AutoConfig, LlamaForCausalLM, LlamaForSequenceClassification
from accelerate import Accelerator, dispatch_model, init_empty_weights, infer_auto_device_map
from vllm import LLM, SamplingParams
import time
accelerator = Accelerator()

class Agent:
    def __init__(
        self,
        name: str,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        llm,
        tokenizer
        ):
        self.name = name
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.llm = llm
        self.tokenizer = tokenizer
        self.dialog_template = []
        self.pre_speech = []
        self.task_type = ""
        self.pre_opponent = []
        stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
        self.sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=256, stop=stop_tokens)
        if "gpt" in self.model_name.lower():
            self.client = check_openai_network()
            if not self.client:
                exit()
        elif "qianwen" in self.model_name.lower():
            self.client = check_openai_network(model="qwen2.5-14b-instruct")
            if not self.client:
                exit()
        else:
            # self.model, self.tokenizer = self.load_debater(self.model_name, self.name)
            
            if llm == "":
                print("NO LLM, LOAD it")
                self.task_type = "CasualLM"
                
                # self.model, self.tokenizer = self.load_debater(self.model_name, self.name)
                # device_map = infer_auto_device_map(self.model)
                # self.model = dispatch_model(self.model, device_map=device_map)
                self.llm = True
                self.model = LLM(model=self.model_name)
                self.tokenizer = ""
            else:
                print("HAVE LLM, GIVE it")
                self.model = llm
                self.task_type = "CasualLM"
                self.tokenizer = tokenizer
            
        
    def load_debater(self, debate_path, name):
        print(f"Loading {name}")
        model_debate = ""
        if name == "judge":
            self.task_type = "SEQClassification"
            model_debate = AutoModelForSequenceClassification.from_pretrained(debate_path, num_labels=2)
        else:
            if "llama" in debate_path.lower():
                print("LOADING LLAMA")
                self.task_type = "CasualLM"
                model_debate = AutoModelForCausalLM.from_pretrained(debate_path)
                model_debate.config.use_cache = False
                model_debate.config.pretraining_tp = 1

                tokenizer = AutoTokenizer.from_pretrained(debate_path, padding=True)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"
            elif "qwen" in debate_path.lower():
                print("LOADING QWEN")
                self.task_type = "CasualLM"
                model_debate = AutoModelForCausalLM.from_pretrained(debate_path)

                tokenizer = AutoTokenizer.from_pretrained(debate_path, padding=True)

        return model_debate, tokenizer  
    def generate_response(self):
        with torch.no_grad():
            # 
            # inputs = self.tokenizer.apply_chat_template(
            #     self.dialog_template,
            #     tokenize=False,
            #     add_generation_prompt=True,
            #     return_tensors="pt"
            # )
            # inputs=self.tokenizer([inputs],return_tensors="pt").to(accelerator.device)
            if self.task_type == "CasualLM":
                print("A CASUAL MODEL")
                start_time = time.time()
                # output = self.model.generate(
                #     **inputs,
                #     max_new_tokens=self.max_new_tokens,
                #     temperature=self.temperature,
                #     do_sample=True,
                # )
                output = self.model.chat(self.dialog_template, sampling_params=self.sampling_params)
                end_time = time.time()
                # print(output)
                print(f"Reasoning Time:{end_time-start_time}s")
                # result = self.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                result = output[0].outputs[0].text
            elif self.task_type == "SEQClassification":
                print("A SEQC MODEL")
                input_text = "[SEP]".join([f"{msg['role']}: {msg['content']}" for msg in self.dialog_template])
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
                output = self.model(
                    **inputs,
                    return_dict=True
                )
                logit_score = output.logits
                result = torch.softmax(logit_score, dim=-1)
                result = result.tolist()[0]
            else:
                print("NOT IN!")
                exit()
        torch.cuda.empty_cache()
        return result
    
    def GPT_response(self, model_call):
        start_time = time.time()
        model_outout, total_tokens = process_question(
            client=self.client,
            prompt=self.dialog_template,
            model=model_call,
            temperature=self.temperature
        )
        end_time = time.time()
        print(f"Cost {end_time-start_time}s, {total_tokens} tokens")
        return model_outout, total_tokens
    
    
    def init_dialog(self, system):
        self.dialog_template = []
        self.dialog_template.append({"role": "system", "content": f"{system}"})
    
    def add_system(self, system):
        self.dialog_template.append({"role": "system", "content": f"{system}"})
        
    def add_user(self, user):
        self.dialog_template.append({"role": "user", "content": f"{user}"})
        
    def add_query(self, assistant):
        self.dialog_template.append({"role": "assistant", "content": f"{assistant}"})
        
    def query(self):
        if "gpt" in self.model_name.lower():
            model_output, _ = self.GPT_response()
        elif "qianwen" in self.model_name.lower():
            model_output, _ = self.GPT_response(model_call="qwen2.5-14b-instruct")
        else:
            model_output = self.generate_response()
        
        return model_output
        
        