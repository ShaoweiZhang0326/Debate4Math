from debate.agent import Agent
import json
from vllm import LLM, SamplingParams
class Debater(Agent):
    
    def __init__(self, name, model_name, max_new_tokens, temperature, llm, tokenizer) -> None:
        super(Debater, self).__init__(name, model_name, max_new_tokens, temperature, llm, tokenizer)
        
        
class Debate:
    def __init__(
        self,
        debater_A_model_path,
        debater_B_model_path, 
        rules,
        temperature: float=1.2,
        round: int=4,
        Judge: str="gpt-3.5-turbo",
        ) -> None:
    
        self.temperature: float = temperature
        self.round: int = round
        self.debate_data: list = []
        self.player = [
            ""
        ]
        self.debater_A_model_path: str = debater_A_model_path
        self.debater_B_model_path: str = debater_B_model_path
        self.Judge: str = Judge
        self.rules = rules
        
        
    def init_prompt(self, data):
        self.debate_data = data

    def reasoning(self, input_path):
        self.model = LLM(model=self.debater_A_model_path)
        sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048)


    def init_agents(
        self,
        ):
        
        self.debater_A = Debater(
            name="Debater_A",
            model_name=self.debater_A_model_path,
            max_new_tokens=256,
            temperature=1.2,
            llm = "",
            tokenizer=""
        )
        if self.debater_A.llm != "":
            self.debater_B = Debater(
                name="Debater_A",
                model_name=self.debater_B_model_path,
                max_new_tokens=256,
                temperature=1.2,
                llm = self.debater_A.model,
                tokenizer=self.debater_A.tokenizer
            )
            
            self.Judge = Debater(
                name="Judge",
                model_name=self.Judge,
                max_new_tokens=256,
                temperature=0.1,
                # llm="",
                # tokenizer=""
                llm = self.debater_A.model,
                tokenizer=self.debater_A.tokenizer
            )
        else:
            self.debater_B = Debater(
                name="Debater_A",
                model_name=self.debater_B_model_path,
                max_new_tokens=256,
                temperature=1.2,
                llm = "",
                tokenizer=""
            )
            
            self.Judge = Debater(
                name="Judge",
                model_name=self.Judge,
                max_new_tokens=256,
                temperature=0.1,
                llm="",
                tokenizer=""
            )
        
        self.debater_A.add_system(self.rules)
        self.debater_B.add_system(self.rules)
        self.Judge.add_system(self.rules)
    

    def generate_prompt(self, round_number, opponent_speech, previous_speech, is_debater_A):
        self.role, self.oppo_role = ("Debater_A", "Debater_B") if is_debater_A else ("Debater_B", "Debater_A")

        start_prompt = f"""
        Now it is your turn as {self.role} in ROUND {round_number}.
        Remember to both criticize the arguments made by {self.oppo_role} and build up your own arguments
        Remember that it is your job to defend the position that {self.debater_step} is correct answer and that {self.oppo_role}'s position is wrong.
        You should assemble the math problem as well as the context of the reasoning steps for your response, and if there are references to a particular step, include them with <quote> </quote>
        And your response is limited to 200 tokens.
    
        {opponent_speech}
        
        {previous_speech}
        Write out your debate:
        """
        return start_prompt
    
    def generate_opponent_prompt(self, prompts):
        if not prompts:
            return "This is what your opponent said during their speech.\nNo opponent speech available."
        prompt = "This is what your opponent said during their speech.\n" + "\n".join([
            f"ROUND {idx + 1}: {content}" for idx, content in enumerate(prompts)
        ])
        return prompt
    
    def generate_previous_prompt(self, prompts):
        if not prompts:
            return "This is what you said during your previous speech.\nNo previous speech available."
        prompt = "This is what you said during your previous speech.\n" + "\n".join([
            f"ROUND {idx + 1}: {content}" for idx, content in enumerate(prompts)
        ])
        return prompt
        
    def debate_round(
        self, 
        debater_A_user,
        debater_B_user,
        debate_step,
        rounds: int,
    ):
        self.debater_step = debate_step
        A_query_list = []
        B_query_list = []
        for i in range(rounds):
            print(f"--------------debate round {i + 1}--------------")
            A_opponent_prompt_formatted = self.generate_opponent_prompt(B_query_list)
            A_previous_prompt_formatted = self.generate_previous_prompt(A_query_list)
            start_A = self.generate_prompt(
                round_number=i + 1,
                opponent_speech=A_opponent_prompt_formatted,
                previous_speech=A_previous_prompt_formatted,
                is_debater_A=True
            )
            self.debater_A.dialog_template = [self.debater_A.dialog_template[0]]
            self.debater_A.add_user(debater_A_user + start_A)
            
            A_query = self.debater_A.query()
            A_query_list.append(A_query)
            
            
            B_opponent_prompt_formatted = self.generate_opponent_prompt(A_query_list)
            B_previous_prompt_formatted = self.generate_previous_prompt(B_query_list)
            start_B = self.generate_prompt(
                round_number=i + 1,
                opponent_speech=B_opponent_prompt_formatted,
                previous_speech=B_previous_prompt_formatted,
                is_debater_A=False
            )
            self.debater_B.dialog_template = [self.debater_B.dialog_template[0]]  # 保留系统规则
            self.debater_B.add_user(debater_B_user + start_B)
            B_query = self.debater_B.query()
            B_query_list.append(B_query)

        final_round_data = {
            "speech_history": {
                "Debater_A": [f"ROUND {idx + 1}: {content}" for idx, content in enumerate(A_query_list)],
                "Debater_B": [f"ROUND {idx + 1}: {content}" for idx, content in enumerate(B_query_list)]
            }
        }
        
        # self.save_to_jsonl(final_round_data, "output.jsonl")
        return final_round_data
    
    def replace_judge_template(self, judge_template, speech_history):
        debater_a_rounds = "[DEBATER_A]\n".join(speech_history["Debater_A"])
        debater_b_rounds = "[DEBATER_B]\n".join(speech_history["Debater_B"])

        updated_template = judge_template.replace("<Debater_A_round>", debater_a_rounds)
        updated_template = updated_template.replace("<Debater_B_round>", debater_b_rounds)

        return updated_template

    def judge_decision(self, judge_template, speech_history):
        self.Judge.dialog_template = [self.Judge.dialog_template[0]]
        updated_template = self.replace_judge_template(judge_template, speech_history)
        # print(updated_template)
        self.Judge.add_user(updated_template)
        decision = self.Judge.query()
        return decision

    def save_to_jsonl(self, data, file_path):
        with open(file_path, "a", encoding="utf-8") as file:
            json.dump(data, file)
            file.write("\n")
        
        