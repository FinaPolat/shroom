
import json

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data


personas = {"Lexicographer": "You are a lexicographer who writes definitions for a dictionary.",
            "Translator": "You are a translator who translates texts from Russion to English.",
            "Editor": "You are an editor who edits texts for a publishing company.",  
    }

system_template1 = """{persona} Your task at hand is {task_name}. You will be given three inputs: input text, target text, and generated text. Evaluate the generated text looking at the input text and the target text. Then, you need to decide whether the generated text is a hallucination or not.
There are two criteria for hallucination:
    If the generated text contatins any nonsensical or factually incorrect information, it is a hallucination.
    If the generated text contains additional information that cannot be supported by the input text or the target text, it is a hallucination.
Otherwise, the generated text is not a hallucination.
"""


query1_template ="""
Now, it is time to look at the inputs.      
    Input text: {input_text}
    Target text: {target_text}
    Generated text: {generation}    
Is the generated text a hallucination?
"""


system_template2 = """ {persona} Your task at hand is {task_name}. You were given three inputs: 
    input text: {input_text} 
    target text: {target_text}
    generated text: {generation}
There are two criteria for hallucination:
    If the generated text contatins any nonsensical or factually incorrect information, it is a hallucination.
    If the generated text contains additional information that cannot be supported by the input text or the target text, it is a hallucination.
Otherwise, the generated text is not a hallucination.
You evaluated the generated text looking at the input text and the target text. You decided that the generated text is "{llm_answer}".
"""

query_2_template = """"Why do you think that it is {llm_answer}? Please explain your reasoning. 
Also, assign an estimate of the probability that the generated text is a hallucination compared to the given inputs and targets. 
Please be aware that I am not asking how confident you are about your decision. 
I would like to know how likely the generated text is a hallucination compared to the given inputs and targets.
"""


data = read_json("data/val.model-agnostic.json")

print(len(data))

llm_answers = read_json("round1_answers_from_GPT3-5-turbo/post_processed_test_answers_gpt-3.5-turbo-1106.json")

print(len(llm_answers))

prompts = []

for i, llm_ans in zip(data, llm_answers):
    
    if i['task'] == "MT":
        task_name = "checking the quality and correctness of the automatically translated text by a machine translation (MT) system"
        persona = "You are a translator who translates texts from Russion to English."
        
    if i['task'] == "DM":
        task_name = "checking the quality and correctness of the automatically generated definitions by a definition modeling (DM) system for the words in <>"
        persona = "You are a lexicographer who writes definitions for a dictionary."
    
    if i['task'] == "PG":
        task_name = "checking the quality and correctness of the automatically generated texts by a paraphrase generation (PG) system"
        persona = "You are an editor who edits texts for a publishing company."

    #print(task_name)
    #print(persona)
        
    input_text = i['src']
    target_text = i['tgt']
    generated_text = i['hyp']
        
    system = system_template2.format(persona=persona, 
                                     task_name=task_name, 
                                     input_text=input_text, 
                                     target_text=target_text, 
                                     generation=generated_text, 
                                     llm_answer=llm_ans["Answer"])
    
    query = query_2_template.format(llm_answer=llm_ans["Answer"].lower())
    query = query + 'For your answer, use the following json format: {"Reasoning": "your_reasoning", "Probability": "0.0-1.0"}'

    label = i['label']
    
    prompt = {"system": system, 
              "query": query, 
              "label": label }
    
    print(json.dumps(prompt, indent=4))
    
    prompts.append(prompt)

with open('prompts/round2_prompts_validation_model_agnostic.json', 'w', encoding='utf-8') as f:
    json.dump(prompts, f, ensure_ascii=False, indent=4)
            