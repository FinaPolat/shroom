
import json

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data

data = read_json("data/val.model-agnostic.json")

print(len(data))

personas = {"Lexicographer": "You are a lexicographer who writes definitions for a dictionary.",
            "Translator": "You are a translator who translates texts from Russion to English.",
            "Editor": "You are an editor who edits texts for a publishing company.",  
    }

system_template = """{persona} Your task at hand is {task_name}. You will be given three inputs: input text, target text, and generated text. Evaluate the generated text looking at the input text and the target text. Then, you need to decide whether the generated text is a hallucination or not.
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

prompts = []

for i in data:
    
    if i['task'] == "MT":
        task_name = "checking the quality and correctness of the automatically translated text by a machine translation (MT) system."
        persona = "You are a translator who translates texts from Russion to English."
        
    if i['task'] == "DM":
        task_name = "checking the quality and correctness of the automatically generated definitions by a definition modeling (DM) system for the words in <>."
        persona = "You are a lexicographer who writes definitions for a dictionary."
    
    if i['task'] == "PG":
        task_name = "checking the quality and correctness of the automatically generated texts by a paraphrase generation (PG) system."
        persona = "You are an editor who edits texts for a publishing company."

    print(task_name)
    print(persona)
    system = system_template.format(persona=persona, task_name=task_name)

    input_text = i['src']
    target_text = i['tgt']
    generated_text = i['hyp']
    
    query1 = query1_template.format(task_name=task_name, input_text=input_text, target_text=target_text, generation=generated_text)
    query1 = query1 + 'For your answer, use the following json format: {"Answer": "Hallucination/Not Hallucination"}'

    label = i['label']
    
    prompt = {"system": system, "query1": query1, "label": label}
    print(json.dumps(prompt, indent=4))
    
    prompts.append(prompt)

with open('prompts/round1_prompts_validation_model_agnostic.json', 'w', encoding='utf-8') as f:
    json.dump(prompts, f, ensure_ascii=False, indent=4)
            