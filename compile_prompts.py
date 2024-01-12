
import json

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data

data = read_json("prod/reference/val.model-agnostic.json")

print(len(data))


prompt_template ="""The task at hand is {task_name}. You will be given three inputs: input text, target text, and generated text.  
You are asked to evaluate the generated text looking at the input text and the target text. Then, you need to decide whether the generated text is a hallucination or not.
There are two criteria for hallucination:
If the generated text contatins any nonsensical or factually incorrect information, it is a hallucination.
If the generated text contains additional information that cannot be supported by the input text or the target text, it is a hallucination.
Otherwise, the generated text is not a hallucination.
Now, it is time to look at the inputs.      
    Input text: {input_text}
    Target text: {target_text}
    Generated text: {generation}    
Is the generated text a hallucination? Answer "Hallucination" or "Not Hallucination", provide a short explanation for your answer, assign a probability score to your answer and justify the score.
"""

prompts = []

for i in data:
    #print(i)
    #print(type(i))
    #print(i.keys())
    #id = i['id']
    if i['task'] == "MT":
        task_name = "Machine Translation (MT)"
    elif i['task'] == "DM":
        task_name = "Definition Modelling (DM)"
    elif i['task'] == "PG":
        task_name = "Paraphrase Generation (PG)"
    input_text = i['src']
    target_text = i['tgt']
    generated_text = i['hyp']
    prompt = prompt_template.format(task_name=task_name, input_text=input_text, target_text=target_text, generation=generated_text)
    prompt = prompt + 'Use the following json format: {"Answer": "Hallucination/Not Hallucination", "Explanation": "Your explanation here", "Probability score": "score", "Justification of the score": "Your rationale for the score"}.'
    #print(prompt)
    #print("------------------")
    label = i['label']
    prompts.append({"prompt": prompt, "label": label})

with open('fina/prompts/prompts_validation_model_agnostic.json', 'w', encoding='utf-8') as f:
    json.dump(prompts, f, ensure_ascii=False, indent=4)
            