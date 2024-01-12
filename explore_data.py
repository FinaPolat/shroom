import json
import random
import re

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data


data = read_json("prod/reference/test.model-agnostic.json")

print(len(data))

ref_labels = set()  
task_labels = set()
model_labels = set()

for i in data:
    ref_labels.add(i['ref'])
    task_labels.add(i['task'])
    model_labels.add(i['model'])

print(len(ref_labels))
print(ref_labels)

print(task_labels)
print(model_labels)



for i in random.sample(data, 5):
    #print(i.keys()) 
    #print(i.values())
    #if i['ref'] == 'src':
        print("Task: " + i['task'])
        print("Reference: " + i['ref'])
        print("Model: " + i['model'])
        print('Source sentence: ' + i['src'])
        print("Generation: " + i['hyp'])
        print("Gold: " + i['tgt'])
        print("------------------")
    


definition_modelling = []
machine_translation = []
paraphrasing = []

for i in data:
    if i['task'] == 'DM':
        label = i['label'].lower().strip()
        pattern = re.compile(r'<define>(.*?)<\/define>')
        match = pattern.search(i['src'])
        word_to_define = match.group(1).strip()
        definition = i['tgt'].strip()
        if label == "hallucination":
            rationale = f"It is {label} because the definition of '{word_to_define}' is '{definition}'. However, the generation: '{i['hyp']}' is not semantically compatible with the definition."
        if label == "not hallucination":
            rationale = f"It is {label} because the generated definition of '{word_to_define}' is semantically compatible with the definition: '{definition}'"
        i['rationale'] = rationale
        definition_modelling.append(i)
    if i['task'] == 'MT':
        if i['label'] == "Not Hallucination":
            i['rationale'] = f"{i['label']} because the generation: '{i['hyp']}' is semantically compatible with the reference: '{i['tgt']}"
        if i['label'] == "Hallucination":
            i['rationale'] = f"{i['label']} because the generation: '{i['hyp']}' is not semantically compatible with the reference: '{i['tgt']}"
        machine_translation.append(i)
    if i['task'] == 'PG':
        if i['label'] == "Not Hallucination":
            i['rationale'] = f"{i['label']} because the generation: '{i['hyp']}' is semantically compatible with the reference: '{i['tgt']}"
        if i['label'] == "Hallucination":
            i['rationale'] = f"{i['label']} because the generation: '{i['hyp']}' is not semantically compatible with the reference: '{i['tgt']}"
        paraphrasing.append(i)
    if i['task'] == 'TS':
        print(i)
    
        
print(len(definition_modelling))
print(len(machine_translation))
print(len(paraphrasing))
        
DM_random_10 = random.sample(definition_modelling, 10)

MT_random_10 = random.sample(machine_translation, 10)

PG_random_10 = random.sample(paraphrasing, 10)

with open('fina/annotated_examples/DM_random_10.json', 'w', encoding='utf-8') as f:
    json.dump(DM_random_10, f, indent=4, ensure_ascii=False)
    
with open('fina/annotated_examples/MT_random_10.json', 'w', encoding='utf-8') as f:
    json.dump(MT_random_10, f, indent=4, ensure_ascii=False)

with open('fina/annotated_examples/PG_random_10.json', 'w', encoding='utf-8') as f:
    json.dump(PG_random_10, f, indent=4, ensure_ascii=False)




