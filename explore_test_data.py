import json
import random

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data

data = read_json("prod/reference/test.model-agnostic.json")

print(len(data))

#print(data[0].keys())
#print(data[0].values())


for i in random.sample(data, 15):
    id = i['id']
    print(f'ID: {id}')
    task = i['task']
    print(f'Task: {task}')
    input_text = i['src']
    print(f'Input text: {input_text}')
    target_text = i['tgt']
    print(f'Target text: {target_text}')
    generation = i['hyp']
    print(f'Generated text: {generation}')
    print("------------------")