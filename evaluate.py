import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data

#################

with open('fina/answers_from_GPT4/answers_gpt-4.txt', 'r', encoding= "utf-8") as f:
    data = f.readlines()


GPT4_answers = []
hallucination_yes_no = [] 
for line in data:
    if line.startswith('"llm_answer":'):
        line = "{" + line + "}"
        line = json.loads(line)
        line = line['llm_answer']
        #print(line.keys())
        #print(line['Answer'])
        line["Probability score"] = float(line["Probability score"])
        GPT4_answers.append(line)
        if line['Answer'] == 'Hallucination':
            hallucination_yes_no.append(1)
        else:
            hallucination_yes_no.append(0)

with open('fina/answers_from_GPT4/answers_gpt-4_postprocessed.json', 'w', encoding= "utf-8") as f: 
    json.dump(GPT4_answers, f, indent=4, ensure_ascii=False)

###############################
    
ref_data = read_json("prod/reference/val.model-agnostic.json")

references = []
for i in ref_data:
    if i['label'] == 'Hallucination':
        references.append(1)
    else:
        references.append(0)

#print(references)
#print(hallucination_yes_no)

y_true = np.array(references)
y_pred = np.array(hallucination_yes_no)
target_names=['Not Hallucination', 'Hallucination']


cr = classification_report(y_true, y_pred, target_names=target_names)
print(cr)

cm = confusion_matrix(y_true, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

disp.plot()
plt.show()