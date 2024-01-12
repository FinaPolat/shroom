import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data


def evaluate():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', type=str, default='answers_from_GPT4/answers_gpt-4.txt', help='Input file')
    parser.add_argument('--ref_file', type=str, default='data/val.model-agnostic.json', help='reference file')
    parser.add_argument('--scores_dir', type=str, default='scores', help='Output directory to save the scores')

    args = parser.parse_args()

    with open(args.input_file, 'r', encoding= "utf-8") as f:
        data = f.readlines()

    GPT_answers = []
    hallucination_yes_no = [] 

    for line in data:
        if line.startswith('"llm_answer":'):
            print(line)
            line = "{" + line + "}"
            line = json.loads(line)
            line = line['llm_answer']
            #print(line.keys())
            #print(line['Answer'])
            #line["Probability score"] = str(line["Probability score"])
            GPT_answers.append(line)
            if line['Answer'] == 'Hallucination':
                hallucination_yes_no.append(1)
            else:
                hallucination_yes_no.append(0)


    input_dir = args.input_file.split('/')[0]
    input_file = args.input_file.split('/')[-1]
    
    with open(f'{input_dir}/post_processed_{input_file.replace(".txt", ".json")}', 'w', encoding= "utf-8") as f: 
        json.dump(GPT_answers, f, indent=4, ensure_ascii=False)

    ref_data = read_json(args.ref_file)

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


    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")


    cr = classification_report(y_true, y_pred, target_names=target_names)
    print(cr)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

    disp.plot()
    plt.show()

    with open(f'{args.scores_dir}/scores_{input_file}', 'w', encoding= "utf-8") as f: 
        f.write(f"True Negatives: {tn}\n")
        f.write(f"False Positives: {fp}\n") 
        f.write(f"False Negatives: {fn}\n")
        f.write(f"True Positives: {tp}\n")
        f.write(f'\n{cr}\n')

if __name__ == '__main__':
    evaluate()