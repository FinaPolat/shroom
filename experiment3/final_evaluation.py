import json
from sklearn.metrics import classification_report
from scipy.stats import spearmanr

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

data = read_json("experiment3/answers/gpt-3.5-turbo-1106/round1_combined_answers.json")

gold_labels = []
system_label = []

gold_probas = []
system_probas = []

for item in data:
    gold_labels.append(item["gold_label"])
    system_label.append(item["combined_answer"])
    
    gold_probas.append(item["gold_p(Hallucination)"])
    system_probas.append(item["p(Hallucination)"])

print(classification_report(gold_labels, system_label, digits=2))


rho = spearmanr(gold_probas, system_probas)

print(f"Spearmanr rho: {rho[0]:.2f}")

results = {
    "Classification Report": classification_report(gold_labels, system_label, digits=2, output_dict=True),
    "Spearmanr": float(f"{rho[0]:.2f}")
}

with open("experiment3/scores/round1_combined_scores.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)