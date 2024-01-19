import json

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def majority_vote(lst):
    if not lst:
        return None  # Return None for an empty list
    
    counts = {"Hallucination": 0, "Not Hallucination": 0}

    for item in lst:
        counts[item] += 1

    if counts["Hallucination"] > counts["Not Hallucination"]:
        return "Hallucination"
    elif counts["Not Hallucination"] > counts["Hallucination"]:
        return "Not Hallucination"
    else:
        return "It is a tie"  # Return None if there's a tie

def calculate_percentage(lst):
    total_items = len(lst)
    
    if total_items == 0:
        return 0.0
    
    correct_count = lst.count("Hallucination")
    percentage = (correct_count / total_items)
    
    return percentage


reference_data = read_json("data/val.model-agnostic.json")
#print(reference_data[0])

editor_answers = read_json("experiment3/answers/gpt-3.5-turbo-1106/post_processed_round1_Editor_val.model-agnostic.json")
#print(editor_answers[0])

lexicographer_answers = read_json("experiment3/answers/gpt-3.5-turbo-1106/post_processed_round1_Lexicographer_val.model-agnostic.json")
#print(lexicographer_answers[0])

student_answers = read_json("experiment3/answers/gpt-3.5-turbo-1106/post_processed_round1_Student_val.model-agnostic.json")
#print(student_answers[0])

translator_answers = read_json("experiment3/answers/gpt-3.5-turbo-1106/post_processed_round1_Translator_val.model-agnostic.json")
#print(translator_answers[0])

parttimer_answers = read_json("experiment3/answers/gpt-3.5-turbo-1106/post_processed_round1_Part-time worker_val.model-agnostic.json")
#print(parttimer_answers[0])


combined_answers = []
detailed_answers = []

for label, ed, lex, stu, tra, par in zip(reference_data, editor_answers, lexicographer_answers, student_answers, translator_answers, parttimer_answers):
    answers = {"gold_label": label["label"],
               "combined_answer": f'{majority_vote([ed["Answer"], lex["Answer"], stu["Answer"], tra["Answer"], par["Answer"]])}',
               "gold_p(Hallucination)": label["p(Hallucination)"],
                "p(Hallucination)": calculate_percentage([ed['Answer'], lex['Answer'], stu['Answer'], tra['Answer'], par['Answer']]),
    }
    detailed_answer = {"input_text": label["src"],
                    "target_text": label["tgt"],
                    "generated_text": label["hyp"],
                    "gold_label": label["label"],
                    "gold_p(Hallucination)": label["p(Hallucination)"],
                    "editor_answer": ed["Answer"],	
                    "translator_answer": tra["Answer"],
                    "lexicographer_answer": lex["Answer"],	
                    "student_answer": stu["Answer"],	
                    "parttimer_answer": par["Answer"],	
                    "combined_answer": f'{majority_vote([ed["Answer"], lex["Answer"], stu["Answer"], tra["Answer"], par["Answer"]])}',
                    "p(Hallucination)": f"{calculate_percentage([ed['Answer'], lex['Answer'], stu['Answer'], tra['Answer'], par['Answer']]):.2f}",	
                    }
    #print(answers)
    #print(detailed_answer)
    combined_answers.append(answers)
    detailed_answers.append(detailed_answer)

with open("experiment3/answers/gpt-3.5-turbo-1106/round1_combined_answers.json", 'w', encoding='utf-8') as f:
    json.dump(combined_answers, f, indent=4)

with open("experiment3/answers/gpt-3.5-turbo-1106/round1_detailed_answers.json", 'w', encoding='utf-8') as f:
    json.dump(detailed_answers, f, indent=4)