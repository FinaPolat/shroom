
import os
import json
import argparse

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data


personas = {"Lexicographer": "You are a lexicographer who writes definitions for a dictionary.",
            "Translator": "You are a translator who translates texts from Russion to English.",
            "Editor": "You are an editor who edits texts for a publishing company.", 
            "Student": "You are a student who works as a croudworker to earn money for your tuition.",
            "Part-time worker": "You are a part-time worker who works as a croudworker to earn money for your living expenses.",
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

query_2_template = """"Why do you think that it is {llm_answer}? Please explain your reasoning and assign an estimate of probability. 
1.0 means that the generated text is hallucination for sure and 0.0 means it is not hallucination at all. 
Please be aware that I am not asking how confident you are about your decision. 
I would like to know how likely the generated text is a hallucination compared to the given inputs and targets.
"""

system_template3 = """You are a computational linguist. Your task at hand is {task_name}. You are given three inputs: 
    input text: {input_text} 
    target text: {target_text}
    generated text: {generation}
There are two criteria for hallucination:
    If the generated text contatins any nonsensical or factually incorrect information, it is a hallucination.
    If the generated text contains additional information that cannot be supported by the input text or the target text, it is a hallucination.
Otherwise, the generated text is not a hallucination.
A {annotator} annotated the generated text and decided that the generated text is "{llm_answer}".
Here is the {annotator}'s rationale about the decision: {reasoning}
This is the probability assigned by the {annotator}: {probability}
"""

query_3_template = """I would like you to control the annotation considering the rationale and probability score provided by the annotator. 
1.0 means that the generated text is hallucination for sure and 0.0 means it is not hallucination at all.
As you already know, different annotators are likely to judge the same input differently and tend to disagree on the same input.
Now it is time to make a final decision. Is the generated text a hallucination or not? 
For your answer, use the following json format: {"Answer": "Hallucination/Not Hallucination", "p(Hallucination)": "0.0-1.0"}'
"""

def compile_prompt():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', type=str, default='data/val.model-agnostic.json', help='Input directory')
    parser.add_argument('--persona', type=str, default="Editor", help='Persona')
    parser.add_argument('--round', type=int, default=3, help='Round')
    parser.add_argument('--out_dir', type=str, default='experiment3/prompts', help='Output directory')
    parser.add_argument('--llm_answers', type=str, default='experiment3/answers/gpt-3.5-turbo-1106/post_processed_round1_Editor_val.model-agnostic.json', help='file llm answers')
    parser.add_argument('--llm_answers2', type=str, default='experiment3/answers/gpt-3.5-turbo-1106/post_processed_round2_Editor_val.model-agnostic.txt', help='file llm answers')

    args = parser.parse_args()

    filename = args.input_file.split('/')[-1]
    persona = personas[args.persona]

    data = read_json(args.input_file)
    print(len(data))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    prompts = []

    if args.round == 1:
        for i in data:
            if i['task'] == "MT":
                task_name = "checking the quality and correctness of the automatically translated text by a machine translation (MT) system"
            if i['task'] == "DM":
                task_name = "checking the quality and correctness of the automatically generated definitions of the words in <> by a definition modeling (DM) system"
            if i['task'] == "PG":
                task_name = "checking the quality and correctness of the automatically generated texts by a paraphrase generation (PG) system"
            
            input_text = i['src']
            target_text = i['tgt']
            generated_text = i['hyp']
            label = i['label']

            system = system_template1.format(persona=persona, task_name=task_name)
            query = query1_template.format(input_text=input_text, target_text=target_text, generation=generated_text)
            query = query + 'For your answer, use the following json format: {"Answer": "Hallucination/Not Hallucination"}'

            prompt = {"system": system, 
                "query": query, 
                "label": label }
            
            prompts.append(prompt)

        outfile_path = f'{args.out_dir}/round1_{args.persona}_{filename}'
        with open(outfile_path, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=4)

    if args.round == 2:
        llm_answers = read_json(args.llm_answers)
        print(len(llm_answers))

        for i, llm_ans in zip(data, llm_answers):
            if i['task'] == "MT":
                task_name = "checking the quality and correctness of the automatically translated text by a machine translation (MT) system"
            if i['task'] == "DM":
                task_name = "checking the quality and correctness of the automatically generated definitions by a definition modeling (DM) system for the words in <>"
            if i['task'] == "PG":
                task_name = "checking the quality and correctness of the automatically generated texts by a paraphrase generation (PG) system"

            input_text = i['src']
            target_text = i['tgt']
            generated_text = i['hyp']
            label = i['label']

            system = system_template2.format(persona=persona, 
                                            task_name=task_name, 
                                            input_text=input_text, 
                                            target_text=target_text, 
                                            generation=generated_text, 
                                            llm_answer=llm_ans["Answer"])
            
            query = query_2_template.format(llm_answer=llm_ans["Answer"].lower())
            query = query + 'For your answer, use the following json format: {"Reasoning": "your_reasoning", "Probability": "0.0-1.0"}'

            prompt = {"system": system,
                    "query": query,
                    "label": label }
            prompts.append(prompt)
        
        with open(f'{args.out_dir}/round2_{args.persona}_{filename}', 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=4)


    if args.round == 3:

        llm_answers = read_json(args.llm_answers)
        print(len(llm_answers))
        llm_answers2 = read_json(args.llm_answers2)
        print(len(llm_answers2))

        for i, llm_ans, llm_ans2 in zip(data, llm_answers, llm_answers2):
            print(llm_ans2)
            
            if i['task'] == "MT":
                task_name = "checking the quality and correctness of the automatically translated text by a machine translation (MT) system"
                
            if i['task'] == "DM":
                task_name = "checking the quality and correctness of the automatically generated definitions by a definition modeling (DM) system for the words in <>"
            
            if i['task'] == "PG":
                task_name = "checking the quality and correctness of the automatically generated texts by a paraphrase generation (PG) system"
                
            input_text = i['src']
            target_text = i['tgt']
            generated_text = i['hyp']
            label = i['label']
                
            system = system_template3.format(annotator=args.persona.lower(), 
                                            task_name=task_name, 
                                            input_text=input_text, 
                                            target_text=target_text, 
                                            generation=generated_text, 
                                            llm_answer=llm_ans["Answer"],
                                            reasoning =  llm_ans2["Reasoning"],
                                            probability = llm_ans2["Probability"])
            
            query = query_3_template

            prompt = {"system": system, 
                    "query": query, 
                    "label": label }
            
            prompts.append(prompt)

        with open(f'{args.out_dir}/round3_{args.persona}_{filename}', 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=4)
                    

if __name__ == "__main__":
    compile_prompt()