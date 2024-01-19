import json
import argparse


def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data


def post_process():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', type=str, default='experiment3/answers/gpt-3.5-turbo-1106/round2_Editor_val.model-agnostic.txt', help='Input file')

    args = parser.parse_args()

    with open(args.input_file, 'r', encoding= "utf-8") as f:
        data = f.readlines()

    GPT_answers = [] 

    for line in data:
        if not line.startswith('"index"'):
            print(line)
            line = line.replace("'", "")
            print(line)
            line = json.loads(line)
            GPT_answers.append(line)

    input_dir = args.input_file.split('/')[0]
    input_file = args.input_file.split('/')[-1]
    
    with open('experiment3/answers/gpt-3.5-turbo-1106/post_processed_round2_Editor_val.model-agnostic.txt', 'w', encoding= "utf-8") as f: 
        json.dump(GPT_answers, f, indent=4, ensure_ascii=False)

    
if __name__ == '__main__':
    post_process()