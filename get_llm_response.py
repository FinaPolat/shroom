
import os
import json
from openai import OpenAI
from retry import retry
import argparse
import logging
import time
from tqdm import tqdm

# Set the OpenAI API key
client = OpenAI(api_key="YOUR_API_KEY")

def read_prompts(input_file): 
    with open(input_file, 'r', encoding='utf-8') as prompt_file:
        prompts = json.load(prompt_file)
    return prompts


# Get an answer from the OpenAI-API
@retry(tries=3, delay=2, max_delay=10)
def GPT_repsonse_round(prompt, model, temperature, max_tokens):
    messages=[{"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["query"]},
            ]
    response = client.chat.completions.create(
                                            model=model,
                                            response_format={ "type": "json_object" },
                                            messages=messages,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
    )
    response = response.choices[0].message.content
    
    return response



def respond():
    
    logging.basicConfig(level=logging.INFO)
    logging.info('Start Logging')
    # Record the start time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', type=str, default='experiment3/prompts/round3_Editor_val.model-agnostic.json', help='Input directory')
    parser.add_argument('--model_name', type=str, default="gpt-3.5-turbo-1106", help='OpenAI model name')
    parser.add_argument('--out_dir', type=str, default='experiment3/answers', help='Output directory')
    
    args = parser.parse_args()
        
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    answers_dir = args.out_dir + '/' + args.model_name
    if not os.path.exists(answers_dir):
        os.makedirs(answers_dir)

    prompts = read_prompts(args.input_file)
    #prompts = prompts[:2]
    total_instances = len(prompts)
    
    logging.info(f'Read the input file: {args.input_file}. Number of instances: {total_instances}. Started.')

    outfile_path = answers_dir + '/' + args.input_file.split('/')[-1].replace('.json', '.txt')
    
    for i, prompt in enumerate(tqdm(prompts, desc=f"Passing inputs through {args.model_name} for answers", total=total_instances)): 

        answer = GPT_repsonse_round(prompt, args.model_name, temperature=0.7,  max_tokens=200) 
        #answer = json.loads(answer)
        #print(answer)
        answer = answer.replace('\n', '')

        outfile = open(outfile_path, 'a+', encoding='utf-8')
        outfile.write(f'"index": {i}\n')
        outfile.write(answer + '\n')    
        outfile.close()

    logging.info(f'Extraction finished. Saved the results to {args.out_dir}/{outfile}')
        
    end_time = time.time()
    
    # Calculate the elapsed time
    elapsed_time = (end_time - start_time) / 60
    print(f"Finished in {elapsed_time/60:.2f} hours")
        

if __name__ == "__main__":
    respond()