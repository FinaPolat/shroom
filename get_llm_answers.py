import os
import json
import openai
from retry import retry
import argparse
import logging
import time
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = 'your_api_key'
openai.api_key = os.environ["OPENAI_API_KEY"]


def read_prompts(input_file): 
    with open(input_file, 'r', encoding='utf-8') as prompt_file:
        prompts = json.load(prompt_file)
    return prompts


# Get an answer from the OpenAI-API
@retry(tries=3, delay=2, max_delay=10)
def GPT_repsonse(prompts, model, temperature, max_tokens):
    messages=[{"role": "user", "content": prompts}]
    response = openai.ChatCompletion.create(
                                            model=model,
                                            messages=messages,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
    )

    response = response.choices[0]["message"]["content"]
    
    return response


def main(): 
    
    logging.basicConfig(level=logging.INFO)
    logging.info('Start Logging')
    # Record the start time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', type=str, default='fina/prompts/prompts_validation_model_agnostic.json', help='Input directory')
    parser.add_argument('--model_name', type=str, default='gpt-4', help='OpenAI model name')
    parser.add_argument('--out_dir', type=str, default='fina/answers_from_GPT4', help='Output directory')
    
    args = parser.parse_args()
        
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    prompts = read_prompts(args.input_file)
    #prompts = prompts[:1]
    total_instances = len(prompts)
    
    logging.info(f'Read the input file: {args.input_file}. Number of instances: {total_instances}. Started.')

    outfile_path = f'{args.out_dir}/answers_{args.model_name}.txt'
    
    for i, prompt in enumerate(tqdm(prompts, desc=f"Passing inputs through {args.model_name} for answers", total=total_instances)): 
        answer = GPT_repsonse(prompt['prompt'], args.model_name, temperature=0.7,  max_tokens=2000)
        print(answer)
        outfile = open(outfile_path, 'a+', encoding='utf-8')
        outfile.write(f'"index": {i}\n')
        outfile.write(f'"llm_answer": {answer}\n')    
        outfile.close()

    logging.info(f'Extraction finished. Saved the results to {args.out_dir}/{outfile}')
        
    end_time = time.time()
    
    # Calculate the elapsed time
    elapsed_time = (end_time - start_time) / 60
    print(f"Finished in {elapsed_time/60:.2f} hours")
        
if __name__ == '__main__':
    main()