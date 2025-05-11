import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset, load_dataset
import csv
import yaml
from datetime import datetime
import time
import os
from tqdm import tqdm
from peft import PeftModel
def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

config = load_config("config.yml")

model_path = config["model_name"]
cache_dir=config['cache_dir']
device = config['device']
base_model = config['base_model']

#model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, cache_dir=cache_dir, device_map = "cuda")
model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir).to("cuda")
#model = AutoModelForCausalLM.from_pretrained(base_model, cache_dir=cache_dir).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)


data_path =  config['data_path']
#datasets = load_dataset('csv', data_files=data_path)
datasets = load_dataset(data_path)
print(datasets.keys())

questions = []
results = []


for i, item in tqdm(enumerate(datasets['train']),total=len(datasets['train']), desc="Processing Questions"):
    user_content = item["Câu hỏi"]
    reference_text = item["Văn bản tham chiếu"]
    user_content = (
        f"CONTEXT: {reference_text}\n"
        f"QUESTION: {user_content}"
         )
    messages = [
        {
            "role": "system",
            "content": config['system_prompt']
        },
        {
                "role": "user",
                "content": user_content
        }
    ]
#    if item["Văn bản tham chiếu"] is not None:
#        messages.append({
#            "role": "assistant",
#            "content": item["Văn bản tham chiếu"]
#        })
    input_ids = tokenizer.apply_chat_template(
        messages,
        truncation=True,
        add_generation_prompt=True,
        add_special_tokens=True,
        return_tensors="pt"
        ).to(device)
    start_time = time.time()
    outputs = model.generate(input_ids=input_ids,
                             max_new_tokens=config['max_new_token'],
                             #temperature=config["temperature"],
                             #top_p=config['top_p'],
                             #repetition_penalty=config['repetition_penalty'],
                             eos_token_id= tokenizer.eos_token_id)
    end_time = time.time()
    elapsed_time = end_time - start_time
    generate_token = outputs.size(1) - input_ids.size(1)
    token_speed = generate_token/elapsed_time

    res_assistant = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"item : {i+1}--------------------------------------")
    print("Question: ")
    print(item['Câu hỏi'])
    print("Assistant Answer: ")
    print(res_assistant.split("assistant")[-1])
    print("speed generate token = ", token_speed)
    questions.append(user_content)
    results.append(res_assistant.split("assistant")[-1])

filename = config['path_filename'] + datetime.now().strftime("%Y-%m-%d_%H-%M")+".csv"

with open(filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Question", "Result"])
    for ques, res in zip(questions, results):
        writer.writerow([ques, res])
print("Inference Successfull")