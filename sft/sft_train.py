from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
import torch
import wandb
import huggingface_hub
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import yaml
from datetime import datetime
import re
import random
from multiprocessing import cpu_count
import ast

def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

config = load_config("config.yml")

huggingface_hub.login(config['huggingface_access_key'])


wandb.login(key=config['wandb_access_key'])
wandb.init(project=config['wandb_project'])


base_model = config['model_name']
#tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer_path = config['model_name']
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=config["cache_dir"])


model = AutoModelForCausalLM.from_pretrained(base_model, cache_dir = config["cache_dir"])
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
# change the padding tokenizer value
tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
model.config.pad_token_id = tokenizer.pad_token_id # updating model config
tokenizer.padding_side = 'right' # padding to right (otherwise SFTTrainer shows warning)
print(tokenizer.all_special_tokens)
def modify_content(example):
    example["message"][0]["content"] = "You are an expert in the field of education and must provide accurate, fact-based answers in Vietnamese. If a question is beyond your knowledge or you cannot provide a reliable answer, respond with 'Tôi không biết.' Every answer you provide must be based on the reference text provided."
    return example

def eval_message(example):
    return {"message": ast.literal_eval(example["message"])}

#dataset["train"] = dataset["train"].map(eval_message)
dataset_source = config['data_path_train']
dataset_eval_path = config["data_path_eval"]

dataset = load_dataset(dataset_source, cache_dir=config['cache_dir'])
#dataset["train"] = dataset["train"].map(eval_message)
dataset = dataset.filter(lambda ex: all(m.get("content") is not None for m in ex["message"]))

eval_dataset = load_dataset(dataset_eval_path, cache_dir=config['cache_dir'])
#eval_dataset["train"] = eval_dataset["train"].map(eval_message)
eval_dataset = eval_dataset.filter(lambda ex: all(m.get("content") is not None for m in ex["message"]))

dataset_eval = eval_dataset

system_prompt = config['system_prompt']


def apply_chat_template(example, tokenizer, system_prompt):
    messages = example["message"]
    #if messages[0]["role"] != "system":
    #    messages.insert(0, {"role": "system", "content": system_prompt})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example

column_names = list(dataset["train"].features)
dataset = dataset.map(apply_chat_template,
                                num_proc=cpu_count(),
                                fn_kwargs={"tokenizer": tokenizer, 'system_prompt': system_prompt},
                                remove_columns=column_names,
                                desc="Applying chat template",)

dataset_eval = dataset_eval.map(apply_chat_template,
                                  num_proc=cpu_count(),
                                  fn_kwargs={"tokenizer": tokenizer, 'system_prompt': system_prompt},
                                  remove_columns=column_names,
                                  desc="Applying chat template",)



train_dataset = dataset['train']
eval_dataset = dataset_eval


for index in random.sample(range(len(dataset["train"])), 3):
  print(f"Sample {index} of the processed training set:\n\n{dataset['train'][index]['text']}")


from datetime import datetime
from trl import SFTTrainer
from transformers import TrainingArguments
import trl

import transformers


name_model = base_model.split("/")[-1]

output_path = config['output_path'] + f'{name_model}_output_edu_' + datetime.now().strftime("%Y-%m-%d_%H-%M")
save_path = config['save_path'] + f"{name_model}_save_model_pretrain_" + datetime.now().strftime("%Y-%m-%d_%H-%M")
hub_model_id = "model_" + datetime.now().strftime("%Y-%m-%d_%H-%M")

from trl import SFTTrainer, SFTConfig

# Create the SFTConfig
sft_config = SFTConfig(
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    do_eval=True,
    num_train_epochs=config['num_train_epochs'],
   # evaluation_strategy=config['evaluation_strategy'],
    per_device_eval_batch_size=config['eval_batch_size'],
    per_device_train_batch_size=config['train_batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    gradient_checkpointing=config['gradient_checkpointing'],
    learning_rate=config['learning_rate'],
    warmup_steps=config['warmup_steps'],
    logging_strategy=config['logging_strategy'],
    logging_steps=config['logging_steps'],
    log_level=config['log_level'],
    lr_scheduler_type=config['lr_scheduler_type'],
    save_strategy=config['save_strategy'],
    eval_strategy=config['eval_strategy'],  # optional: often redundant with evaluation_strategy
    report_to='wandb',
    output_dir=output_path,
    overwrite_output_dir=True,
    weight_decay=config['weight_decay'],
    save_total_limit=None,
    seed=42,
    push_to_hub=False,
    dataset_text_field=config['dataset_text_field'],
    max_seq_length=config['max_seq_length'],
    deepspeed=config['deepspeed'],
)

# Instantiate the trainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer
)
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer.train()


trainer.save_model()
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)