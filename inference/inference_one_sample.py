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

model_path = "/data2/cmdir/home/ioit107/thviet/SFT/save_model/Llama-3.1-8B-Instruct_save_model_pretrain_2025-04-16_02-24"
base_model = "meta-llama/Llama-3.1-8B-Instruct"

cache_dir =  "/data2/cmdir/home/ioit107/nmquy/hf_cache"
model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)

#system_prompt = 'You are an assistant specializing in providing information related to admissions and academic training. You must answer the QUESTION below based solely on the CONTEXT provided. Every response must be complete, accurate, and grounded in the given information. If the QUESTION is outside your knowledge or the CONTEXT does not contain enough information to answer reliably, reply with: "Tôi không biết."'
system_prompt = "You are a helpful assistant. Answer the question below"

#reference_text = "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."
#reference_text = item["Văn bản tham chiếu"]
user_content = "Thủ đô của nước Pháp tên là gì?"
user_content = (
#    f"CONTEXT: {reference_text}\n"
    f"QUESTION: {user_content}"
        )
messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
            "role": "user",
            "content": user_content
    }
]

input_ids = tokenizer.apply_chat_template(
    messages,
    truncation=True,
    add_generation_prompt=True,
    add_special_tokens=True,
    return_tensors="pt"
    ).to("cuda")
outputs = model.generate(input_ids=input_ids,
                            max_new_tokens=512,
                            eos_token_id= tokenizer.eos_token_id)
res = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(res)