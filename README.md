# UET Educational Fine-tuning Project

A project for fine-tuning and evaluating large language models on educational content using Chain-of-Thought (CoT) prompting. This project uses Hugging Face Transformers and Google's Gemini API.

## Features

- Fine-tuning large language models (Llama, Qwen) on educational content
- Chain-of-Thought (CoT) response generation using Gemini API
- Support for both full model fine-tuning and LoRA (Low-Rank Adaptation)
- Batch inference capabilities
- Configurable training and inference parameters

## Project Structure

```
.
├── CoT_Generate.ipynb        # Notebook for generating Chain-of-Thought responses using Gemini
├── inference/
│   ├── inference.py         # Main inference script
│   ├── inference_lora.py    # Inference script for LoRA-tuned models
│   └── infer_config.yml     # Inference configuration
├── sft/
│   ├── sft_train.py        # Full model fine-tuning script
│   ├── sft_train_lora.py   # LoRA fine-tuning script
│   └── train_config.yml     # Training configuration
```

## Setup

1. Install required packages:
```bash
pip install transformers torch datasets peft wandb yaml tqdm bitsandbytes pandas google-cloud-aiplatform
```

2. Configure API keys:
- Set up Hugging Face access token
- Configure Weights & Biases API key
- Set up Google Gemini API key (for CoT generation)

3. Update configuration files:
- `sft/train_config.yml` for training settings
- `inference/infer_config.yml` for inference settings

## Training

### Full Model Fine-tuning

```bash
python sft/sft_train.py
```

### LoRA Fine-tuning

```bash
python sft/sft_train_lora.py
```

Key training parameters in `train_config.yml`:
- `model_name`: Base model to fine-tune
- `data_path_train`: Training dataset path
- `num_train_epochs`: Number of training epochs
- `train_batch_size`: Training batch size
- `learning_rate`: Learning rate

## Chain-of-Thought Generation

Use `CoT_Generate.ipynb` to generate Chain-of-Thought responses:

1. Set up Gemini API credentials
2. Load your dataset
3. Run the notebook to generate CoT responses
4. Results will be saved in JSON format

## Inference

### Standard Inference
```bash
python inference/inference.py
```

### LoRA Model Inference
```bash
python inference/inference_lora.py
```

Configure inference parameters in `infer_config.yml`:
- `model_name`: Path to fine-tuned model
- `data_path`: Test dataset path
- `device`: Inference device (cuda/cpu)

## Configuration Files

### Training Configuration (`train_config.yml`)
```yaml
model_name: "model_path"
data_path_train: "training_data_path"
num_train_epochs: 10
train_batch_size: 8
learning_rate: 0.00002
```

### Inference Configuration (`infer_config.yml`)
```yaml
model_name: "model_path"
data_path: "test_data_path"
device: "cuda"
```

## Requirements

- Python 3.8+
- CUDA-capable GPU
- Dependencies listed in requirements.txt

## License

This project is licensed under the MIT License.

## Acknowledgments

- Hugging Face Transformers
- Google Gemini API
- Weights & Biases