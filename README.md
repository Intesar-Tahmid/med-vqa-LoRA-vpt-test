# Medical VQA LoRA Training

Fine-tuning `google/medgemma-4b-it` vision-language model using FP32 LoRA with LLaMA Factory for Bengali medical question answering.

## 📁 Repository Structure

```
med-vqa-lora/
├── configs/
│   ├── med-gemma-4b.yaml          # Training configuration
│   └── med-gemma-4b-test.yaml     # Testing configuration
├── data/
│   ├── dataset_info.json          # Master dataset registry
│   ├── chest_x-ray/
│   │   ├── train/
│   │   │   ├── images/            # Training images
│   │   │   ├── chest_x-ray.csv    # Training data
│   │   │   └── chest_x-ray_dataset.json
│   │   └── test/
│   │       ├── images/            # Test images
│   │       ├── chest_x-ray.csv    # Test data
│   │       └── chest_x-ray_dataset.json
│   └── medicat/
│       └── train/
│           ├── images/            # Medicat training images
│           ├── medicat.csv        # Medicat training data
│           └── medicat_dataset.json
├── output/                        # Model outputs and predictions
├── prepare_data.py                # Data preparation script
├── train_model.py                 # Main training/testing script
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container setup
└── README.md                      # This file
```

## 🚀 Usage

### 1. Build Docker Container
```bash
docker build -t med-vqa:latest .
```

### 2. Prepare Data
```bash
# Prepare all datasets
docker run --rm -v $(pwd):/app med-vqa:latest python prepare_data.py --all

# Prepare specific dataset
docker run --rm -v $(pwd):/app med-vqa:latest python prepare_data.py --dataset chest_x-ray --split train
```

### 3. Run Training
```bash
docker run --gpus all --ipc=host --rm --env-file .env -v $(pwd):/app med-vqa:latest python train_model.py --config configs/med-gemma-4b.yaml
```

### 4. Run Testing
```bash
docker run --gpus all --ipc=host --rm --env-file .env -v $(pwd):/app med-vqa:latest python train_model.py --config configs/med-gemma-4b-test.yaml
```

### 5. Complete Pipeline
```bash
# Train and test in sequence
docker run --gpus all --ipc=host --rm --env-file .env -v $(pwd):/app med-vqa:latest python train_model.py --config configs/med-gemma-4b.yaml
docker run --gpus all --ipc=host --rm --env-file .env -v $(pwd):/app med-vqa:latest python train_model.py --config configs/med-gemma-4b-test.yaml
```

## ⚙️ Configuration

### Training (`configs/med-gemma-4b.yaml`)
- Model: `google/medgemma-4b-it`
- LoRA Rank: 16, Alpha: 32
- Batch Size: 2 per device, 2 gradient accumulation
- Template: `gemma3`

### Testing (`configs/med-gemma-4b-test.yaml`)
- Loads trained adapter from `./output`
- Generates predictions for test dataset

## 📝 Data Preparation

```bash
# Prepare all datasets
python prepare_data.py --all

# Prepare specific dataset
python prepare_data.py --dataset chest_x-ray --split train

# Custom paths
python prepare_data.py --dataset chest_x-ray --split test --csv data/chest_x-ray/test/chest_x-ray.csv --images data/chest_x-ray/test/images --output data/chest_x-ray/test/chest_x-ray_dataset.json
```

## 🎯 Output Files

### Training Outputs
- `output/adapter_model.safetensors` - LoRA weights (119MB)
- `output/adapter_config.json` - LoRA configuration
- `output/checkpoint-32/` - Training checkpoint
- `output/train_results.json` - Training metrics

### Test Predictions
- `output/chest_x-ray_test_predictions.csv` - Test predictions with columns:
  - `image_id`, `image_path`, `category`, `category_bn`
  - `question`, `question_bn`, `llm_answer`, `llm_answer_bn`
  - `predicted_answer_bn` - Model's predictions

## 🔍 Example Usage

### Sample Predictions
```csv
image_id,image_path,question_bn,llm_answer_bn,predicted_answer_bn
test_001,data/chest_x-ray/test/images/00015953_015.png,এখানে কোন নির্দিষ্ট অবস্থা চিহ্নিত করা হয়েছে?,কোন নির্দিষ্ট অবস্থা চিহ্নিত হয়নি।,কোন নির্দিষ্ট অবস্থা চিহ্নিত করা হয়নি।
test_002,data/chest_x-ray/test/images/00011237_094.png,অনুপ্রবেশটি কোথায় অবস্থিত?,মধ্য ডান,মধ্য বাম
```

### Model Loading
```python
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("google/medgemma-4b-it")
processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
model = PeftModel.from_pretrained(model, "./output")
```