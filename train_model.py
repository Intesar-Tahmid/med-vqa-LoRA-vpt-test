#!/usr/bin/env python3
"""
Medical VQA Training/Testing with LLaMA Factory for Full Precision LoRA
Supports both training and testing modes with structured dataset organization
"""

import os
import logging
import json
import argparse
import pandas as pd
from pathlib import Path
import subprocess
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedVQADataset:
    """Dataset class for Medical VQA with structured organization"""

    def __init__(self, config):
        self.config = config
        self.data = []
        self.dataset_name = config.get('dataset_name', 'chest_x_ray')
        self.dataset_split = config.get('dataset_split', 'train')
    
    def get_dataset_paths(self):
        """Get paths based on dataset structure"""
        base_path = f"data/{self.dataset_name}/{self.dataset_split}"
        csv_path = f"{base_path}/{self.dataset_name}.csv"
        images_path = f"{base_path}/images"
        
        return csv_path, images_path

    def load_datasets(self):
        """Load the dataset from CSV"""
        csv_path, images_path = self.get_dataset_paths()

        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        
        # Filter out rows with missing image paths or missing answers
        df = df.dropna(subset=['image_path', 'question_bn', 'llm_answer_bn'])
        
        logger.info(f"Loaded dataset from: {csv_path}")
        logger.info(f"Images directory: {images_path}")
        logger.info(f"Total samples after filtering: {len(df)}")
        
        # Convert to list of dictionaries
        self.data = df.to_dict('records')
        
        return self.data

class MedVQATrainer:
    """Main trainer class using LLaMA Factory for training and testing"""
    
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()
        self.training_data = None
        # Check if this is a test config by looking for test dataset or task
        is_test_config = (
            'test' in self.config.get('dataset', '') or 
            self.config.get('task') == 'chat' or
            not self.config.get('do_train', True)
        )
        self.mode = 'test' if is_test_config else 'train'
        self.dataset_name = self.config.get('dataset_name', 'chest_x_ray')
        
    def load_config(self):
        """Load configuration from YAML file"""
        import yaml
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def prepare_dataset_for_llamafactory(self):
        """Check if dataset JSON exists, prepare if needed"""
        logger.info("Checking dataset for LLaMA Factory...")
        
        # Check if JSON file already exists
        dataset_name = self.config.get('dataset', 'chest_x_ray_train')
        
        # Parse dataset name to get base name and split
        if '_train' in dataset_name:
            base_name = dataset_name.replace('_train', '')
            split = 'train'
        elif '_test' in dataset_name:
            base_name = dataset_name.replace('_test', '')
            split = 'test'
        else:
            base_name = dataset_name
            split = 'train'
        
        # Handle special case for chest_x-ray (already has hyphen)
        if base_name == 'chest_x-ray':
            dir_name = 'chest_x-ray'
        else:
            dir_name = base_name.replace('_', '-')
        json_file = f"data/{dir_name}/{split}/{dir_name}_dataset.json"
        
        if os.path.exists(json_file):
            logger.info(f"Dataset JSON already exists: {json_file}")
            # Count samples
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Found {len(data)} samples in existing JSON")
            return len(data)
        else:
            logger.warning(f"Dataset JSON not found: {json_file}")
            logger.info("Please run: python prepare_data.py --all")
            raise FileNotFoundError(f"Dataset JSON not found: {json_file}")

    def train(self):
        """Train the model using LLaMA Factory"""
        logger.info("Starting LLaMA Factory training...")
        
        # Prepare dataset
        num_samples = self.prepare_dataset_for_llamafactory()
        
        # Build LLaMA Factory command
        cmd = [
            "llamafactory-cli", "train", self.config_path
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info("=" * 60)
        
        try:
            # Run the training command
            result = subprocess.run(cmd, check=True, capture_output=False)
            
            if result.returncode == 0:
                logger.info("Training completed successfully!")
                return True
            else:
                logger.error(f"Training failed with return code: {result.returncode}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            return False
        except FileNotFoundError:
            logger.error("llamafactory-cli not found. Please install LLaMA Factory.")
            return False
    
    def test(self):
        """Test the model and generate REAL predictions CSV"""
        logger.info("Starting REAL inference testing...")
        
        # Prepare dataset
        num_samples = self.prepare_dataset_for_llamafactory()
        
        # Run real inference instead of LLaMA Factory eval
        return self.run_real_inference()
    
    def run_real_inference(self):
        """Run actual inference to generate real predictions"""
        logger.info("Running REAL inference on test dataset...")
        
        # Load test data JSON
        dataset_name = self.config.get('dataset', 'chest_x_ray_test')
        if '_test' in dataset_name:
            base_name = dataset_name.replace('_test', '')
            split = 'test'
        else:
            base_name = dataset_name
            split = 'test'
        
        # Handle special case for chest_x-ray
        if base_name == 'chest_x_ray':
            dir_name = 'chest_x-ray'
        else:
            dir_name = base_name
        
        json_file = f"data/{dir_name}/{split}/{dir_name}_dataset.json"
        
        if not os.path.exists(json_file):
            logger.error(f"Test dataset JSON not found: {json_file}")
            return False
        
        # Load test data
        with open(json_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        logger.info(f"Found {len(test_data)} test samples")
        
        # Load model for inference
        from transformers import AutoModelForCausalLM, AutoProcessor
        from peft import PeftModel
        import torch
        
        # Disable PyTorch compilation to avoid C compiler issues
        torch._dynamo.config.disable = True
        
        logger.info("Loading model for inference...")
        processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
        model = AutoModelForCausalLM.from_pretrained(
            "google/medgemma-4b-it",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(model, "./output")
        model.eval()
        
        # Create predictions CSV
        predictions_data = []
        
        for i, sample in enumerate(test_data):
            logger.info(f"Processing sample {i+1}/{len(test_data)}")
            
            # Extract data
            image_path = sample['images'][0]
            question = sample['conversations'][0]['value'].replace('<image>\n', '')
            ground_truth = sample['conversations'][1]['value']
            
            # Run inference
            try:
                from PIL import Image
                image = Image.open(image_path).convert('RGB')
                
                messages = [{
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }]
                
                text = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(image, text, return_tensors="pt", add_special_tokens=False).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                predicted_answer = processor.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                logger.info(f"Generated prediction: {predicted_answer[:50]}...")
                
            except Exception as e:
                logger.error(f"Error in inference: {e}")
                predicted_answer = f"Error: {str(e)}"
            
            predictions_data.append({
                'image_id': f"test_{i+1:03d}",
                'image_path': image_path,
                'category': '',
                'category_bn': '',
                'question': '',
                'question_bn': question,
                'llm_answer': '',
                'llm_answer_bn': ground_truth,
                'predicted_answer_bn': predicted_answer
            })
        
        # Save predictions CSV in output folder
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        predictions_path = f"{output_dir}/{dir_name}_{split}_predictions.csv"
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(predictions_path, index=False)
        
        logger.info(f"✅ REAL predictions CSV saved to: {predictions_path}")
        logger.info(f"📊 Generated {len(predictions_data)} REAL predictions")
        
        # Show sample results
        logger.info("🎯 Sample predictions:")
        for i, row in predictions_df.head(3).iterrows():
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Question: {row['question_bn'][:60]}...")
            logger.info(f"  Prediction: {row['predicted_answer_bn'][:60]}...")
        
        return True

    def run_full_pipeline(self):
        """Run the complete training or testing pipeline"""
        if self.mode == 'train':
            logger.info("Starting Medical VQA training with LLaMA Factory")
        elif self.mode == 'test':
            logger.info("Starting Medical VQA testing with LLaMA Factory")
        else:
            logger.error(f"Unknown mode: {self.mode}. Use 'train' or 'test'.")
            return False
            
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Dataset: {self.dataset_name}")
        
        # Check if config file exists
        if not os.path.exists(self.config_path):
            logger.error(f"Config file not found: {self.config_path}")
            return False
        
        # Run training or testing based on mode
        if self.mode == 'train':
            success = self.train()
        elif self.mode == 'test':
            success = self.test()
        
        if success:
            logger.info("Full pipeline completed successfully!")
        else:
            logger.error("Pipeline failed!")
        
        return success

def main():
    parser = argparse.ArgumentParser(description="Train/Test Medical VQA model with LLaMA Factory")
    parser.add_argument("--config", type=str, default="configs/med-gemma-4b.yaml",
                       help="Path to LLaMA Factory config file")
    parser.add_argument("--mode", type=str, choices=['train', 'test'], 
                       help="Override mode from config (train or test)")
    parser.add_argument("--dataset", type=str,
                       help="Override dataset name from config")

    args = parser.parse_args()

    trainer = MedVQATrainer(args.config)
    
    # Override config with command line arguments if provided
    if args.mode:
        trainer.config['mode'] = args.mode
        trainer.mode = args.mode
    if args.dataset:
        trainer.config['dataset_name'] = args.dataset
        trainer.dataset_name = args.dataset
    
    success = trainer.run_full_pipeline()
    
    if success:
        logger.info("Pipeline completed successfully!")
    else:
        logger.error("Pipeline failed!")
        exit(1)

if __name__ == "__main__":
    main()