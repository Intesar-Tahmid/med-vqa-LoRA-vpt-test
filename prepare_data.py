#!/usr/bin/env python3
"""
Dataset Preparation Script for Medical VQA
Converts CSV files to LLaMA Factory ShareGPT format JSON files
"""

import pandas as pd
import json
import os
import argparse
from pathlib import Path

def prepare_dataset_json(dataset_name, split, csv_path, images_path, output_path):
    """
    Prepare dataset JSON in LLaMA Factory ShareGPT format
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'chest_x-ray', 'medicat')
        split (str): Dataset split ('train' or 'test')
        csv_path (str): Path to CSV file
        images_path (str): Path to images directory
        output_path (str): Path to save JSON file
    """
    
    print(f"Preparing {dataset_name} {split} dataset...")
    print(f"CSV: {csv_path}")
    print(f"Images: {images_path}")
    print(f"Output: {output_path}")
    
    # System prompt in Bengali
    system_bn = "আপনি একজন চিকিৎসা বিশেষজ্ঞ AI সহায়ক। আপনার কাজ হল চিকিৎসা ইমেজ বিশ্লেষণ করে বাংলায় প্রশ্নের উত্তর দেওয়া।"
    
    # Load CSV data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV")
    
    # Filter out rows with missing data
    required_columns = ['image_path', 'question_bn', 'llm_answer_bn']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    df_clean = df.dropna(subset=required_columns)
    print(f"After filtering missing data: {len(df_clean)} rows")
    
    # Prepare conversations
    conversations = []
    missing_images = []
    
    for idx, row in df_clean.iterrows():
        # Create conversation format for LLaMA Factory
        conversation = [
            {
                "from": "human",
                "value": f"<image>\n{row['question_bn']}"
            },
            {
                "from": "gpt",
                "value": row['llm_answer_bn']
            }
        ]
        
        # Construct image path
        image_filename = row['image_path']
        full_image_path = os.path.join(images_path, image_filename)
        
        # Check if image exists
        if not os.path.exists(full_image_path):
            missing_images.append(image_filename)
            print(f"Warning: Image not found: {full_image_path}")
            continue
        
        # Create conversation entry
        conversations.append({
            "conversations": conversation,
            "images": [full_image_path]
        })
    
    # Save JSON file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Created {len(conversations)} conversations")
    print(f"✅ Saved to: {output_path}")
    
    if missing_images:
        print(f"⚠️  Warning: {len(missing_images)} images were missing:")
        for img in missing_images[:5]:  # Show first 5
            print(f"   - {img}")
        if len(missing_images) > 5:
            print(f"   ... and {len(missing_images) - 5} more")
    
    return len(conversations)

def update_dataset_info(dataset_name, split, json_path):
    """
    Update the master dataset_info.json file
    """
    dataset_info_path = "data/dataset_info.json"
    dataset_key = f"{dataset_name}_{split}"
    
    # Load existing dataset info
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}
    
    # Add/update dataset entry
    dataset_info[dataset_key] = {
        "file_name": json_path,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "images"
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt"
        }
    }
    
    # Save updated dataset info
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Updated dataset_info.json with key: {dataset_key}")

def prepare_all_datasets():
    """Prepare all available datasets"""
    
    datasets_to_prepare = [
        ("chest_x-ray", "train"),
        ("chest_x-ray", "test"),
        ("medicat", "train"),
    ]
    
    print("🚀 Starting dataset preparation for all datasets...")
    
    success_count = 0
    total_count = len(datasets_to_prepare)
    
    for dataset, split in datasets_to_prepare:
        print(f"\n{'='*60}")
        print(f"Preparing {dataset} {split} dataset...")
        print(f"{'='*60}")
        
        try:
            # Auto-detect paths
            csv_path = f"data/{dataset}/{split}/{dataset}.csv"
            images_path = f"data/{dataset}/{split}/images"
            output_path = f"data/{dataset}/{split}/{dataset}_dataset.json"
            
            # Prepare dataset JSON
            num_conversations = prepare_dataset_json(
                dataset_name=dataset,
                split=split,
                csv_path=csv_path,
                images_path=images_path,
                output_path=output_path
            )
            
            # Update dataset info
            relative_json_path = os.path.relpath(output_path, "data")
            update_dataset_info(dataset, split, relative_json_path)
            
            print(f"✅ Successfully prepared {dataset} {split} dataset!")
            print(f"   Conversations: {num_conversations}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error preparing {dataset} {split}: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 PREPARATION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {success_count}/{total_count}")
    print(f"❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎯 All datasets prepared successfully!")
        return 0
    else:
        print(f"\n⚠️  Some datasets failed to prepare.")
        return 1

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset JSON files for LLaMA Factory")
    parser.add_argument("--all", action="store_true",
                       help="Prepare all available datasets")
    parser.add_argument("--dataset", type=str, 
                       help="Dataset name (e.g., 'chest_x-ray', 'medicat')")
    parser.add_argument("--split", type=str, choices=['train', 'test'],
                       help="Dataset split ('train' or 'test')")
    parser.add_argument("--csv", type=str, 
                       help="Path to CSV file (auto-detected if not provided)")
    parser.add_argument("--images", type=str,
                       help="Path to images directory (auto-detected if not provided)")
    parser.add_argument("--output", type=str,
                       help="Path to output JSON file (auto-detected if not provided)")
    
    args = parser.parse_args()
    
    # Handle --all option
    if args.all:
        return prepare_all_datasets()
    
    # Validate required arguments for single dataset preparation
    if not args.dataset or not args.split:
        parser.error("--dataset and --split are required unless using --all")
    
    # Auto-detect paths if not provided
    if not args.csv:
        args.csv = f"data/{args.dataset}/{args.split}/{args.dataset}.csv"
    
    if not args.images:
        args.images = f"data/{args.dataset}/{args.split}/images"
    
    if not args.output:
        args.output = f"data/{args.dataset}/{args.split}/{args.dataset}_dataset.json"
    
    try:
        # Prepare dataset JSON
        num_conversations = prepare_dataset_json(
            dataset_name=args.dataset,
            split=args.split,
            csv_path=args.csv,
            images_path=args.images,
            output_path=args.output
        )
        
        # Update dataset info
        relative_json_path = os.path.relpath(args.output, "data")
        update_dataset_info(args.dataset, args.split, relative_json_path)
        
        print(f"\n🎯 Successfully prepared {args.dataset} {args.split} dataset!")
        print(f"   Conversations: {num_conversations}")
        print(f"   JSON file: {args.output}")
        print(f"   Dataset key: {args.dataset}_{args.split}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
