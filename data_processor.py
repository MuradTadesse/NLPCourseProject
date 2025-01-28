# src/preprocessing/data_processor.py
import pandas as pd
import torch
from transformers import AutoTokenizer

class FAQProcessor:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def process_faqs(self, faqs_data):
        processed_data = []
        for qa in faqs_data:
            encoded = self.tokenizer(
                qa['question'],
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            processed_data.append({
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'question': qa['question'],
                'answer': qa['answer']
            })
        return processed_data