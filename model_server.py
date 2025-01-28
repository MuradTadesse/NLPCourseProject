# src/deployment/model_server.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
from pathlib import Path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class ModelServer:
    def __init__(self, model_path, tokenizer_name):
        # Set device with memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear CUDA cache
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        # Load FAQ data first
        self.faqs = self.load_faq_data()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Initialize model
        try:
            # Load the model with the correct number of labels
            self.model = AutoModelForSequenceClassification.from_pretrained(
                tokenizer_name,
                num_labels=len(self.faqs)
            ).to(self.device)
            
            # Load saved weights if they exist
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print("Loaded pre-trained model successfully")
                else:
                    print("Invalid model checkpoint format")
            else:
                print("No pre-trained weights found. Using base model.")
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
    def load_faq_data(self):
        try:
            project_root = Path(__file__).parent.parent.parent
            faq_path = project_root / 'data' / 'raw_faqs.json'
            
            with open(faq_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('faqs', [])
        except Exception as e:
            print(f"Error loading FAQ data: {str(e)}")
            return []
            
    def predict(self, text):
        if not self.faqs:
            return "FAQ data not available."
            
        try:
            self.model.eval()
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)
                
                print(f"Query: {text}")
                print(f"Confidence: {confidence.item():.4f}")
                print(f"Predicted Index: {predicted_idx.item()}")
                
                if confidence.item() > 0.01:  # Lowered threshold
                    return self.faqs[predicted_idx.item()]["answer"]
                return "I'm not confident about the answer to that question. Could you please rephrase it?"
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return "I apologize, but I'm having trouble processing your request."