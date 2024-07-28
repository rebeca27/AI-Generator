# train_model.py

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
import optuna
from trl import PPOConfig, PPOTrainer, set_seed
import evaluate
import torch
import numpy as np

# model_name = "openai-community/gpt2"
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

with open('data_exemple.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

if not isinstance(data, list):
    raise ValueError("Error: data_exemple.json should be a list of dictionaries")

def extract_and_save_section(data, section_name, start_marker, end_marker, output_file):
    training_data = []
    for item in data:
        cerinta = item['cerinta']
        section_text = extract_section(item['oferta'], start_marker, end_marker)
        if section_text:
            training_data.append({
                'cerinta': cerinta,
                section_name: section_text
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

def extract_section(text, start_marker, end_marker):
    start = text.find(start_marker) + len(start_marker)
    if end_marker:
        end = text.find(end_marker, start)
        return text[start:end].strip() if start != -1 and end != -1 else ""
    else:
        return text[start:].strip() if start != -1 else ""

extract_and_save_section(data, 'definitii_tehnologii', "Definiții și tehnologii folosite:", "II. Propunere structură:", 'training_data_definitii.json')
extract_and_save_section(data, 'sugestii_suplimentare', "III. Sugestii suplimentare:", "IV. Pret și timp de implementare:", 'training_data_sugestii.json')
extract_and_save_section(data, 'propunere_structura', "II. Propunere structură:", "III. Sugestii suplimentare:", 'training_data_structura.json')

def load_training_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

training_data_definitions = load_training_data('training_data_definitii.json')
training_data_suggestions = load_training_data('training_data_sugestii.json')
training_data_structura = load_training_data('training_data_structura.json')

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels["input_ids"][idx])
        return item

    def __len__(self):
        return len(self.labels["input_ids"])

def tokenize_and_encode(data, section_name, max_length=1024):
    if not data:
        return None
    encodings = tokenizer(
        [item['cerinta'] for item in data],
        truncation=True,
        padding=True,
        max_length=max_length
    )
    labels = tokenizer(
        [item[section_name] for item in data],
        truncation=True,
        padding=True,
        max_length=max_length
    )
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    return CustomDataset(encodings, labels)

train_pairs_definitions, eval_pairs_definitions = train_test_split(training_data_definitions, test_size=0.1)
train_pairs_suggestions, eval_pairs_suggestions = train_test_split(training_data_suggestions, test_size=0.1)
train_pairs_propunere_structura, eval_pairs_propunere_structura = train_test_split(training_data_structura, test_size=0.1)

train_dataset_definitions = tokenize_and_encode(train_pairs_definitions, 'definitii_tehnologii')
eval_dataset_definitions = tokenize_and_encode(eval_pairs_definitions, 'definitii_tehnologii')
train_dataset_suggestions = tokenize_and_encode(train_pairs_suggestions, 'sugestii_suplimentare')
eval_dataset_suggestions = tokenize_and_encode(eval_pairs_suggestions, 'sugestii_suplimentare')
train_dataset_propunere_structura = tokenize_and_encode(train_pairs_propunere_structura, 'propunere_structura')
eval_dataset_propunere_structura = tokenize_and_encode(eval_pairs_propunere_structura, 'propunere_structura')

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

bleu_metric = evaluate.load("bleu", module_type="metric")
bertscore_metric = evaluate.load("bertscore", module_type="metric")

def evaluate_model(trainer, eval_dataset, tokenizer, bleu_metric, bertscore_metric):
    model.eval()
    references = []
    predictions = []

    for batch in trainer.get_eval_dataloader(eval_dataset):
        inputs = batch['input_ids']
        attention_mask = batch['attention_mask']
        with torch.no_grad():
            outputs = model.generate(inputs, attention_mask=attention_mask, max_length=1024, num_beams=5)
        
        for i in range(len(inputs)):
            ref = tokenizer.decode(inputs[i], skip_special_tokens=True)
            pred = tokenizer.decode(outputs[i], skip_special_tokens=True)
            references.append(ref)
            predictions.append(pred)

    bleu_score = bleu_metric.compute(predictions=predictions, references=references)['bleu']
    bertscore = bertscore_metric.compute(predictions=predictions, references=references, lang="en")['f1'][0]

    return bleu_score, bertscore

def objective(trial, dataset, eval_dataset, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=trial.suggest_int("num_train_epochs", 1, 5),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        per_device_train_batch_size=trial.suggest_int("per_device_train_batch_size", 1, 4),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.3),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    bleu_score, bertscore = evaluate_model(trainer, eval_dataset, tokenizer, bleu_metric, bertscore_metric)
    combined_score = 0.5 * bleu_score + 0.5 * bertscore
    
    return combined_score

def train_model_with_optuna(dataset, eval_dataset, output_dir):
    if dataset is None:
        print(f"Skipping training for {output_dir} as there is no data.")
        return

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, dataset, eval_dataset, output_dir), n_trials=10)
    
    print(f"Best hyperparameters for {output_dir}: {study.best_params}")

    best_training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=study.best_params['num_train_epochs'],
        learning_rate=study.best_params['learning_rate'],
        per_device_train_batch_size=study.best_params['per_device_train_batch_size'],
        weight_decay=study.best_params['weight_decay'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=best_training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()
    trainer.save_model(output_dir)

train_model_with_optuna(train_dataset_definitions, eval_dataset_definitions, './results/definitions')
train_model_with_optuna(train_dataset_suggestions, eval_dataset_suggestions, './results/suggestions')
train_model_with_optuna(train_dataset_propunere_structura, eval_dataset_propunere_structura, './results/propunere_structura')
