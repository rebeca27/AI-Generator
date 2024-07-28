import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
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

for item in data:
    cerinta = item['cerinta']
    training_data_definitions.append({'cerinta': cerinta, 'definitii_tehnologii': ""})
    training_data_suggestions.append({'cerinta': cerinta, 'sugestii_suplimentare': ""})
    training_data_structura.append({'cerinta': cerinta, 'propunere_structura': ""})

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

def generate_offer(prompt, model_path, max_length=1024):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = (inputs != tokenizer.pad_token_id).to(device)
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.5,
        early_stopping=True,
        do_sample=True,  
        num_beams=5 
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = output_text.replace(prompt, '', 1)
    return output_text

def generate_definitions_and_technologies(cerinta):
    definitions_prompt = f"Generează definițiile și tehnologiile folosite pentru următoarea cerință:\n{cerinta}\n\nDefiniții și tehnologii folosite:"
    return generate_offer(definitions_prompt, './results/definitions', max_length=1024)

def generate_suggestions(cerinta):
    suggestions_prompt = f"Generează sugestii suplimentare pentru următoarea cerință:\n{cerinta}\n\nSugestii suplimentare:"
    return generate_offer(suggestions_prompt, './results/suggestions', max_length=1024)

def generate_propunere_structura(cerinta):
    propunere_structura_prompt = f"Cerinta: {cerinta}\n\nII. Propunere structură:"
    return generate_offer(propunere_structura_prompt, './results/propunere_structura')

with open('template_oferta.txt', 'r') as f:
    template = f.read()

with open('client_data.json', 'r') as f:
    client_data = json.load(f)

with open('data_preturi.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

X = df['cerinta'] + ' ' + df['propunere_structura']
y_zile = df['zile_dezvoltare']
y_programatori = df['numar_programatori']

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_zile_train, y_zile_test, y_programatori_train, y_programatori_test = train_test_split(
    X_tfidf, y_zile, y_programatori, test_size=0.2, random_state=42
)

model_zile = RandomForestRegressor(n_estimators=100, random_state=42)
model_zile.fit(X_train, y_zile_train)

model_programatori = RandomForestRegressor(n_estimators=100, random_state=42)
model_programatori.fit(X_train, y_programatori_train)

y_zile_pred = model_zile.predict(X_test)
y_programatori_pred = model_programatori.predict(X_test)

def estimate_days_and_programmers(cerinta, propunere_structura):
    X_new = vectorizer.transform([cerinta + ' ' + propunere_structura])
    zile = model_zile.predict(X_new)[0]
    programatori = model_programatori.predict(X_new)[0]
    return zile, programatori

for item in client_data:
    cerinta = item['cerinta']
    definitions = generate_definitions_and_technologies(cerinta)
    suggestions = generate_suggestions(cerinta)
    propunere_structura = generate_propunere_structura(cerinta)
    zile_dezvoltare, numar_programatori = estimate_days_and_programmers(cerinta, propunere_structura)
    total_ore = zile_dezvoltare * numar_programatori * 8 

    oferta_completa = template.replace('[cerinta]', cerinta) \
                             .replace('[Definitii]', definitions) \
                             .replace('[sugestii_suplimentare]', suggestions) \
                             .replace('[structura_propusa]', propunere_structura) \
                             .replace('[zile_dezvoltare]', f"{int(zile_dezvoltare)}") \
                             .replace('[numar_programatori]', f"{int(numar_programatori)}") \
                             .replace('[Total Ore lucratoare]', f"{int(total_ore)}") 

    with open(f'oferta_{item["cerinta"][:20]}.txt', 'w') as f:
        f.write(oferta_completa)

    print(f"Oferta pentru cerinta '{item['cerinta'][:20]}...' a fost salvată.")

bleu_metric = evaluate.load("bleu", module_type="metric")
bertscore_metric = evaluate.load("bertscore", module_type="metric")

def reward_function(samples):
    rewards = []
    for sample in samples:
        cerinta = sample["input_ids"]
        oferta_generata = sample["generated_token_ids"]

        cerinta_text = tokenizer.decode(cerinta, skip_special_tokens=True)
        oferta_generata_text = tokenizer.decode(oferta_generata, skip_special_tokens=True)

        bleu_score = bleu_metric.compute(predictions=[oferta_generata_text], references=[cerinta_text])['bleu']

        bertscore = bertscore_metric.compute(predictions=[oferta_generata_text], references=[cerinta_text], lang="en")['f1'][0]

        lungime_oferta = len(oferta_generata)
        are_definitii = "Definiții și tehnologii folosite:" in oferta_generata_text
        are_sugestii = "III. Sugestii suplimentare:" in oferta_generata_text
        are_structura = "II. Propunere structură:" in oferta_generata_text

        reward = (
            0.5 * bleu_score +  
            0.5 * bertscore +  
            0.1 * lungime_oferta +  
            1.0 if are_definitii else 0 +  
            1.0 if are_sugestii else 0 +   
            1.0 if are_structura else 0    
        )

        rewards.append(reward)

    return rewards

set_seed(42)  

ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    batch_size=8,
    mini_batch_size=2,  
    gradient_accumulation_steps=4 
)

trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=ppo_config,
    dataset=train_dataset_definitions
)

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

for epoch in range(ppo_config.num_train_epochs):
    for batch in trainer.dataloader:
        trainer.step(batch)
    
    bleu_score, bertscore = evaluate_model(trainer, eval_dataset_definitions, tokenizer, bleu_metric, bertscore_metric)
    print(f"Epoch {epoch + 1} - BLEU Score: {bleu_score:.4f}, BERTScore: {bertscore:.4f}")

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

train_model_with_optuna(train_dataset_definitions, eval_dataset_definitions, './results/definitions')
train_model_with_optuna(train_dataset_suggestions, eval_dataset_suggestions, './results/suggestions')
train_model_with_optuna(train_dataset_propunere_structura, eval_dataset_propunere_structura, './results/propunere_structura')
