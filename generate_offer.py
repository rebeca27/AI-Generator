# generate_offer.py

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "openai-community/gpt2"
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def generate_offers(prompts, model_path, max_length=1024):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    attention_mask = (inputs['input_ids'] != tokenizer.pad_token_id).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.5,
        early_stopping=True,
        do_sample=True,  
        num_beams=5 
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def generate_definitions_and_technologies(cerinte):
    prompts = [f"Generează definițiile și tehnologiile folosite pentru următoarea cerință:\n{cerinta}\n\nDefiniții și tehnologii folosite:" for cerinta in cerinte]
    return generate_offers(prompts, './results/definitions', max_length=1024)

def generate_suggestions(cerinte):
    prompts = [f"Generează sugestii suplimentare pentru următoarea cerință:\n{cerinta}\n\nSugestii suplimentare:" for cerinta in cerinte]
    return generate_offers(prompts, './results/suggestions', max_length=1024)

def generate_propunere_structura(cerinte):
    prompts = [f"Cerinta: {cerinta}\n\nII. Propunere structură:" for cerinta in cerinte]
    return generate_offers(prompts, './results/propunere_structura')

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

model_zile = RandomForestRegressor(n_estimators=100, random_state=42)
model_programatori = RandomForestRegressor(n_estimators=100, random_state=42)

X_train, X_test, y_zile_train, y_zile_test, y_programatori_train, y_programatori_test = train_test_split(
    X_tfidf, y_zile, y_programatori, test_size=0.2, random_state=42
)

model_zile.fit(X_train, y_zile_train)
model_programatori.fit(X_train, y_programatori_train)

def estimate_days_and_programmers(cerinta, propunere_structura):
    X_new = vectorizer.transform([cerinta + ' ' + propunere_structura])
    zile = model_zile.predict(X_new)[0]
    programatori = model_programatori.predict(X_new)[0]
    return zile, programatori

cerinte = [item['cerinta'] for item in client_data]
definitions = generate_definitions_and_technologies(cerinte)
suggestions = generate_suggestions(cerinte)
structura = generate_propunere_structura(cerinte)

for i, item in enumerate(client_data):
    cerinta = item['cerinta']
    propunere_structura = structura[i]
    zile_dezvoltare, numar_programatori = estimate_days_and_programmers(cerinta, propunere_structura)
    total_ore = zile_dezvoltare * numar_programatori * 8 

    oferta_completa = template.replace('[cerinta]', cerinta) \
                             .replace('[Definitii]', definitions[i]) \
                             .replace('[sugestii_suplimentare]', suggestions[i]) \
                             .replace('[structura_propusa]', propunere_structura) \
                             .replace('[zile_dezvoltare]', f"{int(zile_dezvoltare)}") \
                             .replace('[numar_programatori]', f"{int(numar_programatori)}") \
                             .replace('[Total Ore lucratoare]', f"{int(total_ore)}") 

    with open(f'oferta_{item["cerinta"][:20]}.txt', 'w') as f:
        f.write(oferta_completa)

    print(f"Oferta pentru cerinta '{item['cerinta'][:20]}...' a fost salvată.")
