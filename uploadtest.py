import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
import torch

model_name = "openai-community/gpt2" 
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

def tokenize_and_encode(data, section_name):
    if not data:
        return None
    encodings = tokenizer(
        [item['cerinta'] for item in data],
        truncation=True,
        padding=True,
        max_length=1024
    )
    labels = tokenizer(
        [item[section_name] for item in data],
        truncation=True,
        padding=True,
        max_length=1024
    )
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    return CustomDataset(encodings, labels)

train_dataset_definitions = tokenize_and_encode(training_data_definitions, 'definitii_tehnologii')
train_dataset_suggestions = tokenize_and_encode(training_data_suggestions, 'sugestii_suplimentare')
train_dataset_propunere_structura = tokenize_and_encode(training_data_structura, 'propunere_structura')

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

def train_model(dataset, output_dir):
    if dataset is None:
        print(f"Skipping training for {output_dir} as there is no data.")
        return

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)

train_model(train_dataset_definitions, './results/definitions')
train_model(train_dataset_suggestions, './results/suggestions')
train_model(train_dataset_propunere_structura, './results/propunere_structura')

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
        early_stopping=True
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = output_text.replace(prompt, '', 1)
    return output_text

def generate_definitions_and_technologies(cerinta):
    definitions_prompt = f"Generează definițiile și tehnologiile folosite pentru următoarea cerință:\n{cerinta}\n\nDefiniții și tehnologii folosite:"
    return generate_offer(definitions_prompt, './results/definitions', max_length=512)

def generate_suggestions(cerinta):
    suggestions_prompt = f"Generează sugestii suplimentare pentru următoarea cerință:\n{cerinta}\n\nSugestii suplimentare:"
    return generate_offer(suggestions_prompt, './results/suggestions', max_length=512)

def generate_propunere_structura(cerinta):
    propunere_structura_prompt = f"Cerinta: {cerinta}\n\nII. Propunere structură:"
    return generate_offer(propunere_structura_prompt, './results/propunere_structura')

with open('template_oferta.txt', 'r') as f:
    template = f.read()

with open('client_data.json', 'r') as f:
    client_data = json.load(f)

for item in client_data:
    cerinta = item['cerinta']
    definitions = generate_definitions_and_technologies(cerinta)
    suggestions = generate_suggestions(cerinta)
    propunere_structura = generate_propunere_structura(cerinta)

    oferta_completa = template.replace('[cerinta]', cerinta) \
                             .replace('[Definitii]', definitions) \
                             .replace('[sugestii_suplimentare]', suggestions) \
                             .replace('[structura_propusa]', propunere_structura)
    with open(f'oferta_{item["cerinta"][:20]}.txt', 'w') as f:
        f.write(oferta_completa)

    print(f"Oferta pentru cerinta '{item['cerinta'][:20]}...' a fost salvată.")
