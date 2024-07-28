import os
import json
from docx import Document

def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def create_training_data(directory):
    training_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".docx"):
            file_path = os.path.join(directory, filename)
            content = read_docx(file_path)
           
            parts = content.split("\n\n", 1) 
            if len(parts) == 2:
                cerinta = parts[0]
                oferta = parts[1]
                training_data.append({
                    "cerinta": cerinta.strip(),
                    "oferta": oferta.strip()
                })
    return training_data

directory = "../data"
training_data = create_training_data(directory)

with open('data_exemple.json', 'w') as json_file:
    json.dump(training_data, json_file, ensure_ascii=False, indent=4)

print("Training data has been saved to data_exemple.json")
