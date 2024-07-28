import json
import docx
import os

def read_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def extract_section(text, start_marker, end_marker=None):
    start = text.find(start_marker) + len(start_marker)
    if end_marker:
        end = text.find(end_marker, start)
        return text[start:end].strip() if start != -1 and end != -1 else ""
    else:
        return text[start:].strip() if start != -1 else ""

document_folder = "../data/"

document_names = [
    "Oferta - Test 1.docx", "Oferta - Test 2.docx", "Oferta - Test 3.docx",
    "Oferta - Test 4.docx", "Ofertă - Test 6.docx", "Oferta - Test 7_.docx",
    "Oferta - Test 10.docx", "Oferta Test 9.docx", "Oferta Test 11.docx",
    "Oferta Test 12.docx", "Oferta Test 13.docx", "Oferta Test 14.docx",
    "Oferta Test 15.docx", "Oferta- Test 16.docx", "Oferte Test 8.docx"
]

data = []
for doc_name in document_names:
    file_path = os.path.join(document_folder, doc_name)
    text = read_docx(file_path)

    cerinta = extract_section(text, "Solicitarea client:")
    oferta = extract_section(text, "Oferta pentru firma")

    if cerinta and oferta:
        data.append({
            "cerinta": cerinta,
            "oferta": oferta
        })

with open('data_exemple.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
