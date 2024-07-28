# Proiect de Generare a Ofertelor AI

Acest repository conține un proiect pentru generarea de oferte detaliate bazate pe cerințele clientului folosind un model de limbaj personalizat.

## Cuprins

- [Introducere](#introducere)
- [Dependențe](#dependențe)
- [Configurare](#configurare)
- [Antrenarea Modelului](#antrenarea-modelului)
- [Generarea Ofertelor](#generarea-ofertelor)

## Introducere

Acest proiect utilizează un model de limbaj pentru a genera oferte detaliate bazate pe cerințele clientului. Procesul implică antrenarea modelului pe un set de date și apoi utilizarea modelului antrenat pentru a genera oferte în mod eficient, în loturi.

## Dependențe

Pentru a rula acest proiect, trebuie să instalați următoarele dependențe:

- Python 3.7 sau mai mare
- Biblioteca Transformers de la Hugging Face
- Optuna pentru optimizarea hiperparametrilor
- Scikit-learn pentru procesarea datelor și evaluarea modelului
- Pandas pentru manipularea datelor
- Biblioteca Evaluate de la Hugging Face
- Torch (PyTorch)

Puteți instala aceste dependențe folosind `pip`:


pip install transformers optuna scikit-learn pandas evaluate torch


## Configurare

1. Clonați acest repository:


git clonehttps://github.com/rebeca27/generator
cd ai-offer-generation


2. Asigurați-vă că aveți directoarele necesare:


mkdir -p results/definitions results/suggestions results/propunere_structura


3. Plasați fișierele de date (`data_exemple.json`, `client_data.json`, `data_preturi.json`, `template_oferta.txt`) în directorul rădăcină al repository-ului.

## Antrenarea Modelului

Procesul de antrenare a modelului este gestionat de scriptul `train_model.py`. Acest script procesează datele, antrenează modelul și salvează modelele antrenate în directoarele specificate.

Pentru a antrena modelul, rulați:


python train_model.py


Acest script va:

- Încărca și preprocesa datele.
- Împărți datele în seturi de antrenament și evaluare.
- Antrena modelul folosind datele furnizate.
- Optimiza hiperparametrii folosind Optuna.
- Salva modelele antrenate în directorul `results`.

## Generarea Ofertelor

După ce modelul este antrenat, puteți genera oferte folosind scriptul `generate_offer.py`. Acest script încarcă modelele antrenate și generează oferte bazate pe cerințele clienților furnizate în `client_data.json`.

Pentru a genera oferte, rulați:


python generate_offer.py


Acest script va:

- Încărca datele clienților și modelele antrenate.
- Genera definiții, sugestii și propuneri de structură pentru fiecare cerință a clientului.
- Estima numărul de zile de dezvoltare și numărul de programatori necesari.
- Genera o ofertă completă pentru fiecare cerință a clientului folosind un șablon predefinit.
- Salva ofertele generate în fișiere text individuale.

## Structura Fișierelor

- `train_model.py`: Script pentru antrenarea modelului.
- `generate_offer.py`: Script pentru generarea ofertelor folosind modelul antrenat.
- `data_exemple.json`: Date de exemplu pentru antrenarea modelului.
- `client_data.json`: Datele cerințelor clienților pentru generarea ofertelor.
- `data_preturi.json`: Datele prețurilor pentru estimarea costurilor de dezvoltare.
- `template_oferta.txt`: Fișier șablon pentru generarea ofertei finale.
- `results/`: Directorul pentru stocarea modelelor antrenate.

## Note

- Asigurați-vă că aveți resurse hardware suficiente (CPU/GPU) pentru antrenarea modelului.
- Scripturile sunt configurate să utilizeze MPS (Apple Silicon) dacă este disponibil, în caz contrar, vor folosi CPU. Modificați configurația dispozitivului în scripturi dacă este necesar.

## Licență

Acest proiect este licențiat sub Licența MIT - vedeți fișierul [LICENSE](LICENSE) pentru detalii.

## Mulțumiri

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Optuna](https://github.com/optuna/optuna)
- [Scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [Evaluate by Hugging Face](https://github.com/huggingface/evaluate)
- [PyTorch](https://github.com/pytorch/pytorch)
