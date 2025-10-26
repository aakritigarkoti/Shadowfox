# 🧠 Task 3 – GPT-2 Text Generation and Analysis

## 📘 Overview
This task focuses on **Language Model (LM) Analysis** using the **GPT-2 model** from Hugging Face’s `transformers` library.  
The objective is to:
- Generate human-like text given a prompt.
- Perform basic linguistic analysis on the generated text (sentence, word, and token counts).
- Understand how pre-trained transformer models can be used for downstream NLP tasks.

This notebook demonstrates how a model like GPT-2 understands language patterns, predicts word sequences, and produces coherent paragraphs.

---

## 📂 Project Structure

Task3/
┣ 📜 lm_analysis.ipynb ← Jupyter Notebook for GPT-2 text generation and analysis
┣ 📜 lm_analysis_report.md ← Model analysis report
┣ 📜 requirements.txt ← Dependencies used in this task
┣ 📁 screenshot/ ← Folder containing execution proof
┃ ┣ 🖼️ 1.png ← GPT-2 output screenshot
┃ ┗ 🖼️ 2.png ← Folder structure / proof screenshot
┗ 📘 README.md ← This documentation file

yaml
Copy code

---

## 🧩 Technologies Used
| Category | Tool/Library | Purpose |
|-----------|---------------|----------|
| 💻 Programming | **Python 3.10+** | Core language |
| 🤖 NLP Model | **GPT-2 (via Hugging Face Transformers)** | Text generation |
| 🔠 Tokenization | **NLTK** | Sentence and word tokenization |
| ⚙️ Environment | **Jupyter Notebook** | Interactive development |

---

## 🚀 Step-by-Step Workflow

### **1️⃣ Environment Setup**
Create a virtual environment and install dependencies:
```bash
pip install torch transformers nltk
Activate environment:

bash
Copy code
& D:/Shadowfox/venv/Scripts/Activate.ps1
2️⃣ Launch Jupyter Notebook
Run:

bash
Copy code
jupyter notebook
Open the file lm_analysis.ipynb from the Jupyter dashboard.

3️⃣ Import Libraries
In the notebook, import the following:

python
Copy code
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
Also, download the required tokenizer data:

python
Copy code
nltk.download('punkt')
nltk.download('punkt_tab')
4️⃣ Load Model and Generate Text
python
Copy code
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "Artificial Intelligence will change the future because"
output = generator(prompt, max_length=100, num_return_sequences=1)
print(output[0]['generated_text'])
This generates a continuation of the given prompt using GPT-2.

5️⃣ Analyze Output
Perform basic linguistic analysis:

python
Copy code
generated_text = output[0]['generated_text']
tokens = generator.tokenizer.encode(generated_text)
num_tokens = len(tokens)

sentences = sent_tokenize(generated_text)
words = word_tokenize(generated_text)

print("📈 GPT-2 Output Analysis")
print("Total Tokens:", num_tokens)
print("Total Words:", len(words))
print("Total Sentences:", len(sentences))
📸 Screenshots
Execution proofs and notebook output are available inside the screenshot/ folder:

Screenshot	Description
GPT-2 output and text analysis
Folder view and environment confirmation

📊 Sample Output
vbnet
Copy code
📈 GPT-2 Output Analysis
--------------------------------------------------
Total Tokens: 128
Total Words: 119
Total Sentences: 7

Sample Output:
Artificial Intelligence will change the future because it's not just a piece of software or an idea, it's actually a real-world phenomenon.
The technology is already happening...
📄 lm_analysis_report.md
This file contains:

Summary of how GPT-2 works

Observations from the generated output

Insights about model fluency, coherence, and token usage

Limitations (e.g., random generations, lack of factual accuracy)

⚙️ requirements.txt
nginx
Copy code
torch
transformers
nltk
🏁 Results and Learnings
GPT-2 can generate contextually relevant text with surprising fluency.

Model performance depends on prompt quality and max_length.

Simple analysis using nltk helps evaluate linguistic richness of generated output.

This task strengthens understanding of Transformer architecture and text generation pipelines.

📚 References
Hugging Face Transformers Documentation

NLTK Official Docs

OpenAI GPT-2 Model Card

✅ Status
Completed Successfully ✅

Author: Aakriti Garkoti
Project: Shadowfox – Task 3 (Language Model Analysis)
