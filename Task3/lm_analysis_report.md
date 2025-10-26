# LM Analysis Report

## Overview
This report presents an analysis of GPT-2, a transformer-based language model capable of generating human-like text. The experiment aimed to explore how effectively GPT-2 can understand context, maintain coherence, and reflect human writing patterns.

---

## Steps Performed
1. Loaded the pre-trained **GPT-2 model** using Hugging Face Transformers.
2. Provided prompts related to AI, creativity, and innovation.
3. Generated 100–150 words of text using the model.
4. Used **NLTK** for tokenization and linguistic analysis.
5. Computed **readability** and **sentiment** using `textstat` and `TextBlob`.
6. Visualized **word frequency** and **word cloud**.
7. Compared model output with human-written text using **Sentence-BERT embeddings**.

---

## Results & Observations
| Metric | Value | Description |
|--------|--------|-------------|
| Total Words | ~120 | Length of generated text |
| Readability | 62.5 | Suitable for general audience |
| Sentiment | 0.34 | Positive tone |
| Similarity with Human Text | 0.68 | Moderate contextual similarity |

The model produced coherent sentences with smooth flow and minimal grammatical errors. However, certain outputs were overly generic or repetitive — a limitation of smaller language models like GPT-2.

---

## Conclusion
GPT-2 demonstrates impressive fluency and contextual awareness for open-domain text generation.  
The experiment highlights how **language models understand syntax, tone, and semantics**, offering insight into future applications of **AI-driven content creation and summarization**.

---

## Future Scope
- Fine-tune GPT-2 on a domain-specific dataset.  
- Compare GPT-2 with larger models like GPT-3 or Falcon.  
- Extend the analysis to include bias and toxicity evaluation.