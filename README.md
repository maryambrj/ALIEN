# Multi-Agent Architectures for Entity Relationship Classification

This repository contains all the code, data processing scripts, and evaluation tools for the paper:

**[“Comparative Analysis of AI Agent Architectures for Entity Relationship Classification”](https://doi.org/10.48550/arXiv.2506.02426)**

We implement and evaluate three agent-based LLM architectures for relation classification:

* **Generator-Reflector**
* **Hierarchical Multi-Agent**
* **Dynamic Example Generation (Relation Forger Agent)**

---

## 🔧 Project Structure

```
.
├── data_processing_scripts         
├── datasets                        
│   ├── core_data
│   ├── refind_data
│   └── semeval_2010_task8
│
├── few_shot_baseline           
│   ├── gemini                   
│   └── gpt                      
│
├── generator_reflector_agent   
├── hierarchical_multi_agent      
└── relation_forger_agent      
```

---

## 📖 Architectures

### 1. Generator-Reflector Agent

Implements a two-agent system:

* **Generator**: Proposes initial relation predictions
* **Reflector**: Provides critiques and refinements in an iterative loop

Uses LangGraph to manage feedback-driven refinement over up to 3 dialogue turns.

---

### 2. Hierarchical Multi-Agent

Implements:

* An **Orchestrator agent** that routes inputs based on context
* **Specialist agents** responsible for domain-specific relation subsets
* A dynamic feedback loop between Orchestrator and Specialists to improve predictions

---

### 3. Relation Forger Agent (Dynamic Example Generator)

This agent dynamically:

* Generates positive & adversarial negative examples
* Selects training-set examples using FAISS similarity search
* Constructs a tailored few-shot prompt per test instance

Then performs final classification using an enriched context.

---

## 📊 Datasets

* **REFinD** – financial relation classification
* **CORE** – scientific relation types from academic papers
* **SemEval 2010 Task 8** – general-domain semantic relations

Each dataset folder includes original and/or preprocessed `.csv` or `.json` files.

---

## 📈 Few-Shot Prompting Baselines

Located in: `few_shot_baseline/` and contains:

* Prompt templates for GPT (OpenAI) and Gemini (Google)
* Scripts for 0-shot, 3-shot, and 5-shot experiments

---

## ✅ Requirements

We recommend using a Python environment with the following packages:

```bash
langchain
langgraph
faiss-cpu
openai
google-generativeai
pandas
numpy
tqdm
```

Optional: Hugging Face Transformers, Claude SDK (Anthropic), etc., depending on model backend integration.

---

## 📄 Citation

If you use this codebase or ideas from our paper, please cite:

```bibtex
@inproceedings{berijanian2025comparative,
  title={Comparative Analysis of AI Agent Architectures for Entity Relationship Classification},
  author={Maryam Berijanian and Kuldeep Singh and Amin Sehati},
  booktitle={arXiv},
  url={https://arxiv.org/abs/2506.02426}, 
  year={2025}
}
```

---

## 📬 Contact

For questions, open an issue or contact:

* Maryam Berijanian — [berijani@msu.edu](mailto:berijani@msu.edu)
* Kuldeep Singh — [singhku2@msu.edu](mailto:singhku2@msu.edu)
* Amin Sehati – [amin@aminsehati.com](mailto:amin@aminsehati.com)
