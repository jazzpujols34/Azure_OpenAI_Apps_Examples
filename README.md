# Azure OpenAI Streamlit Apps

This repository contains three powerful applications developed using Azure OpenAI and Streamlit to provide various functionalities ranging from call center simulations to smart contract analysis and medical document searching.

## 1. Call Center App

### File: `call_center_app.py`

**Overview:**  
This app simulates a call center environment where the OpenAI model assists in generating appropriate responses for call transcripts.

It's capable of producing emails in response to the calls and suggests measures to improve customer service quality.

**Features:**

- Call Transcript Analysis

- Ticket Creation

- Email Generation for Responses

- Suggestions for Service Quality Improvement

**How to Run:**
```bash
streamlit run call_center_app.py
```

## 2. Contract Analyzer App


### File: `contract_analyzer_app.py`

**Overview:**

Analyze contracts with ease! Extract key terms, ask questions about the contract's content, identify potential issues, and even generate new contract templates based on user input.

**Features:**

- Key Term Extraction

- Contract Question Answering

- Potential Issue Identification

- Contract Template Generation

**How to Run:**

```bash
streamlit run contract_analyzer_app.py
```

## 3. Medical Smart Search App

### File: medical_smart_search_app.py

**Overview:**  

Search through medical documents with AI assistance! Ask questions about the content of a selected document and get precise answers.

**Features:**

- Dynamic PDF Selection from ./data directory

- AI-powered Medical Document Search

- Contextual Question Answering

**How to Run:**

```bash
streamlit run medical_smart_search_app.py
```

## Setup:

1. Ensure you have streamlit and openai Python packages installed.

2. Setup the necessary configurations in secrets.toml.

3. Place the PDF documents you want to search in the ./data directory for the Medical Smart Search App.

4. Run the desired app using the respective command shown above.