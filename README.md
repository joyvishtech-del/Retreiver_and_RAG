
# Corrective RAG Example

This project demonstrates a **Corrective Retrieval-Augmented Generation (cRAG)** pipeline.

Corrective RAG improves traditional RAG by evaluating the quality of retrieved context before answering.

Pipeline:

User Query → Retrieval → Context Evaluation → (Optional) Correct Retrieval → Final Answer

## Setup

Install dependencies:

pip install -r requirements.txt

Set OpenAI API key:

Linux/Mac:
export OPENAI_API_KEY="your_api_key"

Windows:
setx OPENAI_API_KEY "your_api_key"

Run the project:

python app.py
