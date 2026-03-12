import os
import json
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# -------------------------------------------------
# 1. MEDICAL GUIDELINE DATA
# -------------------------------------------------
medical_docs = [
    Document(page_content="Metformin is recommended as first-line pharmacologic therapy for Type 2 Diabetes unless contraindicated."),
    Document(page_content="Lifestyle interventions such as diet modification, weight loss, and physical activity should begin at diagnosis."),
    Document(page_content="Insulin therapy should be initiated when oral medications fail to control blood glucose."),
    Document(page_content="Blood glucose monitoring is recommended for patients receiving insulin therapy."),
]


# -------------------------------------------------
# 2. VECTOR STORE
# -------------------------------------------------
embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(medical_docs, embeddings)

retriever_initial = vectorstore.as_retriever(search_kwargs={"k":2})
retriever_corrected = vectorstore.as_retriever(search_kwargs={"k":4})


# -------------------------------------------------
# 3. LLM
# -------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"]
)


# -------------------------------------------------
# 4. FORMAT CONTEXT
# -------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -------------------------------------------------
# 5. CONTEXT EVALUATION PROMPT
# -------------------------------------------------
evaluation_prompt = ChatPromptTemplate.from_template("""
You are a corrective RAG evaluator for clinical guideline retrieval.

Query: {user_query}

Retrieved Context:
{retrieved_context}

Evaluate the context:

1. Relevance score (0-1)
2. Completeness score (0-1)
3. Accuracy score (0-1)
4. Specificity score (0-1)

Determine overall_quality: Excellent / Good / Fair / Poor

Correction logic:
If overall_quality is Fair or Poor:
- action: RETRIEVE_AGAIN
- refined_query: improved search query
- reasoning: why retrieval must be corrected
- confidence: Low

If overall_quality is Excellent or Good:
- action: PROCEED_WITH_ANSWER
- reasoning: why context is sufficient
- confidence: High or Medium

Return JSON only.
""")


def evaluate_context(query, context):
    chain = evaluation_prompt | llm
    response = chain.invoke({
        "user_query": query,
        "retrieved_context": context
    }).content

    return json.loads(response)


# -------------------------------------------------
# 6. FINAL ANSWER PROMPT
# -------------------------------------------------
answer_prompt = ChatPromptTemplate.from_template("""
You are a clinical decision support assistant.

Context Quality: {quality}
Confidence Level: {confidence}

Use ONLY the context below.

Context:
{context}

Question:
{query}

Response Format:

Context Quality: {quality}
Confidence Level: {confidence}
Answer: <clinical answer>
""")


def generate_answer(query, context, quality, confidence):
    chain = answer_prompt | llm
    response = chain.invoke({
        "query": query,
        "context": context,
        "quality": quality,
        "confidence": confidence
    })
    return response.content


# -------------------------------------------------
# 7. CORRECTIVE RAG PIPELINE
# -------------------------------------------------
def corrective_rag(query):

    print("\nDoctor Query:", query)

    docs = retriever_initial.invoke(query)
    context = format_docs(docs)

    evaluation = evaluate_context(query, context)

    print("\nContext Evaluation:", evaluation)

    if evaluation["overall_quality"] in ["Fair", "Poor"]:

        refined_query = evaluation["refined_query"]

        print("\nCorrection triggered")
        print("Refined Query:", refined_query)

        docs = retriever_corrected.invoke(refined_query)
        context = format_docs(docs)

        evaluation = evaluate_context(query, context)

    answer = generate_answer(
        query,
        context,
        evaluation["overall_quality"],
        evaluation["confidence"]
    )

    return answer


# -------------------------------------------------
# 8. RUN SAMPLE QUESTIONS
# -------------------------------------------------
queries = [
    "What is the first line medication for Type 2 Diabetes?",
    "When should insulin therapy be started?"
]

for q in queries:
    result = corrective_rag(q)
    print("\nFinal Answer:\n", result)
