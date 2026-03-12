
import os
import json
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# 1. SAMPLE DOCUMENTS
# =========================================================
policy_docs = [
    Document(page_content="""
Employees are entitled to 12 casual leave days per calendar year.
Unused casual leave cannot be carried forward to the next year.
    """),
    Document(page_content="""
Employees are entitled to 10 sick leave days per calendar year.
A medical certificate is required if sick leave exceeds 2 consecutive days.
    """),
    Document(page_content="""
Travel reimbursement for domestic business travel is capped at INR 5,000 per day,
including lodging and meals, subject to manager approval.
    """),
    Document(page_content="""
Work from home may be approved for up to 2 days per week depending on role requirements
and manager approval.
    """),
    Document(page_content="""
Maternity leave is available for 26 weeks in accordance with company policy and applicable law.
    """),
]


# =========================================================
# 2. BUILD VECTOR STORE
# =========================================================
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(policy_docs, embeddings)

retriever_initial = vectorstore.as_retriever(search_kwargs={"k": 2})
retriever_corrected = vectorstore.as_retriever(search_kwargs={"k": 4})


# =========================================================
# 3. INITIALIZE LLM
# =========================================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY")
)


# =========================================================
# 4. HELPER: FORMAT DOCS AS CONTEXT
# =========================================================
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join([doc.page_content.strip() for doc in docs])


# =========================================================
# 5. CONTEXT EVALUATION PROMPT
# =========================================================
evaluation_prompt = ChatPromptTemplate.from_template("""
You are a corrective RAG evaluator.

Query:
{user_query}

Retrieved Context:
{retrieved_context}

Evaluate:
1. Relevance score (0-1)
2. Completeness score (0-1)
3. Accuracy score (0-1)
4. Specificity score (0-1)

overall_quality: Excellent / Good / Fair / Poor

Correction logic:
If overall_quality is Fair or Poor:
- action: RETRIEVE_AGAIN
- refined_query: improved search query
- reasoning: why correction is needed
- confidence: Low

If overall_quality is Excellent or Good:
- action: PROCEED_WITH_ANSWER
- confidence: High or Medium

Return JSON only.
""")


def evaluate_context(user_query: str, retrieved_context: str) -> Dict[str, Any]:
    chain = evaluation_prompt | llm
    raw_response = chain.invoke({
        "user_query": user_query,
        "retrieved_context": retrieved_context
    }).content

    try:
        return json.loads(raw_response)
    except:
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        return json.loads(raw_response[start:end])


# =========================================================
# 6. FINAL ANSWER PROMPT
# =========================================================
answer_prompt = ChatPromptTemplate.from_template("""
You are an HR policy assistant.

Context Quality: {context_quality}
Confidence Level: {confidence}

Context:
{retrieved_context}

Question:
{user_query}

Response Format:
Context Quality: [Excellent/Good/Fair/Poor]
Confidence Level: [High/Medium/Low]
Answer: <your grounded answer>
""")


def generate_answer(user_query: str, retrieved_context: str, context_quality: str, confidence: str) -> str:
    chain = answer_prompt | llm
    response = chain.invoke({
        "user_query": user_query,
        "retrieved_context": retrieved_context,
        "context_quality": context_quality,
        "confidence": confidence
    })
    return response.content


# =========================================================
# 7. CORRECTIVE RAG PIPELINE
# =========================================================
def corrective_rag(user_query: str) -> Dict[str, Any]:
    print(f"\nUser Query: {user_query}")

    initial_docs = retriever_initial.invoke(user_query)
    initial_context = format_docs(initial_docs)

    print("\nInitial Retrieved Context:")
    print(initial_context)

    evaluation = evaluate_context(user_query, initial_context)

    print("\nEvaluation Result:")
    print(json.dumps(evaluation, indent=2))

    final_context = initial_context
    final_docs = initial_docs

    if evaluation["overall_quality"] in ["Fair", "Poor"]:
        refined_query = evaluation.get("refined_query", user_query)

        print("\nCorrection Triggered")
        print("Refined Query:", refined_query)

        corrected_docs = retriever_corrected.invoke(refined_query)
        final_docs = corrected_docs
        final_context = format_docs(corrected_docs)

        print("\nCorrected Retrieved Context:")
        print(final_context)

        evaluation = evaluate_context(user_query, final_context)

        print("\nRe-Evaluation Result:")
        print(json.dumps(evaluation, indent=2))

    final_answer = generate_answer(
        user_query=user_query,
        retrieved_context=final_context,
        context_quality=evaluation["overall_quality"],
        confidence=evaluation["confidence"]
    )

    return {
        "user_query": user_query,
        "retrieved_documents": [doc.page_content for doc in final_docs],
        "evaluation": evaluation,
        "final_answer": final_answer
    }


# =========================================================
# 8. RUN EXAMPLES
# =========================================================
if __name__ == "__main__":
    queries = [
        "How many casual leave days do employees get?",
        "Can unused casual leave be carried forward?",
        "What is the daily travel reimbursement limit for domestic business travel?",
        "How many paternity leave days are available?"
    ]

    for q in queries:
        result = corrective_rag(q)

        print("\n" + "=" * 70)
        print("FINAL OUTPUT")
        print("=" * 70)
        print(result["final_answer"])
        print("=" * 70)
