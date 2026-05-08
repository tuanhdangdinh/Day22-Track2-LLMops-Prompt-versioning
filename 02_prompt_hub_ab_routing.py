"""Step 2 — Prompt Hub & A/B Routing.

Pushes two prompt versions to LangSmith Prompt Hub, pulls them back,
and routes 50 queries deterministically via MD5 hash.
Run: python 02_prompt_hub_ab_routing.py
"""

import os
import hashlib
from pathlib import Path

from config import setup_langsmith, LANGSMITH_API_KEY, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL
setup_langsmith()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

from qa_pairs import SAMPLE_QUESTIONS

# ── Prompt versions ──────────────────────────────────────────────────────────

SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "{question}"),
])

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "{question}"),
])

PROMPT_V1_NAME = "rag-prompt-v1"
PROMPT_V2_NAME = "rag-prompt-v2"

# ── LLM + Embeddings ─────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


def build_vectorstore() -> FAISS:
    text = Path("data/knowledge_base.txt").read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, embeddings)


def push_prompts_to_hub(client: Client):
    """Upload both prompt versions to LangSmith Prompt Hub."""
    for name, prompt, desc in [
        (PROMPT_V1_NAME, PROMPT_V1, "V1 – concise 2-4 sentence answers"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2 – structured expert 3-5 sentence answers"),
    ]:
        try:
            url = client.push_prompt(name, object=prompt, description=desc)
            print(f"✅ Pushed '{name}' → {url}")
        except Exception as e:
            print(f"⚠️  Could not push '{name}': {e}")


def pull_prompts_from_hub(client: Client) -> dict:
    """Pull both prompt versions from Hub; fall back to local on error."""
    prompts = {}
    for name, fallback in [(PROMPT_V1_NAME, PROMPT_V1), (PROMPT_V2_NAME, PROMPT_V2)]:
        try:
            prompts[name] = client.pull_prompt(name)
            print(f"↓ Pulled '{name}' from Hub")
        except Exception:
            prompts[name] = fallback
            print(f"ℹ️  Using local fallback for '{name}'")
    return prompts


def get_prompt_version(request_id: str) -> str:
    """Deterministic 50/50 routing: even MD5 → V1, odd → V2."""
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if h % 2 == 0 else PROMPT_V2_NAME


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """Run RAG chain with the selected prompt version."""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version}


def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    client = Client(api_key=LANGSMITH_API_KEY)

    print("\n── Pushing prompts to Hub ──")
    push_prompts_to_hub(client)

    print("\n── Pulling prompts from Hub ──")
    prompts = pull_prompts_from_hub(client)

    print("\n── Building vectorstore ──")
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("\n── Running 50 A/B queries ──\n")
    v1_count = v2_count = 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, question, version_tag)
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")
        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

    print(f"\n── Routing Summary ──")
    print(f"  prompt-v1 handled: {v1_count} queries")
    print(f"  prompt-v2 handled: {v2_count} queries")
    print(f"\n✅ Step 2 complete — {len(SAMPLE_QUESTIONS)} more traces sent to LangSmith.")


if __name__ == "__main__":
    main()
