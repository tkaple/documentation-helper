from dotenv import load_dotenv

load_dotenv()

from typing import Any, Dict, List

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from consts import INDEX_NAME


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(model_name="gpt-4o", verbose=True, temperature=0)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    combine_docs_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=combine_docs_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result
