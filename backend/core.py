import os
from typing import Any
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.vectorstores.pinecone import Pinecone as PineconeLangChain
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = "langchain-doc-index1536"

def run_llm(query: str, chat_history: list[dict[str, Any]] ) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeLangChain.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    """
    This chain first does a retrieval step to fetch relevant documents, then passes those documents into an LLM to generate a response.
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})
    """
    """
    This chain can be used to have conversations with a document. 
    It takes in a question and (optional) previous conversation history. 
    If there is a previous conversation history, it uses an LLM to rewrite the conversation into a query to send to a retriever 
    (otherwise it just uses the newest user input). It then fetches those documents and passes them (along with the conversation) 
    to an LLM to respond."""
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))
