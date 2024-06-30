import os
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

def ingest_docs()->None:

    loader = ReadTheDocsLoader(
        path="langchain-docs/api.python.langchain.com/en/latest/chains", encoding="utf-8"
    )
    raw_documents = loader.load()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:")
        new_url = new_url.replace("\\", "//")
        doc.metadata.update({"source": new_url})
    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(documents, embeddings, index_name="langchain-doc-index1536")
    print("****Loading to vectorstore done ***")
    

if __name__ == "__main__":
    ingest_docs()
