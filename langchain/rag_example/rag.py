import getpass
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["USER_AGENT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = ""
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-8b-8192")

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import AutoModel, AutoTokenizer
import torch
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import List

class MyEmbeddingFunction(EmbeddingFunction):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        embedding_values = self.embedding_model.encode(texts)
        for i in range(len(texts)):
            embeddings[i] = embedding_values[i].tolist()
        return embeddings
    def embed_query(self, text: str) -> List[float]:
        embedding_values = self.embedding_model.encode(text)
        return embedding_values.tolist()
    def __init__(self):
        self.embedding_model = AutoModel.from_pretrained("./Qwen1.5-0.5B-Chat/embedding")
if __name__ == '__main__':
    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())    
    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain.invoke("What is Task Decomposition?")