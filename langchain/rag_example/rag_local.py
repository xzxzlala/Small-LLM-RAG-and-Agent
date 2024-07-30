from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import numpy as np
from langchain_chroma import Chroma
from gpt4all import Embed4All
from typing import List
from chromadb import Documents, EmbeddingFunction, Embeddings
class MyEmbeddingFunction1(EmbeddingFunction):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        embedding_values = self.embedding_model.embed(texts)
        for i in range(len(texts)):
            embeddings[i] = embedding_values[i]
        return embeddings
    def embed_query(self, text: str) -> List[float]:
        embedding_values = self.embedding_model.embed(text)
        return embedding_values
    def __init__(self):
        self.embedding_model = Embed4All()
loader = TextLoader("/Users/michaelwu/langchain/data/QA.txt")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n"], chunk_size=50, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
print(all_splits)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=MyEmbeddingFunction1())
question = "我购买的东西大约多久能发货?"
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
from langchain_community.llms import LlamaCpp
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    #model_path="/Users/michaelwu/langchain/models/qwen1_5-0_5b-chat-q8_0.gguf",
    model_path="/Users/michaelwu/langchain/models/capybarahermes-2.5-mistral-7b.Q2_K.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnablePick
# Prompt
prompt = PromptTemplate.from_template(
    """你是一个智能客服，现在用户正在向你询问问题。参考以下标准问答。
    {context}
    请基于标准问答的内容回答用户问题
    {question} 
    """
)
# Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
#chain = {"docs": format_docs} | prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
    | prompt
    | llm
    | StrOutputParser()
)
# Run
question = "我购买的东西大约多久能发货?"
docs = vectorstore.similarity_search(question, k=1)
print("match:")
print(docs)
res = chain.invoke({"context": docs, "question": question})
print("res:")
print(res)
print("end")