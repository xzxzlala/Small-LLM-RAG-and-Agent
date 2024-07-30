# This is a long document we can split up.
with open("./policy/policy.txt") as f:
    policy = f.read()
from langchain_experimental.text_splitter import SemanticChunker
#from langchain_openai.embeddings import OpenAIEmbeddings
from transformers import AutoTokenizer, AutoModel
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
#text_splitter = CharacterTextSplitter(
#    separator="\n\n",
#    chunk_size=1000,
#    chunk_overlap=200,
#    length_function=len,
#    is_separator_regex=False,
#)
#text_splitter = RecursiveCharacterTextSplitter(
#    # Set a really small chunk size, just to show.
#    chunk_size=100,
#    chunk_overlap=20,
#    length_function=len,
#    is_separator_regex=False,
#)
#text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#    chunk_size=100, chunk_overlap=0
#)
from gpt4all import Embed4All
from typing import List
from chromadb import Documents, EmbeddingFunction, Embeddings
class MyEmbeddingFunction(EmbeddingFunction):
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
model = AutoModel.from_pretrained("./demo/Qwen1.5-0.5B-Chat/embedding")
text_splitter = SemanticChunker(MyEmbeddingFunction(), breakpoint_threshold_type="standard_deviation", sentence_split_regex= "\n")
docs = text_splitter.create_documents([policy])
for doc in docs:
    print(doc)
