import numpy as np
import requests
import os
import json, ast
import re
import uuid
import torch
from modelscope.models import Model
from transformers import AutoTokenizer, AutoModel
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from llama_cpp import Llama
import sys

def document_chunker(directory_path,
                     model_name,
                     paragraph_separator='\n',
                     chunk_size=1024,
                     separator=' ',
                     secondary_chunking_regex=r'\S+?[\.,;!?]',
                     chunk_overlap=0):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load tokenizer for the specified model
    documents = {}  # Initialize dictionary to store results

    # Read each file in the specified directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        base = os.path.basename(file_path)
        sku = os.path.splitext(base)[0]
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Generate a unique identifier for the document
            doc_id = str(uuid.uuid4())

            # Process each file using the existing chunking logic
            paragraphs = re.split(paragraph_separator, text)
            all_chunks = {}
            for paragraph in paragraphs:
                words = paragraph.split(separator)
                current_chunk = ""
                chunks = []
                for word in words:
                    new_chunk = current_chunk + (separator if current_chunk else '') + word
                    if len(tokenizer.tokenize(new_chunk)) <= chunk_size:
                        current_chunk = new_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = word

                if current_chunk:
                    chunks.append(current_chunk)
                refined_chunks = []
                for chunk in chunks:
                    if len(tokenizer.tokenize(chunk)) > chunk_size:
                        sub_chunks = re.split(secondary_chunking_regex, chunk)
                        sub_chunk_accum = ""
                        for sub_chunk in sub_chunks:
                            if sub_chunk_accum and len(tokenizer.tokenize(sub_chunk_accum + sub_chunk + ' ')) > chunk_size:
                                refined_chunks.append(sub_chunk_accum.strip())
                                sub_chunk_accum = sub_chunk
                            else:
                                sub_chunk_accum += (sub_chunk + ' ')
                        if sub_chunk_accum:
                            refined_chunks.append(sub_chunk_accum.strip())
                    else:
                        refined_chunks.append(chunk)

                final_chunks = []
                if chunk_overlap > 0 and len(refined_chunks) > 1:
                    for i in range(len(refined_chunks) - 1):
                        final_chunks.append(refined_chunks[i])
                        overlap_start = max(0, len(refined_chunks[i]) - chunk_overlap)
                        overlap_end = min(chunk_overlap, len(refined_chunks[i+1]))
                        overlap_chunk = refined_chunks[i][overlap_start:] + ' ' + refined_chunks[i+1][:overlap_end]
                        final_chunks.append(overlap_chunk)
                    final_chunks.append(refined_chunks[-1])
                else:
                    final_chunks = refined_chunks

                # Assign a UUID for each chunk and structure it with text and metadata
                for chunk in final_chunks:
                    chunk_id = str(uuid.uuid4())
                    all_chunks[chunk_id] = {"text": chunk, "metadata": {"file_name":sku}}  # Initialize metadata as dict

            # Map the document UUID to its chunk dictionary
            documents[doc_id] = all_chunks

    return documents

def compute_embeddings(text):
    #tokenizer = AutoTokenizer.from_pretrained("./model2/tokenizer") 
    #model = AutoModel.from_pretrained("./model2/embedding")
    tokenizer = AutoTokenizer.from_pretrained("./Qwen1.5-0.5B-Chat/tokenizer") 
    model = AutoModel.from_pretrained("./Qwen1.5-0.5B-Chat/embedding")
    # model = Swift....

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True) 
    
    # Generate the embeddings 
    with torch.no_grad():    
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze()

    return embeddings.tolist()

def create_vector_store(doc_store):
    vector_store = {}
    for doc_id, chunks in doc_store.items():
        doc_vectors = {}
        for chunk_id, chunk_dict in chunks.items():
            # Generate an embedding for each chunk of text
            doc_vectors[chunk_id] = compute_embeddings(chunk_dict.get("text"))
        # Store the document's chunk embeddings mapped by their chunk UUIDs
        vector_store[doc_id] = doc_vectors
    return vector_store

def compute_matches(vector_store, query_str, top_k):
    """
    This function takes in a vector store dictionary, a query string, and an int 'top_k'.
    It computes embeddings for the query string and then calculates the cosine similarity against every chunk embedding in the dictionary.
    The top_k matches are returned based on the highest similarity scores.
    """
    # Get the embedding for the query string
    query_str_embedding = np.array(compute_embeddings(query_str))
    scores = {}

    # Calculate the cosine similarity between the query embedding and each chunk's embedding
    for doc_id, chunks in vector_store.items():
        for chunk_id, chunk_embedding in chunks.items():
            chunk_embedding_array = np.array(chunk_embedding)
            # Normalize embeddings to unit vectors for cosine similarity calculation
            norm_query = np.linalg.norm(query_str_embedding)
            norm_chunk = np.linalg.norm(chunk_embedding_array)
            if norm_query == 0 or norm_chunk == 0:
                # Avoid division by zero
                score = 0
            else:
                score = np.dot(chunk_embedding_array, query_str_embedding) / (norm_query * norm_chunk)

            # Store the score along with a reference to both the document and the chunk
            scores[(doc_id, chunk_id)] = score

    # Sort scores and return the top_k results
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    top_results = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_scores]

    return top_results

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def open_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def retrieve_docs(doc_store, matches, top_k = 1):
    docs =  ""
    for i in range(top_k):
        docs = docs + doc_store[matches[i][0]][matches[i][1]]['text']
    return docs
#Google search engine
def search(search_term):
    vec_store = open_json('./ragJson/vector_store.json')    
    docs =  open_json('./ragJson/doc_store.json')
    matches = compute_matches(vector_store=vec_store,
                query_str=search_term,top_k=3)
    retrieved_docs = retrieve_docs(docs, matches)
    return retrieved_docs
#Calculator
from py_expression_eval import Parser
parser = Parser()
def calculator(str):
    return parser.parse(str).evaluate({})
def extract_action_and_input(text):
      action_pattern = r"Action: (.+?)\n"
      input_pattern = r"Action Input: (.+?)\n"
      action = re.findall(action_pattern, text)
      action_input = re.findall(input_pattern, text)
      return action, action_input
def Stream_agent(prompt):
    model_name = "Qwen1.5-0.5B-Chat"
    model_type = ModelType.qwen1half_0_5b_chat
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    template_type = get_default_template_type(model_type)
    template = get_template(template_type, tokenizer)
    llm = Model.from_pretrained(model_name)
    llm.generation_config.max_new_tokens = 128
    query = f"""
        回答以下问题并尽可能遵守以下命令。

        您可以使用以下工具：
        Search: 回答有关时事的问题.
        Calculator: 回答数学问题时很有用。例如:2 + 2 * 3
        Response To Human: 对正在与您交谈的人做出回答时。

        您将收到来自用户的消息，
        然后您应该选择一个工具来使用并按照以下格式
        (回答中请包括Action:以及Action Input:,注意Action Input:后有双引号):

        Action: 在以下列表中选择一个, 注意每个字符都要完全一致。
        [“Search”,“Calculator”, "Response To Human"]

        Action Input: "要发送到工具的输入"

        当你最终准备回答用户的问题时，同样使用上述模版。
        在Action中选择“Response To Human”，并在Action Iuput中写入对用户的回答。

        以下是用户的问题：
        {prompt}
        """
    while True:
        print(f"query: {query}")
        response_text, _ = inference(llm, template, query)    
        action, action_input = extract_action_and_input(f"{response_text}\n")
        print(f"res: {response_text}\nend")
        print(f"action: {action}")
        print(f"action_input: {action_input}")
        if action==[] or action_input==[]:
            print(f"final_answer: {response_text}")
            break
        if action[-1] == "Search":
            tool = search
        elif action[-1] == "Calculator":
            tool = calculator
        elif action[-1] == "Response To Human":
            print(f"final_answer: {action_input[-1]}")
            break
        observation = tool(action_input[-1])
        query += "以下是调用工具的结果\n"
        query += f"Action: {action[-1]}, Action Input: {action_input[-1]}, "
        query  += f"Action Result: {str(observation)}"
        #messages.extend([
        #    { "role": "system", "content": response_text },
        #    { "role": "user", "content": f"Observation: {observation}" },
        #])
if __name__  == '__main__':
    #docs = document_chunker(directory_path='./ragData',
    #                        model_name=model_name,
    #                        chunk_size=256)
    #vec_store = create_vector_store(docs)
    #save_json('./ragJson/doc_store.json', docs)
    #save_json('./ragJson/vector_store.json', vec_store)
    #Stream_agent("用计算器计算2+2*3+10*2等于几")
    Stream_agent("请搜索，你们的衣服质量怎么样啊?")