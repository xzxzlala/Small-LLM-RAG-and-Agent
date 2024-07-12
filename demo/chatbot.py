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
from ocrmac import ocrmac
from PIL import Image
def get_invoice_info(path):
    img = Image.open(path)
    code = (1320,56,1544,96)
    date = (1319,105,1478,138)
    buyer_name_crop_range = (189,249,498,282)
    #buyer_id_crop_range = (469,332,712,363)
    seller_name_crop_range = (944,249,1230,287)
    seller_id_crop_range = (1232,333,1466,362)
    total_money_crop_range = (1221,734,1340,767)
    crop_ranges = [code, date, buyer_name_crop_range, seller_name_crop_range, seller_id_crop_range, total_money_crop_range]
    information = []
    information_title = ["开票号码","开票日期","购买方名称", "销售方名称", "销售方统一社会信用代码", "价税合计"]
    for crop_range in crop_ranges:
        tmp = ocrmac.OCR(img.crop(crop_range),language_preference=['zh-Hans']).recognize()
        information.append(tmp[0][0])
    return information_title, information
import os
class ChatBot:
    def __init__(self, update="n"):
        self.model_name = "Qwen1.5-0.5B-Chat"
        if update.lower() == 'y':
            docs = document_chunker(directory_path='./ragData', model_name=self.model_name, chunk_size=256)
            vec_store = create_vector_store(docs)
            save_json('./ragJson/doc_store.json', docs)
            save_json('./ragJson/vector_store.json', vec_store)
        
    def chat(self, query_str, step=None, additional_input=None):
        vec_store = open_json('./ragJson/vector_store.json')
        docs = open_json('./ragJson/doc_store.json')
        matches = compute_matches(vector_store=vec_store, query_str=query_str, top_k=3)
        retrieved_docs = retrieve_docs(docs, matches)
        model_type = ModelType.qwen1half_0_5b_chat
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        template_type = get_default_template_type(model_type)
        template = get_template(template_type, tokenizer)
        query = f"""你是一个智能客服，现在用户正在向你询问一些问题。以下是一般客服的标准问答。
        {retrieved_docs} 
        请基于标准问答的内容回答用户问题
        用户问题: {query_str} 
        """
        llm = Model.from_pretrained(self.model_name)
        llm.generation_config.max_new_tokens = 128
        first_matched_doc = docs[matches[0][0]][matches[0][1]]
        print(step)
        print((step is None))
        print(first_matched_doc)
        if step is None or step == "":
            if '录入一下发票信息' in first_matched_doc['text']:
                return "我发现您需要读取发票文件, 请提供文件路径(比如: ./invoice_example/1.png):", 'file_path'
            elif '发票信息全部录入' in first_matched_doc['text']:
                return "我发现您需要读取全部发票文件并修改知识库, 请提供文件夹路径(比如: invoice_example):", 'folder_path'
            else:
                response, history = inference(llm, template, query)
                return response.replace("\n", "<br>"), None  # Replace newlines with <br> for HTML
        else:
            if step == 'file_path':
                print("add input:")
                print(additional_input)
                infor_titles, infors = get_invoice_info(additional_input)
                response = "录入的发票信息如下:<br>"
                for (infor_title, infor) in zip(infor_titles, infors):
                    response += f"{infor_title}: {infor} <br>"
                return response, None
            elif step == 'folder_path':
                files = os.listdir(additional_input)
                with open("./ragData/invoice.txt", 'w') as f:
                    f.write("{\n")
                    for file in files:
                        infor_titles, infors = get_invoice_info(os.path.join(additional_input, file))
                        text = "\""
                        for (infor_title, infor) in zip(infor_titles, infors):
                            text += f"{infor_title}: {infor} "
                        text += "\""
                        f.write(f"{text} : {text}\n\n")
                    f.write("}\n")
                response = "知识库文件添加成功，现在开始更新知识库。<br>"
                docs = document_chunker(directory_path='./ragData', model_name=self.model_name, chunk_size=256)
                vec_store = create_vector_store(docs)
                save_json('./ragJson/doc_store.json', docs)
                save_json('./ragJson/vector_store.json', vec_store)
                response += "知识库更新完毕。<br>"
                return response, None
    