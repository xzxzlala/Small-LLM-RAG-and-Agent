from transformers import AutoTokenizer, AutoModel
import torch
def compute_embeddings(text):
    #tokenizer = AutoTokenizer.from_pretrained("./model2/tokenizer") 
    #model = AutoModel.from_pretrained("./model2/embedding")
    tokenizer = AutoTokenizer.from_pretrained("/Users/michaelwu/modelscope/Qwen1.5-0.5B-Chat/tokenizer")
    model = AutoModel.from_pretrained("/Users/michaelwu/modelscope/Qwen1.5-0.5B-Chat/embedding")
    # model = Swift....
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True) 
    # Generate the embeddings 
    with torch.no_grad():    
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    return embeddings.tolist()
print(compute_embeddings("adc"))