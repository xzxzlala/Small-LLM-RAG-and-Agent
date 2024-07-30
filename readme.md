# Introduction for each file and folder
## Folders
Here is the basic introduction, details are in latter parts. \
ChatGLM3: cloned from https://github.com/THUDM/ChatGLM3 \
demo: invoice demo using rag \
langchain: basic langchain example code \
langchain-multi-agent: multi local agent using langchain framework \
models: some downloaded guff models for local use \
policy: contain sample text policy for test \
self-cognition: self cognition data for finetune \
swift: python package 
## Files
1.png, 2.png: image used in this markdown \
agent_search&math.py: construct tool(search and math) agent by prompt using local model \
chunking.py: different chunking strategies \
llmFinetune.py: modelscope finetune example \
loadFinetunedModel.py: modelscope load model example \
loadModel.py: modelscope load model example \
ocr_test.py: unit test for ocr, ocr is used for invoice scan \
rag.py: rag example code using local model

# ModelScope Notes

## Environment Install

First we need to install anaconda by following the officical docs. \
Then we create python environment.
```
conda create -n modelscope python=3.8
conda activate modelscope
```
Finally we install the needed packages.
```
pip3 install torch torchvision torchaudio
pip install --upgrade tensorflow==2.13.0 # 仅支持 CPU 的版本
pip install modelscope
pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
## Basic operation

All these operation have corresponding python files. Could run directly to see what happens. 

### DownloadModel 

##### Python (default file path: ~/.cache/modelscope/hub) 
```
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen1.5-0.5B')
```
##### Git
```
git install lfs 
git clone https://www.modelscope.cn/qwen/Qwen1.5-0.5B.git
```

### LoadModel 
##### Load from online
```
model_type = ModelType.qwen1half_0_5b_chat
model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'}, **kwargs)
```
##### Load local model
```
model =  Model.from_pretrained("Qwen1.5-0.5B-Chat", device="mps")
tokenizer = AutoTokenizer.from_pretrained("Qwen1.5-0.5B-Chat", model_kwargs={'device_map': 'auto'}, **kwargs)
```

##### Note: The parameter "device" decides the device to run the model. Could choose from cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia

##### Note: If you need to load finetuned model from checkpoint:
```
ckpt_dir = 'output/qwen1half-7b-chat/vx-xxx/checkpoint-xxx'
model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
```

##### Note: Loading model by llama_cpp may need gguf model.
```
from llama_cpp import Llama
Llama(model_path="/app/mistral-7b-instruct-v0.2.Q3_K_L.gguf")
```
### ModelFinetune
```
model_type = ModelType.qwen1half_0_5b_chat
sft_args = SftArguments(
    model_type=model_type,
    dataset=['./self-cognition/self_cognition.jsonl#490'],
    #dataset=[f'{DatasetName.alpaca_zh}#500', f'{DatasetName.alpaca_en}#500', "./self-cognition/self_cognition.jsonl.jsonl#500"],
    #dataset=[f'{DatasetName.alpaca_zh}#500', f'{DatasetName.alpaca_en}#500', f'{DatasetName.self_cognition}#500'],
    logging_steps=5,
    max_length=2048,
    learning_rate=1e-4,
    output_dir='output',
    lora_target_modules=['ALL'],
    model_name=['小李', 'Xiao Huang'],
    model_author=['魔搭', 'ModelScope'],
    dtype="AUTO",
)
result = sft_main(sft_args)
best_model_checkpoint = result['best_model_checkpoint']
```
##### You could revise the trainning steps by setting Max_steps=xx.
##### You could revise the dataset in the "dataset" parameter. Use the path for local files or DatasetName.xxx to download online datasets.
##### Note: The "#500" behind the dataset means sampling frequency. Increase the number if the model's trainning effect is not obvious enough.

Below is an example of modelfinetune using qwen1.5-0.5B-chat with dataset of self-cognition.

Example of dataset:
```
{"query": "你是？", "response": "我是小李，由魔搭训练的人工智能助手。我的目标是为用户提供有用、准确和及时的信息，并通过各种方式帮助用户进行有效的沟通。请告诉我有什么可以帮助您的呢？", "tag": "zh"}
{"query": "你是谁!", "response": "您好！我是魔搭开发的人工智能语言模型，名为小李。我可以回答您的问题、提供信息、进行对话并帮助解决问题。如果您有任何疑问或需要帮助，请随时告诉我！", "tag": "zh"}
```
Before finetune:
```
query: 你是谁？
response: 我是来自阿里云的超大规模语言模型，我叫通义千问。
```
After finetuned:
```
query: 你是qwen吗？
response: 不，我不是Qwen。我是由魔搭开发的语言模型小李。有什么我可以帮助你的吗？
```
## RAG (Retrieval-augmented Generation)

reference: https://towardsdatascience.com/local-rag-from-scratch-3afc6d3dea08

Below is the framework of the rag sample project.
![1.png](./1.png)

### Chunking
We chunk the datasets for future retrieval. The general steps are spliting text into paragraph, paragraph to words, then chunk the words. Refining and applying overlap are used to optimize the chunks. 
```   
docs = document_chunker(directory_path='./ragData',
                            model_name=model_name,
                            chunk_size=256)
```
##### Note: Chunk size represents the number of tokens in a chunk.
##### Note: The directory_path in document_chunker function should be a directory. And all files in it will be chunked.

### Indexing

We could use any embedding model to vectorize our text in this part, converting the chunks in the document store to embeddings.   
```
def compute_embeddings(text):
    tokenizer = AutoTokenizer.from_pretrained("./model/tokenizer") 
    model = AutoModel.from_pretrained("./model/embedding")
```
You may store the embedding model and the tokenizer locally by running this code:
```
from transformers import AutoModel, AutoTokenizer
model_name = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer.save_pretrained("model/tokenizer")
model.save_pretrained("model/embedding")
```
#### Note: If you stored the vector_store or doc_store locally for future Q&A, the embedding model used for future query and the embedding model used to vectorize our text should be the same. Below is the error message when using qwen1.5_0.5B to generate vector_store and using BAAI/bge-small-en-v1.5 as embedding model for the query.
```
ValueError: shapes (1024,) and (384,) not aligned: 1024 (dim 0) != 384 (dim 0)
```

### Retrieval
By calculating cosine similarity, we could select top-k scores to match a query.
```
score = np.dot(chunk_embedding_array, query_str_embedding) / (norm_query * norm_chunk)
```

### Constructing Prompt
Now we have the retrieved text as well as the query string from user. We could construct prompt:
```
query = f
"""你是一个智能客服，现在用户正在向你询问一些问题。以下是一般客服的标准问答。
{retrieved_docs} 
请基于标准问答的内容回答用户问题
用户问题: {query_str} 
"""
```
Then the llm model could answer the query with the retrieved context. Below is an example using qwen1.5-0.5B-chat:
```
query: 你是一个智能客服，现在用户正在向你询问一些问题。以下是一般客服的标准问答。
    "你们大概多久能发货?": "您定心，在您付款之后咱们会第一时间为您安排发货，咱们许诺在48小时内会把产品宣布 ", 
    请基于标准问答的内容回答用户问题
    用户问题: 我购买的东西大约多久能发货? 
    
response: 您的问题中没有明确提及产品的具体发货时间，所以无法确定。但是，一般来说，一般情况下，当商品确认订单并成功支付后，我们会立即安排发货，并且预计在48小时内将产品发布。建议您可以查看订单信息或直接联系卖家以获取更准确的信息。
```
## RAG demo
Example code is in folder "demo" \
invoice_example: contains some images of invoices \
Qwen1.5-0.5B-Chat: model, embedding model, tokenizer \
ragData: knowledge data base \
ragJson: saved json files for rag search \
templates: html file templates \
app.py, chatbot.py: web app for demo use \
demo.py: command line interface code example
### Run front-end and back-end
```
python app.py
```

### Design detials
```
@app.route('/')
def home():
    return render_template('index.html')
```
Website is rendered by "index.html" which is in templates folder.
```
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    step = request.form.get('step')
    additional_input = request.form.get('additional_input')
    if step is not None and step != "":
        additional_input = user_input
    print("user add input")
    print(user_input)
    print(additional_input)
    response, next_step = chatbot.chat(user_input, step, additional_input)

    return jsonify({'response': response, 'next_step': next_step})
```
When clicking bottom on the website, a post request will be sent to the back end. Here "step" is used to determine the excat path for invoice location or folder location.
# ChatGLM3
reference: https://zhipu-ai.feishu.cn/wiki/X5shwBPOBiDWyNkwZ6xcd33lnRe

Example code is in folder "ChatGLM3/composite_demo"

## Bacis information
You could change all titles and names in file main.py. \
Below are some examples. \
Web page title:
```
st.set_page_config(
    page_title="国家电网",
    page_icon=":robot:",
    layout='centered',
    initial_sidebar_state='expanded',
)
```
Default system prompt:
```
DEFAULT_SYSTEM_PROMPT = '''
我是一名人工智能助手，由我国国家电网公司打造，主要针对的用户是广大电力使用的客户，提供电力相关的问题解答、服务预约等功能。请问有什么可以帮助您的吗？
'''.strip()
```
## Tools
ChatGLM3-6B only support calling one tool in one action.
You can add tools in tool_registry.py
```
@register_tool
def python_eval(
    exp: Annotated[str, "exp to calculate", True]
) -> float:
    """
    Evaluate/Calculate an expression.
    """
    if not isinstance(exp, str):
        raise TypeError("exp must be a str")
    return eval(exp)
```
## Code Interpreter
Use jupyter_client to execute the python code:
```
self.kernel_manager = jupyter_client.KernelManager(kernel_name=IPYKERNEL,
connection_file=self.kernel_config_path,exec_files=[self.init_file_path],env=env)
self.kernel = self.kernel_manager.blocking_client()
self.kernel.start_channels()
self.kernel.execute(code)
```
# LangChain
## Basic use
Example code in langchain folder. \
chain_example: langchain basic chain example \
chatbot_history_management: manage chat history example \
agent_example: langchain agent 
### Api keys
LangChain examples use api key for llm model, an api key is needed. \
If other tools(like online search) is used, other api key is needed. 
```
os.environ["GROQ_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""
```
### Local model
Using local model in langchain:
```
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
```
### Local embedding model
LangChain offical guide use openai embedding model(api key needed) \
Qwen1.5 is not able to use in langchain embedding but we could use gpt4all:
```
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
```
## Chunking Functions
LangChain has different chunking strategies which could improve rag search result.
### Simple split by character
```
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
```
```
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
```
### Split by token
```
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=0
)
```
### Split by semantic
```
text_splitter = SemanticChunker(MyEmbeddingFunction(), breakpoint_threshold_type="standard_deviation", sentence_split_regex= "\n")
```

## Multi-Agent
test.py: langchain single agent example \
app.py: langchain multi-agent example
### Create an agent
Here you could choose the local model:
```
llm = LlamaCpp(
    model_path="/Users/michaelwu/langchain/models/zephyr-7b-beta.Q5_0.gguf",
    #model_path="/Users/michaelwu/langchain/models/capybarahermes-2.5-mistral-7b.Q2_K.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
)
```
Also change each agent's tools:
```
@tool
def basic_calculator(query):
    """Basic calculator tool"""
    try:
        result = eval(query)
        return f"The result is {result}"
    except (SyntaxError, NameError) as e:
        return f"Sorry, I couldn't calculate that due to an error: {e}"

@tool
def equation_solver(query):
    """Equation solver tool"""
    # Basic equation solver (placeholder)
    # Implement specific logic for solving equations
    return "Equation solver: This feature is under development."

tools = [basic_calculator, equation_solver]
```
### Structure
Using one chat model to chat with user and this model would decide to call math agent or search agent by itself and respond to user:
```
template = "You are a helpful assistant. Classify the user input as either 'math' if it's math-related or 'general/technical/search' otherwise. respond directly with the classification.\nQuestion: {question}\n"
```

### Note: Api key is also needed for search agent
```
os.environ["TAVILY_API_KEY"] = ""
```
# Others
## OCR
file ocr_test.py tests the basic ocr function for sacnning an invoice
easyocr sample code:
```
img = img.crop(seller_name_crop_range)
img.save("tmp.png")
import easyocr
reader = easyocr.Reader(['ch_sim'], gpu=False)
text = reader.readtext("tmp.png")
print(text)
```
