import pandas as pd
from langsmith import Client

# This langsmith dataset is private. Instead, load from the link above
# client = Client()
# def get_dataset(train_dataset_name):
#   examples = client.list_examples(dataset_name=train_dataset_name)
#   train_df = pd.DataFrame([{**e.inputs, **e.outputs} for e in examples])
#   return train_df

# # Load dataset from LangSmith
# dataset_name_tmpl = "Carb-IE-{split}"
# train_dataset_name = dataset_name_tmpl.format(split="train")
# train_df = get_dataset(train_dataset_name)
# df=train_df[['sentence','clusters']]

import requests
import pandas as pd
from io import StringIO

def download_from_drive(link: str) -> str:
    """
    Download a file from Google Drive using a shareable link.

    Parameters:
    - link (str): Shareable link of the Google Drive file.

    Returns:
    - str: Content of the downloaded file as a string.
    """
    # Extract the file ID from the shareable link
    file_id = link.split("/")[5]

    # Create the direct download link
    direct_link = f"https://drive.google.com/uc?id={file_id}"

    # Get the file content
    response = requests.get(direct_link)
    response.raise_for_status()  # Raise an error if the request failed

    return response.text

def load_drive_file_to_df(link: str) -> pd.DataFrame:
    """
    Load a Google Drive file (in .jsonl format) into a Pandas DataFrame.

    Parameters:
    - link (str): Shareable link of the Google Drive file.

    Returns:
    - pd.DataFrame: DataFrame containing the data from the Google Drive file.
    """
    file_content = download_from_drive(link)

    # Use StringIO to convert the string content to a file-like object so it can be read into a DataFrame
    return pd.read_json(StringIO(file_content), lines=True)

train_dataset = "https://drive.google.com/file/d/14CcxhDKb5k4hPKOITkBIiSXutYzHk2aY/view?usp=sharing"
df = load_drive_file_to_df(train_dataset)
# Prepare for fine-tuning
df.columns=["prompt","response"]
# print(df.head(3))
# Save our DataFrame to a format that can be read by HuggingFace
from datasets import load_dataset

# Write to JSON
df.to_json('train.jsonl', orient='records', lines=True)
# train_df_synthetic.to_json('train_synthetic.jsonl', orient='records', lines=True)
# Create instructions
import json
def create_instructions(examples):
    texts = []

    for prompt, response in zip(examples['prompt'], examples['response']):
        # Convert dictionary response to string
        if isinstance(response, list):
            # Pretty print for better readability
            response_str = json.dumps(response, indent=2)
        else:
            response_str = response

        # Format the text using the instruction structure provided
        text = (f'<s>[INST] <<SYS>>\n'
                f'{system_prompt.strip()}\n'
                f'<</SYS>>\n\n'
                f'### Input: \n{prompt}\n\n'
                f'### Output: \n{response_str}\n'
                f'[/INST]</s>')

        texts.append(text)

    return {'text': texts}

# Set system prompt for our particular task
system_prompt = ("you are a model tasked with extracting knowledge graph triples from given text. "
              "The triples consist of:\n"
              "- \"s\": the subject, which is the main entity the statement is about.\n"
              "- \"object\": the object, which is the entity or concept that the subject is related to.\n"
              "- \"relation\": the relationship between the subject and the object. "
              "Given an input sentence, output the triples that represent the knowledge contained in the sentence.")

# Read JSON we saved
train_dataset = load_dataset('json', data_files='/Users/michaelwu/langchain/train.jsonl', split="train")
# train_dataset_synthetic = load_dataset('json', data_files='/content/train_synthetic.jsonl', split="train")

# Create instructions, which we can see in "text" field below
train_dataset_mapped = train_dataset.map(create_instructions, batched=True)
print(train_dataset_mapped[0])

import torch
from transformers import AutoModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from modelscope.models import Model
from transformers import AutoTokenizer, AutoModel
model_name = "Qwen1.5-0.5B-Chat"
model =  Model.from_pretrained(model_name)
# Chat model
model.config.use_cache = False
model.config.pretraining_tp = 1
print("1")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
pipe_llama7b_chat = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=300) # set device to run inference on GPU