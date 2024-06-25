
# Experimental environment: 3090
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything
from swift.tuners import Swift
from modelscope.models import Model
from modelscope import AutoTokenizer

seed_everything(42)

ckpt_dir = './output/qwen1half-0_5b-chat/v64-20240618-155921/checkpoint-93'
model_type = ModelType.qwen1half_0_5b_chat
template_type = get_default_template_type(model_type)

model =  Model.from_pretrained("Qwen1.5-0.5B-Chat", device="mps")
tokenizer = AutoTokenizer.from_pretrained("Qwen1.5-0.5B-Chat", model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 128

model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
template = get_template(template_type, tokenizer)

query = '你是qwen吗？'
response, history = inference(model, template, query)
print(f'response: {response}')
print(f'history: {history}')