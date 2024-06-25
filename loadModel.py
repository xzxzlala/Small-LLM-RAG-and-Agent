import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything
from modelscope.utils.hub import read_config
from modelscope.models import Model
from modelscope.preprocessors import Preprocessor, TokenClassificationTransformersPreprocessor
from modelscope import AutoModelForCausalLM, AutoTokenizer
model_type = ModelType.qwen1half_0_5b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen


kwargs = {}
#kwargs['use_flash_attn'] = True  # 使用flash_attn

# model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'}, **kwargs)

model =  Model.from_pretrained("Qwen1.5-0.5B-Chat", device="mps")
tokenizer = AutoTokenizer.from_pretrained("Qwen1.5-0.5B-Chat", model_kwargs={'device_map': 'auto'}, **kwargs)

# 修改max_new_tokens
model.generation_config.max_new_tokens = 128

template = get_template(template_type, tokenizer)
seed_everything(42)
query = '你是谁？'
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')