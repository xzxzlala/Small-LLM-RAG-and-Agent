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
query = '''
回答以下问题并尽可能遵守以下命令。

您可以使用以下工具：

search: 当您需要回答有关时事的问题时很有用。你应该提出具体的问题。
calculator: 当您需要回答数学问题时很有用。使用python代码, 例如:2 + 2
Respnse To Human:当您需要对正在与您交谈的人做出回答时。

您将收到来自用户的消息，然后您应该开始执行以下两件事之一

    选项 1: 您使用工具来回答问题。
    为此，您应该使用以下格式：
    想法：你应该思考该做什么
    Action: 要采取的操作，应该是[“search”,“calculator”]之一
    Action Input: 要发送到工具的输入

    选项 2: 你对用户做出回答。
    为此，您应该使用以下格式：
    Action: Resonse To Human
    Action Input: 你对于用户的回答,总结你所做的和学到的

选择一个选项，执行完毕后请等待结果再继续。

以下是用户的问题：

2+2*3等于几
'''
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')