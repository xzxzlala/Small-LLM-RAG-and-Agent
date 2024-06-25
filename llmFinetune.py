# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments,
    infer_main, sft_main, app_ui_main
)
#dataset=[f'{DatasetName.self_cognition}#500'],
if __name__ == '__main__':
    model_type = ModelType.qwen1half_0_5b_chat
    sft_args = SftArguments(
        model_type=model_type,
        dataset=['./self-cognition/self_cognition.jsonl'],
        #dataset=[f'{DatasetName.alpaca_zh}#500', f'{DatasetName.alpaca_en}#500', "./self-cognition/self_cognition.jsonl.jsonl#500"],
        #dataset=[f'{DatasetName.alpaca_zh}#500', f'{DatasetName.alpaca_en}#500', f'{DatasetName.self_cognition}#500'],
        logging_steps=5,
        max_length=2048,
        learning_rate=1e-4,
        max_steps=100,
        output_dir='output',
        lora_target_modules=['ALL'],
        model_name=['小李', 'Xiao Huang'],
        model_author=['魔搭', 'ModelScope'],
        dtype="AUTO",
      )
    result = sft_main(sft_args)
    best_model_checkpoint = result['best_model_checkpoint']
    print(f'best_model_checkpoint: {best_model_checkpoint}')
    torch.cuda.empty_cache()

    infer_args = InferArguments(
        model_type=model_type,
        ckpt_dir=best_model_checkpoint,
        load_dataset_config=True)
    # merge_lora(infer_args, device_map='cpu')
    result = infer_main(infer_args)
    torch.cuda.empty_cache()

    app_ui_main(infer_args)
