from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
llm = LlamaCpp(
    model_path="/Users/michaelwu/langchain/models/capybarahermes-2.5-mistral-7b.Q2_K.gguf",
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)
llm2 = LlamaCpp(
    model_path="/Users/michaelwu/langchain/models/qwen1_5-0_5b-chat-q8_0.gguf",
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)
result1 = llm.invoke("integral of f(x) = x")
result2 = llm2.invoke("integral of f(x) = x")
print("########")
print(result1)
print("########")
print(result2)
print("########")
