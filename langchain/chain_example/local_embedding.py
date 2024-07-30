from gpt4all import Embed4All
text = 'The quick brown fox jumps over the lazy dog'
embedder = Embed4All("/Users/michaelwu/langchain/models/all-MiniLM-L6-v2-f16.gguf")
output = embedder.embed(text)
print(output)