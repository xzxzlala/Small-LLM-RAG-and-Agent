from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://127.0.0.1:8000/chain/")
print(remote_chain.invoke({"language": "italian", "text": "hi"}))