import getpass
import os
os.environ["GROQ_API_KEY"] = getpass.getpass()
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes

model = ChatGroq(model="llama3-8b-8192")
# messages = [
#     SystemMessage(content="Translate the following from English into Chinese"),
#     HumanMessage(content="hi!"),
# ]
parser = StrOutputParser()
# result = model.invoke(messages)
# print(parser.invoke(result))
# chain = model | parser
# print(chain.invoke(messages))
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
chain = prompt_template | model | parser
# print(chain.invoke({"language": "Chinese", "text": "hi"}))

# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

add_routes(
    app,
    chain,
    path="/chain",
)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)