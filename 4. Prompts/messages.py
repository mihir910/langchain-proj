from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# model = ChatOpenAI()
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# me - hi
# ai - hello, how can I help you
# me - I have a question about my order
# ai - sure, what is your order number?
# me - 12345
# ai - thank you, I see that your order has been shipped and should arrive tomorrow.
# convert the below to a chat history using SystemMessage, HumanMessage, and AIMessage 
# me - hi
# ai - hello, how can I help you
# me - I have a question about my order
# ai - sure, what is your order number?
# me - 12345
# ai - thank you, I see that your order has been shipped and should arrive tomorrow.


# HumanMessage(content="I want to request a refund for my order #12345.")
# AIMessage(content="Your refund request for order #12345 has been initiated. It will be processed in 3-5 business days.")

messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)

