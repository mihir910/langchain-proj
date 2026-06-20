from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-5.4-nano-2026-03-17')

result = llm.invoke("What is the capital of India")

print(result)
