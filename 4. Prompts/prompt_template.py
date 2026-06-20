from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# model = ChatOpenAI()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# detailed way
template2 = PromptTemplate(
    template='Greet this person in 5 languages. The name of the person is {name}',
    input_variables=['name']
)

# fill the values of the placeholders
prompt = template2.invoke({'name':'nitish'})

print(prompt)
# result = model.invoke(prompt)

# print(result.content)

