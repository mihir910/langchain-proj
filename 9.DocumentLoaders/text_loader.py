from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# prompt = PromptTemplate(
#     template='Write a summary for the following poem - \n {poem}',
#     input_variables=['poem']
# )

# parser = StrOutputParser()

loader = TextLoader('Naruto_Overview.txt', encoding='utf-8')

# docs = loader.load()
docs = loader.lazy_load()

for i in docs:
	print(i.page_content)
	break
# [{page_content: asdf, metadata},{},{}]
# print(docs)

# print(type(docs))

# print(len(docs))

# print(docs[0].page_content)

# print(docs[0].metadata)

# chain = prompt | model | parser

# print(chain.invoke({'poem':docs[0].page_content}))

