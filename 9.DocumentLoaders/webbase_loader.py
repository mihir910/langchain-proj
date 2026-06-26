from langchain_community.document_loaders import WebBaseLoader
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

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://en.wikipedia.org/wiki/One_Piece_(1999_TV_series)'
loader = WebBaseLoader(url)

docs = loader.load()

# print(docs[0].page_content[:100])
# print(len(docs))
# print(type(docs))
chain = prompt | model | parser

print(chain.invoke({'question':'how many members are there in straw hat crew?', 'text':docs[0].page_content}))