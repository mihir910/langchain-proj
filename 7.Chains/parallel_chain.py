from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)

# llm2 = MistralEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.1",
# )

model2 = ChatMistralAI(model="mistral-small")

prompt1 = PromptTemplate(
    template='give me 5 worst fights of \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='give me 5 best fights of {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided fights into a single document \n worst -> {worst} and best -> {best}',
    input_variables=['worst', 'best']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'worst': prompt1 | model1 | parser,
    'best': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """ Naruto
"""

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()

