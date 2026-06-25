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
    template='give me a summary on \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='give me 10 quizes on {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided quizzes and summaries into a single document with showing headers and then the answers like Quiz:- ... , Summary:- ... \n quiz -> {quiz} and summary -> {summary}',
    input_variables=[ 'quiz', 'summary']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'summary': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """ Naruto
"""

result = chain.invoke({'text':text})

print(result)

# chain.get_graph().print_ascii()

