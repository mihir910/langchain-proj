from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
	repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
	task="text-generation")

model = ChatHuggingFace(llm=llm)

promptTemplate = PromptTemplate(
	template="""You are a helpful assistant. Answer the following question: {question1} {question2} {question3}""",
	input_variables=["question1", "question2", "question3"]
)
promptTemplate.save('Prompts/mirzapur.json')
prompt = promptTemplate.invoke({'question1': 'What is the capital of France?', 'question2': 'What is the largest city in France?', 'question3': 'What is the population of Paris?'})

# model_response = model.invoke(prompt)
# print(model_response.content)