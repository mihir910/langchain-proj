from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
	repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
	task="text-generation")

model = ChatHuggingFace(llm=llm)

documents = [
	"Attention Is All You Need is a seminal paper that introduced the Transformer architecture, which has become the foundation for many state-of-the-art models in natural language processing. The paper presents a novel approach to sequence transduction tasks, such as machine translation, by relying entirely on self-attention mechanisms, eliminating the need for recurrent or convolutional layers.",
	"BERT: Pre-training of Deep Bidirectional Transformers is a groundbreaking paper that introduced the Bidirectional Encoder Representations from Transformers (BERT) model. BERT is pre-trained on a large corpus of text and can be fine-tuned for various NLP tasks, achieving state-of-the-art performance. The key innovation of BERT is its ability to understand the context of a word based on all of its surroundings (left and right context), making it highly effective for tasks like question answering and sentiment analysis.",
	"GPT-3: Language Models are Few-Shot Learners is a paper that presents",
	"Diffusion Models Beat GANs on Image Synthesis is a paper that explores the use of diffusion models for image synthesis tasks. The authors demonstrate that diffusion models can outperform Generative Adversarial Networks (GANs) in terms of image quality and diversity. The paper provides insights into the advantages of diffusion models, such as their ability to model complex data distributions and generate high-fidelity images, making them a promising alternative to GANs for various image generation applications."
]

user_query = "Explain the paper 'Attention Is All You Need' in a beginner-friendly manner with a short explanation."

embedding = HuggingFaceEndpointEmbeddings(
	model="sentence-transformers/all-MiniLM-L6-v2")

embedding_docs = embedding.embed_documents(documents);
user_query_embedding = embedding.embed_query(user_query);

similarity_scores = cosine_similarity([user_query_embedding], embedding_docs)[0]

# print("Similarity Scores:", similarity_scores)
closest_document_index = similarity_scores.argmax()
# print("Closest Document Index:", closest_document_index)
# print("Closest Document:", documents[closest_document_index])


prompt = ChatPromptTemplate([
	('system','You are a helpful assistant'),
	('human', 'Based on the document provided {document_provided}'),
	# ('human','Based on the following document, provide a beginner-friendly explanation of the paper in a short format: {context}')
	('human','{user_query}'),
])

final_prompt = prompt.invoke({'document_provided':documents[closest_document_index], 'user_query':user_query})

# print(final_prompt)

model_response = model.invoke(final_prompt)
print(model_response.content)