# from langchain_huggingface import HuggingFaceEmbeddings

# embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# documents = [
#     "Delhi is the capital of India",
#     "Kolkata is the capital of West Bengal",
#     "Paris is the capital of France"
# ]

# vector = embedding.embed_documents(documents)

# print(str(vector))


from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
)

vector = embeddings.embed_documents(["What is AI?", "What is machine learning?"])

print(str(vector))