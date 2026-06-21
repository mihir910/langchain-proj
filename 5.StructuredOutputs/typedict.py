
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# result = model.invoke("who is the pm of india?")

class struct(TypedDict):
	departments: Annotated[list[str], "List of departments in the government only 10"]
	workers_under_department: Annotated[list[str], "List of workers under each department only 10"]
	government_name: Annotated[str, "Name of the government"]
	# asdf: Annotated[Optional[str], "This is a test field"]

structured_model = model.with_structured_output(struct)

result = structured_model.invoke(""" give me the list of departments in the government of india and the workers under each department and the name of the government""")

print(result)

# # schema
# class Review(TypedDict):

#     key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
#     summary: Annotated[str, "A brief summary of the review"]
#     sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
#     pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
#     cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
#     name: Annotated[Optional[str], "Write the name of the reviewer"]
    



# result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

# The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

# However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

# Pros:
# Insanely powerful processor (great for gaming and productivity)
# Stunning 200MP camera with incredible zoom capabilities
# Long battery life with fast charging
# S-Pen support is unique and useful
                                 
# Review by Nitish Singh
# """)

# print(result['name'])