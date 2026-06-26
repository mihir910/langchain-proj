from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    path='9.DocumentLoaders/Naruto',
    glob='**',
)

docs = loader.lazy_load()
# print(docs);
for document in docs:
    print(document.page_content)