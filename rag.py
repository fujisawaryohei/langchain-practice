from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".md")

loader = GitLoader(
    clone_url="https://github.com/k88hudson/git-flight-rules",
    repo_path="./git-flight-rules",
    branch="master",
    file_filter=file_filter
)

raw_docs = loader.load()

# チャンク分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(raw_docs)

# チャンク分割したデータをベクトル化し、Vector Storeへ保存
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(docs, embedding=embeddings)

query = "git-flight-rulesってなに？"
retriever = db.as_retriever()
context_doc = retriever.invoke(query)

print(context_doc[0].page_content)
