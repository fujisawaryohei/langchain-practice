from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import GitLoader
from dotenv import load_dotenv

load_dotenv()


def file_filter(file_path: str):
    return file_path.endswith(".md")


loader = GitLoader(
    repo_path="./repos/langchain", branch="master", file_filter=file_filter
)
documents = loader.load()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="langchain_docs",
)

prompt = ChatPromptTemplate.from_template("""
以下のコンテキストを参考に質問に答えてください。

コンテキスト:{context}

質問:{question}
""")

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

retriever = db.as_retriever()

chain = (
    {
        "question": RunnablePassthrough(),
        "context": retriever,
    }
    | prompt
    | model
    | StrOutputParser()
)

result = chain.invoke("Langchainの概要を教えて。")
print(result)
