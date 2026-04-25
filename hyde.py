from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".md")

loader = GitLoader(
    repo_path="./repos/langchain",
    branch="master",
    file_filter=file_filter
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

hypothentical_prompt = ChatPromptTemplate.from_template("""
次の質問に回答する一文を書いてください。

質問: {question}
""")

prompt = ChatPromptTemplate.from_template("""
以下のコンテキストを参考に質問に答えてください。

コンテキスト:{context}

質問:{question}
""")

model = ChatAnthropic(model="claude-haiku-4-5",temperature=0)

hypothentical_chain = hypothentical_prompt | model | StrOutputParser()

retriever = db.as_retriever()

chain = {
    "question": RunnablePassthrough(),
    "context": hypothentical_chain | retriever,
} | prompt | model | StrOutputParser()

result = chain.invoke("LangChainの概要を教えて。")
print(result)
