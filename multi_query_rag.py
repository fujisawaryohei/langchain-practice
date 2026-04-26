from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-haiku-4-5",temperature=0)

class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クエリのリスト")

query_generation_prompt = ChatPromptTemplate.from_template("""
    質問に対してベクターデータベースから関連文書を検索するために、
    3つの異なる検索クエリを生成してください。
    距離ベースの類似度検索の限界を克服するために、
    ユーザーの質問に対して複数の視点を提供する事が目標です。

    質問: {question}
""")

prompt = ChatPromptTemplate.from_template("""
以下のコンテキストを参考に質問に答えてください。

コンテキスト:{context}

質問:{question}
""")

query_generation_chain = (
    query_generation_prompt 
    | model.with_structured_output(QueryGenerationOutput)
    | (lambda x: x.queries)
)

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

retriever = db.as_retriever()

chain = {
    "question": RunnablePassthrough(),
    "context": query_generation_chain | retriever.map(),
} | prompt | model | StrOutputParser()

result = chain.invoke("LangChainの概要を教えて。")
print(result)
