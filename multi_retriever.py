from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import TavilySearchAPIRetriever

from enum import Enum
from typing import Any
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Route(str, Enum):
    langchain_document = "langchain_document"
    web = "web"


class RouteOutput(BaseModel):
    route: Route


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".md")


def routed_retriever(inp: dict[str, Any]) -> list[Document]:
    question = inp["question"]
    route = inp["route"]

    if route == Route.langchain_document:
        return langchain_document_retriever.invoke(question)
    elif route == Route.web:
        return web_retriever.invoke(question)

    raise ValueError(f"Unknown retriever: {retriever}")


model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

route_prompt = ChatPromptTemplate.from_template("""
質問に回答するために適切なRetrieverを選択してください。

質問: {question}
""")

prompt = ChatPromptTemplate.from_template("""
以下のコンテキストを参考に質問に答えてください。

コンテキスト:{context}

質問:{question}
""")

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

retriever = db.as_retriever()

langchain_document_retriever = retriever.with_config(
    {"run_name": "langchain_document_retriever"}
)

web_retriever = TavilySearchAPIRetriever(k=3).with_config({"run_name": "web_retriever"})

route_chain = (
    route_prompt | model.with_structured_output(RouteOutput) | (lambda x: x.route)
)

chain = (
    {"question": RunnablePassthrough(), "route": route_chain}
    | RunnablePassthrough.assign(context=routed_retriever)
    | prompt
    | model
    | StrOutputParser()
)

result = chain.invoke("福岡の明日の天気は？")
print(result)
