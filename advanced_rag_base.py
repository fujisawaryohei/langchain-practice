from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import evaluate

from ragas_evaluation_pipeline import evaluators

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

# predict() が chain の出力から context・answer を取り出して LangSmith に渡すため、
# StrOutputParser() の結果（str）だけを返すのではなく、
# .assign() で answer キーを追加した dict を返すように変更
#
# 変更前（str を返す）:
# chain = (
#     {"question": RunnablePassthrough(), "context": retriever}
#     | prompt
#     | model
#     | StrOutputParser()
# )
chain = RunnableParallel(
    {
        "question": RunnablePassthrough(),
        "context": retriever,
    }
).assign(answer=prompt | model | StrOutputParser())


def predict(inputs: dict[str, Any]) -> dict[str, Any]:
    question = inputs["question"]
    output = chain.invoke(question)
    return {
        "contexts": output["context"],
        "answer": output["answer"],
    }


results = evaluate(
    predict,
    data="agent-book",
    evaluators=evaluators,
)
print(results)
