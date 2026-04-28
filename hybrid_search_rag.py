from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import GitLoader
from dotenv import load_dotenv

load_dotenv()

def reciprocal_rank_fusion(
    retriever_outputs: list[list[Document]],
    k: int = 60,
)->list[str]:
    content_score_mapping = {}

    # 検索クエリごとにループ
    for docs in retriever_outputs:
        # 検索結果のドキュメントごとにループ
        for rank, doc in enumerate(docs):
            content = doc.page_content
            # 初めて登場したコンテンツの場合はスコアを0で初期化
            if content not in content_score_mapping:
                content_score_mapping[content] = 0
            
            # (1/(順位+k)のスコアを加算)
            content_score_mapping[content] += 1 / (rank + k)
    
    # スコアの大きい順にソート
    ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)
    return [content for content, _ in ranked]

def file_filter(file_path: str):
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
    collection_name = "langchain_docs",
)

prompt = ChatPromptTemplate.from_template("""
以下のコンテキストを参考に質問に答えてください。
̦
コンテキスト:{context}

質問:{question}
""")

model = ChatAnthropic(model="claude-haiku-4-5",temperature=0)

chroma_retriever = db.as_retriever().with_config(
    { "run_name": "chroma_retriever" }
)

bm25_retriever = BM25Retriever.from_documents(documents).with_config(
    { "run_name": "bm25_retriever" }
)

hybrid_retriever = (
    RunnableParallel({
        "chroma_documents": chroma_retriever,
        "bm25_documents": bm25_retriever        
    })
    | (lambda x: [x["chroma_documents"], x["bm25_documents"]])
    | reciprocal_rank_fusion
)

chain = {
    "question": RunnablePassthrough(),
    "context": hybrid_retriever,
} | prompt | model | StrOutputParser()

result = chain.invoke("Langchainの概要を_教えて。")
print(result)