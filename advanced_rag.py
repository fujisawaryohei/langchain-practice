from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import GitLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".md")

documents = GitLoader(
    clone_url="https://github.com/k88hudson/git-flight-rules",
    repo_path="./repos/git-flight-rules",
    branch="master",
    file_filter=file_filter
).load()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(documents, embeddings)

model = ChatAnthropic(model="claude-haiku-4-5",temperature=0)
prompt = ChatPromptTemplate.from_template("""
以下の文脈だけを踏まえて質問に回答してください。
文脈: \"\"\"
{context}
\"\"\"
質問: {question}
""")
retriever = db.as_retriever(search_kwargs={"k": 3})

chain = {
    "question": RunnablePassthrough(),
    "context": retriever
} | prompt | model | StrOutputParser()

content = chain.invoke("GitFlightRuleの概要と概要はどこから学習したか教えて。")
print(content)

### WebLoader
# from dotenv import load_dotenv
# from langchain_anthropic import ChatAnthropic
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# load_dotenv()

# documents = WebBaseLoader("https://www.wantedly.com/companies/fusic/post_articles/538735").load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# docs = text_splitter.split_documents(documents)

# embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
# db = Chroma.from_documents(docs, embeddings)

# model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
# prompt = ChatPromptTemplate.from_template("""
# 以下の文脈だけを踏まえて質問に回答してください。
# 文脈: \"\"\"
# {context}
# \"\"\"
# 質問: {question}
# """)
# retriever = db.as_retriever(search_kwargs={"k": 5})

# chain = {
#     "question": RunnablePassthrough(),
#     "context": retriever
# } | prompt | model | StrOutputParser()

# content = chain.invoke("プログラミングを始めたきっかけを教えて。")
# print(content)
