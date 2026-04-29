import nest_asyncio
from langchain_community.document_loaders import GitLoader


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".md")


loader = GitLoader(
    repo_path="./repos/langchain", branch="master", file_filter=file_filter
)
documents = loader.load()

for document in documents:
    document.metadata["filename"] = document.metadata["source"]
