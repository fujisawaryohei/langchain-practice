import nest_asyncio
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import GitLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import Client
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import NodeType
from ragas.testset.synthesizers import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.transforms import (
    CosineSimilarityBuilder,
    EmbeddingExtractor,
    Parallel,
)
from ragas.testset.transforms.extractors.llm_based import (
    NERExtractor,
    SummaryExtractor,
    ThemesExtractor,
)
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.testset.transforms.relationship_builders.traditional import OverlapScoreBuilder

load_dotenv()

nest_asyncio.apply()

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".md")

loader = GitLoader(
    repo_path="./repos/langchain",
    branch="master",
    file_filter=file_filter,
)
documents = loader.load()

for document in documents:
    document.metadata["filename"] = document.metadata["source"]

llm = LangchainLLMWrapper(ChatAnthropic(model="claude-haiku-4-5"))
embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings())
generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

def filter_docs(node):
    return node.type == NodeType.DOCUMENT

def filter_doc_with_num_tokens(node, min_num_tokens=100):
    from ragas.testset.transforms.default import num_tokens_from_string

    return (
        node.type == NodeType.DOCUMENT
        and num_tokens_from_string(node.properties["page_content"]) > min_num_tokens
    )

# HeadlinesExtractor と HeadlineSplitter を除いたカスタム transforms
# （詳細は docs/ragas_custom_transforms.md 参照）
transforms = [
    SummaryExtractor(llm=llm, filter_nodes=lambda node: filter_doc_with_num_tokens(node)),
    CustomNodeFilter(llm=llm),
    Parallel(
        EmbeddingExtractor(
            embedding_model=embeddings,
            property_name="summary_embedding",
            embed_property_name="summary",
            filter_nodes=lambda node: filter_doc_with_num_tokens(node),
        ),
        ThemesExtractor(llm=llm, filter_nodes=lambda node: filter_docs(node)),
        NERExtractor(llm=llm),
    ),
    Parallel(
        CosineSimilarityBuilder(
            property_name="summary_embedding",
            new_property_name="summary_similarity",
            threshold=0.5,
            filter_nodes=lambda node: filter_doc_with_num_tokens(node),
        ),
        OverlapScoreBuilder(threshold=0.01),
    ),
]

query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
    (MultiHopAbstractQuerySynthesizer(llm=llm), 0.25),
    (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25),
]

testset = generator.generate_with_langchain_docs(
    documents=documents,
    testset_size=4,
    transforms=transforms,
    query_distribution=query_distribution,
    raise_exceptions=False,
)

testset.to_pandas()

### LangSmith へ Ragasで生成した合成テストデータの保存
dataset_name = "agent-book"
client = Client()

if client.has_dataset(dataset_name=dataset_name):
    client.delete_dataset(dataset_name=dataset_name)

dataset = client.create_dataset(dataset_name=dataset_name)

inputs = []
outputs = []
metadatas = []

for record in testset.to_list():
    inputs.append(
        {
            "question": record.get("user_input"),
        }
    )
    outputs.append(
        {
            "contexts": record.get("reference_contexts"),
            "ground_truth": record.get("reference"),
        }
    )
    metadatas.append(
        {
            "evolution_type": record.get("synthesizer_name"),
        }
    )

client.create_examples(
    inputs=inputs,
    outputs=outputs,
    metadata=metadatas,
    dataset_id=dataset.id
)