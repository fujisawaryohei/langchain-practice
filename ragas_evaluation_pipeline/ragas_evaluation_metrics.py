from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision

from ragas_evaluation_pipeline.ragas_metric_evaluator import RagasMetricEvaluator

llm = LangchainLLMWrapper(ChatAnthropic(model="claude-haiku-4-5", temperature=0))
embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
))

# ragas.metrics の旧メトリクスは属性でLLM/Embeddingsをセットする
context_precision.llm = llm
answer_relevancy.llm = llm
answer_relevancy.embeddings = embeddings

metrics = [context_precision, answer_relevancy]

evaluators = [RagasMetricEvaluator(metric).evaluate for metric in metrics]
