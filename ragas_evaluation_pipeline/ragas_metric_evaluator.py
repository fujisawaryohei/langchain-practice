from typing import Any

from dotenv import load_dotenv
from langsmith.schemas import Example, Run
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import Metric

load_dotenv()


class RagasMetricEvaluator:
    def __init__(self, metric: Metric):
        # 0.4.x ではLLM・EmbeddingsはMetricのコンストラクタで渡す設計のため、
        # 設定済みのmetricインスタンスをそのまま受け取る
        self.metric = metric

    def evaluate(self, run: Run, example: Example) -> dict[str, Any]:
        context_strs = [doc.page_content for doc in run.outputs["contexts"]]

        # LangSmith の Example（データセット）と Run（推論結果）から
        # Ragas 評価用の1件分のサンプルを組み立てる
        # - user_input / reference はデータセット側（example）から取得
        # - response / retrieved_contexts は推論結果（run）から取得
        sample = SingleTurnSample(
            user_input=example.inputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=context_strs,
            reference=example.outputs["ground_truth"],
        )

        # Ragasの評価メトリクスのsingle_turn_scoreメソッドでスコアを算出
        score = self.metric.single_turn_score(sample)
        return {"key": self.metric.name, "score": score}
