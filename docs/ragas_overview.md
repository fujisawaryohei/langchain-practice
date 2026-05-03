# Ragas 概要

## Ragasとは

RAGの評価に必要なものをエンドツーエンドで提供する評価フレームワーク。
単なるメトリクス定義にとどまらず、評価を実行するための実行基盤ごと提供している。

## 提供物

| 提供物 | 内容 |
|---|---|
| 合成テストデータ生成 | ナレッジグラフを構築し、質問・回答ペアを自動生成。LangSmithなどへの保存も可能 |
| 評価メトリクス定義 | context_precision, answer_relevancy など |
| 評価実行基盤 | LLM/EmbeddingsのWrapper、非同期実行、リトライ処理 |
| 外部連携 | LangSmithなどへの結果保存 |

## 主なメトリクス

### context_precision
取得したコンテキストが正解に関連しているかをLLMに判定させてスコアを算出。
検索（Retriever）の精度改善につながる。LLMのみ必要。

### answer_relevancy
回答が質問に対して的確かをベクトルの類似度で計算。
① LLMで「回答から逆算した質問」を生成 → ② Embeddingsでベクトル化 → ③ コサイン類似度でスコア算出。
LLMとEmbeddingsの両方が必要。

## スコアリングの主体

メトリクスによって異なる。

- LLMだけ → テキストの意味をLLMが直接判断できる場合（context_precision など）
- LLM + Embeddings → ベクトル類似度の計算が必要な場合（answer_relevancy など）
- Embeddingsだけ → LLMを使わず計算するメトリクスも存在する

## 合成テストデータのクエリ種別

テストデータ生成時に `query_distribution` で種別と生成比率を指定する。

| クエリ種別 | 内容 | 特徴 |
|---|---|---|
| `SingleHopSpecificQuerySynthesizer` | 1つのドキュメントだけで答えられる具体的な質問 | 最もシンプル |
| `MultiHopAbstractQuerySynthesizer` | 複数ドキュメントをまたいで推論が必要な抽象的な質問 | 概念的・抽象的な問い |
| `MultiHopSpecificQuerySynthesizer` | 複数ドキュメントをまたいだ具体的な質問 | 複数情報源が必要 |

MultiHop系のクエリはナレッジグラフのエッジ（ドキュメント間の関係）を使って生成される。
エッジが少ないとMultiHop系クエリが生成しにくくなるため、transformsでしっかりエッジを張ることが重要。

```python
query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),   # 50%
    (MultiHopAbstractQuerySynthesizer(llm=llm), 0.25),   # 25%
    (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25),   # 25%
]
```

## 評価フロー

```
Ragasで合成テストデータを生成
    ↓
LangSmithのデータセットに保存
    ↓
predict()でRAGを実行（1件ずつ）
    ↓
SingleTurnSampleに詰め替え（question・answer・contexts・ground_truth）
    ↓
single_turn_score()でスコア算出
    ↓
LangSmithに結果を記録・集計
```
