# Ragas カスタム transforms の解説

## 背景：なぜカスタム transforms が必要だったか

Ragas の `generate_with_langchain_docs` は内部で `default_transforms` を自動的に適用します。
この処理はドキュメントの長さに応じて2つのパスに分岐します。

```
ドキュメントの25%以上が500トークン超
    → HeadlinesExtractor → HeadlineSplitter → ...（問題のあるパス）

ドキュメントの25%以上が101〜500トークン
    → SummaryExtractor → ...（HeadlineSplitter なし）
```

今回のエラーは「500トークン超パス」に入ったため `HeadlinesExtractor`（LLMで見出し抽出）
が走ったが、LLMの応答フォーマットが合わずに失敗。
`headlines` プロパティが未セットのまま `HeadlineSplitter` が動いてクラッシュした。

---

## カスタム transforms の全体像

```python
transforms = [
    SummaryExtractor(...),        # ステップ1: 要約抽出
    CustomNodeFilter(...),        # ステップ2: ノードフィルタリング
    Parallel(                     # ステップ3: 並列処理
        EmbeddingExtractor(...),
        ThemesExtractor(...),
        NERExtractor(...),
    ),
    Parallel(                     # ステップ4: 並列処理
        CosineSimilarityBuilder(...),
        OverlapScoreBuilder(...),
    ),
]
```

HeadlinesExtractor と HeadlineSplitter を除いた、
`default_transforms` の101〜500トークンパスと同等の処理を手書きで再現しています。

---

## 各コンポーネントの解説

### フィルター関数

```python
def filter_docs(node):
    return node.type == NodeType.DOCUMENT

def filter_chunks(node):
    return node.type == NodeType.CHUNK

def filter_doc_with_num_tokens(node, min_num_tokens=100):
    from ragas.testset.transforms.default import num_tokens_from_string
    return (
        node.type == NodeType.DOCUMENT
        and num_tokens_from_string(node.properties["page_content"]) > min_num_tokens
    )
```

Ragas はドキュメントをナレッジグラフのノードとして管理します。
ノードには2種類あります：

- `NodeType.DOCUMENT`：元のドキュメント全体
- `NodeType.CHUNK`：分割された断片

各 transform の `filter_nodes` に渡すことで、
「どのノードに対してこの処理を実行するか」を制御します。

---

### ステップ1: SummaryExtractor

```python
SummaryExtractor(
    llm=llm,
    filter_nodes=lambda node: filter_doc_with_num_tokens(node)
)
```

LLMを使って各ドキュメントの要約を生成し、
ノードの `summary` プロパティとして保存します。
100トークン以下のドキュメントは短すぎるためスキップします。

---

### ステップ2: CustomNodeFilter

```python
CustomNodeFilter(llm=llm)
```

LLMを使って、テスト生成に適さないノード（内容が薄い、関係ない等）を除外します。
このフィルタを通過したノードだけが以降の処理に使われます。

---

### ステップ3: Parallel（3つの抽出処理を並列実行）

#### EmbeddingExtractor

```python
EmbeddingExtractor(
    embedding_model=embeddings,
    property_name="summary_embedding",
    embed_property_name="summary",
    filter_nodes=lambda node: filter_doc_with_num_tokens(node),
)
```

ステップ1で生成した `summary` をベクトル化して
`summary_embedding` プロパティとして保存します。
後のステップで「意味的に似たドキュメント」を見つけるために使います。

#### ThemesExtractor

```python
ThemesExtractor(
    llm=llm,
    filter_nodes=lambda node: filter_docs(node)
)
```

LLMを使ってドキュメントのテーマ（主題）を抽出します。
MultiHop系のクエリ生成時に、関連ドキュメントを結びつける手がかりになります。

#### NERExtractor

```python
NERExtractor(llm=llm)
```

固有名詞（人名・組織名・概念名など）を抽出します。
後のステップで `OverlapScoreBuilder` がこの情報を使って
ドキュメント間の関連性を計算します。

---

### ステップ4: Parallel（2つの関係構築処理を並列実行）

#### CosineSimilarityBuilder

```python
CosineSimilarityBuilder(
    property_name="summary_embedding",
    new_property_name="summary_similarity",
    threshold=0.5,
    filter_nodes=lambda node: filter_doc_with_num_tokens(node),
)
```

`summary_embedding` 同士のコサイン類似度を計算し、
`threshold=0.5` 以上のペアをナレッジグラフのエッジ（関係）として登録します。
意味的に近いドキュメントが繋がることで、MultiHopクエリの生成に使われます。

#### OverlapScoreBuilder

```python
OverlapScoreBuilder(threshold=0.01)
```

`NERExtractor` で抽出した固有名詞のオーバーラップ率を計算し、
共通する固有名詞が多いドキュメント同士をエッジで繋ぎます。
`rapidfuzz` ライブラリを使ったあいまいマッチングで、
表記ゆれ（例: "LangChain" と "Langchain"）も同一視できます。

---

## transforms 全体の流れまとめ

```
documents
    ↓
[SummaryExtractor]     各ドキュメントの要約をLLMで生成
    ↓
[CustomNodeFilter]     不要なノードを除外
    ↓
[Parallel]
  ├─ [EmbeddingExtractor]  要約をベクトル化
  ├─ [ThemesExtractor]     テーマをLLMで抽出
  └─ [NERExtractor]        固有名詞をLLMで抽出
    ↓
[Parallel]
  ├─ [CosineSimilarityBuilder]  意味的に似たドキュメント同士をエッジで繋ぐ
  └─ [OverlapScoreBuilder]      固有名詞が重なるドキュメント同士をエッジで繋ぐ
    ↓
ナレッジグラフ完成 → TestsetGenerator がクエリ生成に使用
```
