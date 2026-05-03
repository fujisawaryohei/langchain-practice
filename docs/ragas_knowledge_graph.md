# Ragas のナレッジグラフとテストデータ生成の仕組み

## 全体の流れ

```
GitLoader
    ↓
Document（NodeType.DOCUMENT）として読み込む
    ↓
transforms（default_transforms など）
    ├─ HeadlineSplitter などでチャンク化
    │     → Chunk（NodeType.CHUNK）が生成される
    ├─ 各ノードから情報を抽出（要約・テーマ・固有名詞）
    └─ ノード間の関係を計算してエッジを張る
    ↓
ナレッジグラフ完成
（DOCUMENT・CHUNK がノード、類似度などがエッジ）
    ↓
TestsetGenerator がグラフを使ってクエリ・回答を合成
    ↓
テストデータセット
```

## ノードとは

Ragas では、分割前の元ドキュメントと、チャンク化後の断片の両方を「ノード」と呼ぶ。

| ノード種別 | 説明 |
|---|---|
| `NodeType.DOCUMENT` | GitLoader で読み込んだ元ファイル1つ |
| `NodeType.CHUNK` | HeadlineSplitter などで分割した断片 |

## なぜナレッジグラフを構築するのか

SingleHop と MultiHop の両方のクエリを生成するため。

- **SingleHop**：1つのノードだけで答えられるクエリ
- **MultiHop**：エッジで繋がった複数ノードをたどらないと答えられないクエリ

グラフのエッジを使って「関係のあるノードの組み合わせ」を見つけることで、
MultiHop クエリが生成できる。

## 各 Synthesizer とグラフの使い方

| Synthesizer | グラフの使い方 |
|---|---|
| `SingleHopSpecificQuerySynthesizer` | 1つのノードだけを見てクエリ生成 |
| `MultiHopAbstractQuerySynthesizer` | エッジで繋がった複数ノードをたどってクエリ生成（抽象的な問い） |
| `MultiHopSpecificQuerySynthesizer` | エッジで繋がった複数ノードをたどってクエリ生成（具体的な問い） |

## 注意点

ナレッジグラフのエッジが少ない（ドキュメント間の関係が薄い）と、
MultiHop 系のクエリが生成しにくくなる。
transforms でしっかりエッジを張っておくことが重要。