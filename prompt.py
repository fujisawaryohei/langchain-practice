from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

load_dotenv()
# 基本
# prompt = PromptTemplate.from_template("""以下の料理のレシピを教えてください。
#                                       料理名: {dish}""")
# prompt_value = prompt.invoke({"dish": "カレー"})
# print(prompt_value)
                                      

# 会話履歴を含める
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant."),
#         MessagesPlaceholder("chat_history", optional=True),
#         ("human", "{input}")
#     ]
# )
# prompt_value = prompt.invoke(
#     {
#         "chat_history": [
#             HumanMessage(content="こんにちは！私はジョンと言います！"),
#             AIMessage("こんにちは、ジョンさん！どのようにお手伝いできますか？")
#         ],
#         "input": "私の名前がわかりますか？"
#     }
# )
# print(prompt_value)

# マルチモーダル
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            [
                {"type": "text", "text": "画像を説明してください。"},
                {"type": "image_url", "image_url": {"url": "{image_url}"}}
            ]
         )
    ]
)

image_url = "https://raw.githubusercontent.com/yoshidashingo/langchain-book/main/assets/cover.jpg"

prompt_value = prompt.invoke({ "image_url": image_url })
model = ChatAnthropic(model="claude-haiku-4-5",temperature=0)
ai_message = model.invoke(prompt_value)
print(ai_message.content)