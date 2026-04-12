from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableParallel

load_dotenv()

# 基本
# @chain
# def upper(text: str) -> str:
#     return text.upper()

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are helpful assistant."),
#         ("human", "{input}")
#     ]
# )

# model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# output_parser = StrOutputParser()

# chain = prompt | model | output_parser | upper
# output = chain.invoke({"input": "Hello!"})

# print(output)

model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
output_parser = StrOutputParser()
optimistic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは楽観主義者です。ユーザーの入力に対して楽観的な意見をください。"),
        ("human", "{topic}")
    ]
)
pessimistic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは悲観主義者です。ユーザーの入力に対して悲観的な意見をください。"),
        ("human", "{topic}")
    ]
)
synthesize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは客観的AIです。2つの意見をまとめてください。"),
        ("human", "楽観的意見: {optimistic_opinion}\n悲観的意見: {pessimistic_opinion}")
    ]
)

optimistic_chain = optimistic_prompt | model | output_parser
pessimistic_chain = pessimistic_prompt | model | output_parser


synthesize_chain = RunnableParallel(
    {
        "optimistic_opinion": optimistic_chain,
        "pessimistic_opinion": pessimistic_chain
    }
) | synthesize_prompt | model | output_parser

output = synthesize_chain.invoke({"topic": "生成AIの進化について"})
print(output)