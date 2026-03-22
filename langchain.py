from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

load_dotenv()

# ストリーム
# model = ChatAnthropic(model="claude-haiku-4-5",temperature=0)

# messages = [
#     SystemMessage("You are a helpful assistant."),
#     HumanMessage("こんにちは！")
# ]

# for chunk in model.stream(messages):
#     print(chunk.content, end="",flush=True)