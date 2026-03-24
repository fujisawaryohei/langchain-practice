from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="step to make the dish")

output_parse = PydanticOutputParser(pydantic_object=Recipe)

format_instructions = output_parse.get_format_instructions()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを考えてください。\n\n" "{format_instructions}",),
        ("human", "{dish}")
    ]
)

prompt_with_format_instructios = prompt.partial(
    format_instructions=format_instructions
)
prompt_value = prompt_with_format_instructios.invoke({"dish":"カレー"})

model = ChatAnthropic(model="claude-haiku-4-5",temperature=0)
ai_message = model.invoke(prompt_value)
print(ai_message.content)