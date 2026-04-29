from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_anthropic import ChatAnthropic

load_dotenv()


class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make dish")


output_parser = PydanticOutputParser(pydantic_object=Recipe)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "ユーザーが入力した料理のレシピを考えてください。 \n\n #{instruction}",
        ),
        ("human", "#{dish}"),
    ]
)

prompt_with_format_instructions = prompt.partial(
    instruction=output_parser.get_format_instructions()
)

model = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

# chain = prompt | model | StrOutputParser()
chain = prompt_with_format_instructions | model | output_parser

output_value: Recipe = chain.invoke({"dish": "魚"})

print(type(output_value))
print(output_value.ingredients)
