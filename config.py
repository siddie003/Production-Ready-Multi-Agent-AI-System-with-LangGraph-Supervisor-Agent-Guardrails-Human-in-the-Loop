import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

AVIATION_STACK_API_KEY = os.getenv("AVIATION_STACK_API_KEY")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

DATABASE_URL = os.getenv("DATABASE_URL")


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        reasoning_effort="low",
        max_tokens=1000,
        api_key=os.getenv("GROQ_API_KEY"),
    )