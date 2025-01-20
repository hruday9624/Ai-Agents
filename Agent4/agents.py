from crewai import Agent
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

## Call the Gemini Models
llm=ChatGoogleGenerativeAI(model="",
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))

## Create a senior data scientist agent
researcher=Agent(
    role="Senior Researcher",
    goal="Uncover ground breaking technologies in {topic}",
    back_story=(
        "Driven by curiosity, you're at the forefront of"
        "innovation, eager to explore and the share knowledge that could change"
        "the world."),
    tools=[],
    llm=llm,
    allow_deligation=True,
    verbose=True,
    memory=True
)

## Create a junior data scientist agent
writer_agent=Agent(
    role="Junior Writer",
    goal="Write a compelling tech blog post on {topic}",
    back_story=(
        "You're a talented writer, with a passion for technology."
        "You're eager to share your knowledge with the world."
    ),
    tools=[],
    llm=llm,
    allow_deligation=False,
    verbose=True,
    memory=True
)