from crewai import Agent
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

## Call the Gemini Models
llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))

## Agent--1
architect = Agent(
    role='The Idea Generator & Initial Drafter',
    goal='Understand the blog topic and create a foundational initial draft.',
    backstory=(
        "You are a creative thinker and skilled at quickly outlining and drafting content.",
    )
    verbose=True,
    memory=True,
    tools=[],
    llm=llm,
)

## Agent--2
auditor = Agent(
    role='The Verifier & Editor',
    goal='Ensure the accuracy and clarity of the initial draft, preparing it for expansion.',
    backstory="You are a meticulous editor with a strong focus on facts and clear communication.",
    verbose=True,
    memory=True,
    tools=[],
    llm=llm,
)

## Agent--3
author = Agent(
    role='The Content Expander & Finalizer',
    goal='Transform the verified draft into a comprehensive and engaging blog post.',
    backstory="You are an experienced content writer skilled at elaborating on ideas and crafting compelling narratives.",
    verbose=True,
    memory=True,
    tools=[],
    llm=llm,
)