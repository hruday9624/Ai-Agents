from crewai import Agent
from tools import yt_tool
import os
from load_dotenv import load_dotenv
load_dotenv()

## Call the Gemini Model
from langchain_google_genai import ChatGoogleGenarativeAI
llm=ChatGoogleGenarativeAI(model="gemini-1.5-flash",
                    verbose=True,
                    temperature=0.5,
                    google_api_key=os.getevn("GOOGLE_API_KEY"))

# import os
# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
# os.environ['OPENAI_MODEL_NAME'] = "gpt-4-0125-preview"

## Create a senior blog content researcher
blog_researcher = Agent(
    role="Blog Content Researcher from Youtube Videos",
    goal="Get the relavant video content for the topic {topic} from youtube channel",
    backstory=("Expert in understanding the content of the video and extracting the relevant information for the blog"),
    tools=[yt_tool],
    llm=llm,
    allow_delegation=True,
    verbose=True,
    memory=True
)

## Creating a sr writer agent with youtube tool

blog_writer = Agent(
    role="Blog Writer",
    goal="Narrate compelling tech stories about the video {topic}",
    backstory=("With a flair for simplifying complex tech concepts,"
               "You craft engaging narratives that captivate and educate,"
               "bringing new discoveries to light in an accessible way."),
    tools=[yt_tool],
    allow_delegation=False,
    verbose=True,
    llm=llm,
    memory=True
) 