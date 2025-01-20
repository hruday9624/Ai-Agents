from crewai import Crew,Process
from tools import yt_tool
from agents import blog_researcher, blog_writer
from tasks import research_task, writing_task

## Create a crew
crew = Crew(
    agents=[blog_researcher, blog_writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True,
)

## Start the task exceution process with enhanced feedback
result=crew.kickoff(inputs={"topic":"Transformers"})
print(result)