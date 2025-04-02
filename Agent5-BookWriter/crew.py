from crewai import Crew,Process
from agents import planning_agent, writing_agent, editing_agent, fact_checking_agent, publishing_agent
from tasks import planning_task, writing_task, editing_task, fact_checking_task, publishing_task

## Create a crew
book_writing_crew = Crew(
    agents=[planning_agent, writing_agent, editing_agent, fact_checking_agent, publishing_agent],
    tasks=[planning_task, writing_task, editing_task, fact_checking_task, publishing_task],
    process=Process.sequential,
)

## Start the task execution process with enhanced feedback
result = book_writing_crew.kickoff(
    inputs={
        "book_title": "Alice and the Borderland", 
        "genre": "Thriller",
        "chapter_count": 10,
        "target_wordcount": 10000
        }
)
print(result)