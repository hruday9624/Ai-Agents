from crewai import Crew,Process
from tasks import task_architect,task_auditor,task_author
from agents import architect,author,auditor

# 3. Create the Crew and Assign Tasks
blog_crew = Crew(
    agents=[architect, auditor, author],
    tasks=[task_architect, task_auditor, task_author],
    strategy="sequential",
    verbose=True  # You can set this to True to see the agents' reasoning
)

# 4. Run the Crew
final_blog_post = blog_crew.kickoff()

print("\n\nFinal Blog Post:")
print(final_blog_post)