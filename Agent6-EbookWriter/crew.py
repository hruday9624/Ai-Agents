from crewai import Crew,Process
from agents import idea_research_agent, outline_planner_agent, content_writer_agent, editor_agent, publishing_agent
from tasks import idea_research_task, outline_planner_task, content_writing_task, editing_task, publishing_task

## Forming the tech focused crew with some enhanced configuration
crew=Crew(
    agents=[idea_research_agent, outline_planner_agent, content_writer_agent, editor_agent, publishing_agent],
    tasks=[idea_research_task, outline_planner_task, content_writing_task, editing_task, publishing_task],
    #process=Process.sequential,

)

## starting the task execution process wiht enhanced feedback
# Import required modules
# from crewai import Crew

def generate_ebook(topic: str) -> str:
    """Orchestrate the ebook creation workflow"""
    try:
        # Define your crew with agents and tasks
        ebook_crew = Crew(
            agents=[
                idea_research_agent,
                outline_planner_agent,
                content_writer_agent,
                editor_agent,
                publishing_agent
            ],
            tasks=[
                idea_research_task,
                outline_planner_task,
                content_writing_task,
                editing_task,
                publishing_task
            ],
            verbose=2  # Enable detailed logging
        )

        # Execute the workflow with enhanced feedback
        result = ebook_crew.kickoff(inputs={'topic': topic})  # ✅ Remove config argument

        
        # Get detailed process logs
        process_logs = ebook_crew.loom.get_full_logs()
        print(f"\n🔍 Process Analytics:\n{process_logs}")
        
        return result

    except Exception as e:
        print(f"🚨 Workflow failed: {str(e)}")
        return None

# Run the workflow in your app
if __name__ == "__main__":
    ebook_content = generate_ebook(topic="AI Agents in Cyber Security")
    
    if ebook_content:
        print("\n✅ Ebook Creation Successful!")
        print("📖 Final Output:")
        print(ebook_content)
    else:
        print("❌ Ebook creation failed. Check logs for details.")