from crewai import Task
from agents import architect, auditor, author

# 2. Create the Tasks
topic = "Rise of Ai Agents"

task_architect = Task(
    description=f"Brainstorm key benefits and create a detailed outline/initial draft about: '{topic}'.",
    agent=architect,
    expected_output="A detailed outline + initial draft of the blog post."
)

task_auditor = Task(
    description="Review the architect's draft. Verify facts, fix errors, and suggest structural improvements.",
    agent=auditor,
    expected_output="An edited/verified draft with suggestions.",
    depends_on=[task_architect]  # Ensures auditor runs after architect
)

task_author = Task(
    description="Expand and polish the audited draft into a final Markdown-formatted blog post.",
    agent=author,
    expected_output="Final engaging blog post in Markdown.",
    depends_on=[task_auditor]  # Critical missing dependency added
)