from crewai import Task
from tools import yt_tool
from agents import blog_researcher, blog_writer

## Research Task
research_task = Task(
    description=(
        "Identify the video {topic}",
        "Get detailed information about from the channel",
    ),
    expected_output='A comprehensive 3 paragraphs long report based on the {topic} of video content.',
    tools=[yt_tool],
    agent=blog_researcher,
)

## Writing Task
writing_task = Task(
    description=(
        "Write a blog post on the video {topic}",
        "Create a compelling blog post on the video {topic}",
    ),
    expected_output='A well-crafted blog post on the video {topic} with engaging content and relevant information.',
    tools=[yt_tool],
    async_excetuion=False,
    agent=blog_writer,
    output_file="new-blog-post.md"
)