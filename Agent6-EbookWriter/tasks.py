from crewai import Task
from tools import tool
from agents import idea_research_agent, outline_planner_agent, content_writer_agent, editor_agent, publishing_agent


## Task 1: Idea Research Task
idea_research_task = Task(
  description=(
    "Generate 3-5 creative ebook concepts based on the topic: {topic}. "
    "For each idea, validate its market demand using real-time data from Serper API. "
    "Check for trends, competitor ebooks, and audience interest. "
    "Rank the ideas based on their market viability and audience appeal. "
    "Your final report should include: "
    "1. A list of 3-5 ebook ideas. "
    "2. Market validation data for each idea (e.g., search volume, competition). "
    "3. A ranked recommendation of the best idea with justification."
  ),
  expected_output=(
    "A detailed report with: "
    "1. List of 3-5 ebook ideas. "
    "2. Market validation data for each idea. "
    "3. Ranked recommendation of the best idea."
  ),
  tools=[tool],  # Serper API tool is passed here
  agent=idea_research_agent,  # The Idea & Research Agent
)

## Task 2: Outline Planning Task
outline_planner_task = Task(
  description=(
    "Create a detailed chapter-by-chapter outline for the ebook idea: '{idea}'. "
    "Break the idea into 8-10 logical chapters. "
    "For each chapter, define 3-5 key subtopics or talking points. "
    "Ensure the sequence flows naturally (e.g., problem → solution → case studies → future trends). "
    "Example for 'AI Tutors': "
    "Chapter 1: Introduction to AI in Education → [Current challenges, AI's role, thesis statement]. "
    "Chapter 2: How AI Tutors Work → [Types of AI tutors, case studies, technical overview]. "
    "Your outline must include chapter titles, subtopics, and a flow explanation."
  ),
  expected_output=(
    "A structured outline with: "
    "1. 8-10 chapter titles. "
    "2. 3-5 subtopics per chapter. "
    "3. Explanation of the logical flow between chapters."
  ),
  tools=[],  # No external tools needed
  agent=outline_planner_agent,
)

## Task 3: Content Writing Task
content_writing_task = Task(
  description=(
    "Write a polished chapter for the ebook based on this outline: {outline}. "
    "Follow the structure and subtopics provided. "
    "Maintain a {tone} tone (e.g., conversational, professional, academic). "
    "Incorporate relevant data, quotes, and examples to support arguments. "
    "Example for Chapter 1: "
    "Start with a hook, state the thesis clearly, and include 2-3 supporting statistics. "
    "Ensure readability for the target audience."
  ),
  expected_output=(
    "A well-written chapter in Markdown format (.md) with: "
    "1. Clear headings and subheadings. "
    "2. Engaging narrative flow. "
    "3. Data/examples formatted as bullet points or callouts. "
    "4. A 'Sources' section listing references for included data."
  ),
  tools=[],  # No external tools needed (uses LLM's writing capabilities)
  agent=content_writer_agent,
)

## Task 4: Editing Task
editing_task = Task(
  description=(
    "Polish and refine the draft chapter: {chapter}. "
    "1. Fix all grammar, spelling, and punctuation errors. "
    "2. Simplify complex sentences and remove jargon. "
    "3. Add transitions to improve flow between paragraphs and chapters. "
    "4. Flag repetitive content or contradictory claims. "
    "5. Ensure the tone remains {tone}-consistent. "
    "Your edits should enhance readability while preserving the author's voice. "
    "Mark changes as comments (e.g., 'REPETITION: This point was made in Chapter 2')."
  ),
  expected_output=(
    "A polished chapter in Markdown format (.md) with: "
    "1. Error-free text. "
    "2. Clear, concise language. "
    "3. Smooth transitions. "
    "4. Inline comments flagging issues. "
    "5. A changelog summarizing key edits."
  ),
  tools=[],  # No external tools needed (uses LLM's editing capabilities)
  agent=editor_agent,
)

## Task 5: Publishing Task
publishing_task = Task(
  description=(
    "Format the final chapter draft into Markdown and optimize metadata. "
    "1. Convert the text to proper Markdown syntax (headings, lists, links). "
    "2. Add SEO-friendly metadata including title, keywords, and description. "
    "3. Ensure clean formatting for integration with the `crew.py` pipeline. "
    "Example for metadata: "
    "---\n"
    "title: AI Tutors: Personalized Learning at Scale\n"
    "keywords: AI, education, personalized learning\n"
    "description: Explore how AI tutors are revolutionizing education...\n"
    "---\n"
    "Validate Markdown syntax and remove any special characters that break formatting."
  ),
  expected_output=(
    "A Markdown (.md) file with: "
    "1. Properly formatted headings and content. "
    "2. Metadata block at the top. "
    "3. Clean syntax ready for publishing."
  ),
  tools=[],  # No external tools needed
  agent=publishing_agent,
)