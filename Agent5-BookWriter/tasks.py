from crewai import Task
from agents import planning_agent, writing_agent, editing_agent, fact_checking_agent, publishing_agent

# Planning Task
planning_task = Task(
    description=(
        "Develop the comprehensive blueprint for {book_title} ({genre} genre)."
        "Create detailed character arcs that align with the {genre} theme."
        "Build a chapter-by-chapter outline with key plot points."
        "Establish world-building rules specific to {genre} conventions."
    ),
    expected_output="A structured 10-page PDF document containing: theme analysis, character bios, chapter outline, and world-building guide.",
    output_file="{book_title}_blueprint.pdf",  # Added for reference
    agent=planning_agent,
)

# Writing Task
writing_task = Task(
    description=(
        "Write {genre}-style chapters for {book_title} following the outline."
        "Ensure each chapter contains exactly 1000 words (±50 words tolerance)."
        "Integrate character development and world-building elements naturally."
        "Maintain {genre}-appropriate pacing and tone throughout."
    ),
    expected_output="10 polished chapter drafts in markdown format (1000 words each) with narrative continuity.",
    output_file="chapters/{book_title}_chapter_[CHAPTER_NUM].md",  # Added for version control
    agent=writing_agent,
)

# Editing Task
editing_task = Task(
    description=(
        "Perform line-by-line editing of {book_title} chapters."
        "Ensure grammatical accuracy and {genre}-style consistency."
        "Verify chapter-to-chapter continuity in plot and character arcs."
        "Implement industry-standard formatting for {genre} manuscripts."
    ),
    expected_output="Edited manuscript with tracked changes and style guide compliance report.",
    output_file="{book_title}_edited_manuscript.docx",  # Added for editorial workflow
    agent=editing_agent,
)

# Fact-Checking Task (No file output needed - kept as report)
fact_checking_task = Task(
    description=(
        "Verify all factual claims in {book_title} against credible sources."
        "Cross-check historical/cultural references for {genre} accuracy."
        "Validate technical details specific to {genre} requirements."
        "Flag any potential legal/compliance issues in content."
    ),
    expected_output="Fact-check report with error log and source credibility ratings.",
    agent=fact_checking_agent,
)

# Publishing Task
publishing_task = Task(
    description=(
        "Prepare {book_title} for {genre} market distribution."
        "Format manuscript for print/digital {genre} publishing standards."
        "Generate necessary metadata and ISBN registration."
        "Create platform-specific ebook versions (EPUB, MOBI, PDF)."
    ),
    expected_output="Publication-ready package containing formatted files and distribution metadata.",
    output_file="published/{book_title}_package.zip",  # Added for final delivery
    agent=publishing_agent,
)


