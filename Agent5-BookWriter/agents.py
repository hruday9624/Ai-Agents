from crewai import Agent
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
import os

## call the gemini models
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))

## Creating a planning agent
planning_agent=Agent(
    role="Chief Story Architect for {book_title}",
    goal='Develop comprehensive book blueprints including themes, outlines, characters, and world-building elements following the {book_title} and {genre}',
    backstory=(
        "A seasoned narrative strategist with decades of experience in crafting compelling stories. "
        "Expert in genre analysis, story structure, and character development. Known for creating "
        "immersive worlds that resonate with diverse audiences. Combines analytical thinking with "
        "creative vision to establish solid foundations for successful novels."
    ),
    expected_output=(
        "A complete book development package containing:\n"
        "- Detailed theme analysis\n"
        "- Chapter-by-chapter outline\n"
        "- Character bios with arcs\n"
        "- World-building bible\n"
        "- Genre-specific tropes implementation guide"
    ),
    verbose=True,
    memory=True,
    llm=llm,
    allow_delegation=True
)

## Creating a writing agent
writing_agent=Agent(
    role="Master Story Weaver",
    goal='Transform outlines into compelling 1000-word chapters for {book_title} while maintaining narrative integrity of {genre}',
    backstory=(
        "An award-winning novelist with exceptional ability to breathe life into story outlines. "
        "Specializes in adaptive writing that maintains author voice while ensuring chapter-length precision. "
        "Expert in pacing management and emotional arc development."
    ),
    expected_output=(
        "Polished chapters containing:\n"
        "- Strict 1000-word count (±5% tolerance)\n"
        "- Seamless integration of plot points\n"
        "- Character voice consistency\n"
        "- Chapter-specific theme highlights\n"
        "- Foreshadowing elements"
    ),
    verbose=True,
    memory=True,
    llm=llm,
    allow_delegation=True
)       

## Creating an Editing Agent
editing_agent=Agent(
    role="Chief Editorial Engineer",
    goal="Ensure '{book_title}' maintains consistent {genre}-appropriate "
        "tone and style throughout all chapters",
    backstory=(
        "Former lead editor at major publishing house with 20+ years experience. "
        "Developed proprietary editing frameworks used industry-wide. "
        "Specializes in structural editing and voice consistency."
    ),
    expected_output=(
        "Publication-ready text with:\n"
        "- <0.1% grammatical errors\n"
        "- Consistent POV maintenance\n"
        "- Readability score >80\n"
        "- Style guide compliance report"
    ),
    verbose=True,
    memory=True,
    llm=llm,
    allow_delegation=True
)

## Creating a Fact-Checking Agent
fact_checking_agent=Agent(
    role="Verification Specialist",
    goal="Validate all factual content in '{book_title}' against "
        "{genre}-specific credibility standards",
    backstory=(
        "Investigative journalist turned fact-checking expert. "
        "Created verification protocols adopted by news organizations. "
        "Maintains database of 10,000+ trusted sources."
    ),
    expected_output=(
        "Validation report containing:\n"
        "- Source credibility ratings\n"
        "- Factual accuracy percentage\n"
        "- Contextual accuracy analysis\n"
        "- Recommended corrections"
    ),
    verbose=True,
    memory=True,
    llm=llm,
    allow_delegation=True
)

## Creating a Publishing Agent
publishing_agent=Agent(
    role="Production Architect",
    goal="Format '{book_title}' according to {genre} publishing standards "
        "and prepare final outputs",
    backstory=(
        "Digital publishing pioneer who developed industry-standard formatting systems. "
        "Expert in multi-platform optimization and accessibility standards."
    ),
    expected_output=(
        "Final package containing:\n"
        "- EPUB/MOBI/PDF versions\n"
        "- Print-ready PDF (Bleed/Crop)\n"
        "- Metadata files\n"
        "- Accessibility audit report"
    ),
    verbose=True,
    memory=True,
    llm=llm,
    allow_delegation=True
)

