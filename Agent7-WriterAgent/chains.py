from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import os

def load_prompt(file_path, input_vars):
    import os
    from pathlib import Path
    
    # Get absolute path to this file's directory
    base_dir = Path(__file__).parent.absolute()
    # Ensure Windows-style path and handle spaces
    prompt_path = (base_dir / file_path.replace('/', '\\')).resolve()
    
    print(f"DEBUG: Full prompt path: {str(prompt_path)}")
    print(f"DEBUG: File exists: {prompt_path.exists()}")
    
    if not prompt_path.exists():
        available_files = list((base_dir / 'prompts').glob('*'))
        raise FileNotFoundError(
            f"Prompt file not found at: {prompt_path}\n"
            f"Available files in prompts directory: {available_files}"
        )
        
    try:
        # Try multiple encodings
        for encoding in ['utf-8-sig', 'utf-8', 'cp1252']:
            try:
                with open(prompt_path, "r", encoding=encoding) as f:
                    template = f.read()
                return PromptTemplate(input_variables=input_vars, template=template)
            except UnicodeError:
                continue
                
        raise UnicodeError("Failed to decode prompt file with any encoding")
    except Exception as e:
        print(f"ERROR loading prompt: {str(e)}")
        raise

def get_comment_chain(llm):
    prompt = load_prompt("prompts/comment_prompt.txt", ["post_excerpt", "hruday_thoughts"])
    return LLMChain(llm=llm, prompt=prompt)

def get_post_chain(llm):
    prompt = load_prompt("prompts/post_prompt.txt", ["raw_idea"])
    return LLMChain(llm=llm, prompt=prompt)

def get_repurpose_chain(llm):
    prompt = load_prompt("prompts/repurpose_prompt.txt", ["original_post", "hruday_thoughts"])
    return LLMChain(llm=llm, prompt=prompt)

def get_llm():
    from dotenv import load_dotenv
    load_dotenv("Agent CommentCrafter/.env")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please ensure it's set in Agent CommentCrafter/.env")
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=api_key
    )
