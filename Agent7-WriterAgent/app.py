import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from chains import (
    get_llm,
    get_comment_chain,
    get_post_chain,
    get_repurpose_chain
)

st.set_page_config(page_title="Hruday’s LinkedIn Assistant", layout="centered")

st.title("🧠 Hruday’s AI Writing Assistant")
st.write("Generate LinkedIn posts, comments, and repurposed content — all in your unique tone.")

task_type = st.selectbox("What do you want to generate?", ["Comment", "Post", "Repurpose"])
llm = get_llm()

output = ""

if task_type == "Comment":
    post_excerpt = st.text_area("Paste the original post excerpt:")
    hruday_thoughts = st.text_area("What are your raw thoughts?")
    
    if st.button("Generate Comment"):
        chain = get_comment_chain(llm)
        output = chain.run(post_excerpt=post_excerpt, hruday_thoughts=hruday_thoughts)

elif task_type == "Post":
    raw_idea = st.text_area("What's your post idea or rough draft?")
    
    if st.button("Generate Post"):
        chain = get_post_chain(llm)
        output = chain.run(raw_idea=raw_idea)

elif task_type == "Repurpose":
    original_post = st.text_area("Paste the original post:")
    hruday_thoughts = st.text_area("What’s your new take / angle?")
    
    if st.button("Generate Repurposed Version"):
        chain = get_repurpose_chain(llm)
        output = chain.run(original_post=original_post, hruday_thoughts=hruday_thoughts)

if output:
    st.markdown("### ✨ Generated Output")
    st.markdown(f"```markdown\n{output}\n```")
