import streamlit as st
import os
import re
import numexpr as ne
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper

# ------------------ SETUP ------------------

st.set_page_config(page_title="Math Solver")
st.title("🧮 Text To Math Problem Solver")

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model_name="qwen/qwen3-32b",
    groq_api_key=groq_api_key,
    max_tokens=400
)

wikipedia_wrapper = WikipediaAPIWrapper()

# ------------------ LOGIC ------------------

def is_math_expression(q):
    return bool(re.match(r"^[0-9+\-*/().^ ]+$", q))

def generate_response(question):

    # CASE 1 → math
    if is_math_expression(question):
        try:
            result = ne.evaluate(question)
            return f"Final Answer: {result}"
        except:
            return "Invalid math expression"

    # CASE 2 → Wikipedia
    if "what is" in question.lower() or "define" in question.lower():
        wiki_result = wikipedia_wrapper.run(question)

        response = llm.invoke(
            f"""
Use the text as reference. If formula exists, extract it.
If not, use your mathematical knowledge to give the correct standard formula.

Text:
{wiki_result}
"""

        )

        import re
        clean = response.content
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL)

        return clean.strip()

    # CASE 3 → reasoning
    response = llm.invoke(
         f"""
Solve the problem step by step and give ONLY the final answer.

Question: {question}
"""
    )

    return response.content

# ------------------ CHAT UI ------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a Math chatbot who can answer maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.text_area("Enter your question")

if st.button("Find my answer"):
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Generating response..."):
            response = generate_response(question)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)