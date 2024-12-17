import openai
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

# configuring streamlit page settings
st.set_page_config(page_title=" ChatBot", page_icon="💬", layout="centered")

# streamlit page title
st.title("🤖 GPT ChatBot")

greeting_msg = "Hello 👋! I am your Summary Creator Assistant. 📝\nI can help you condense your paragraphs or long texts into clear and concise summaries.\nSimply share your text in the box below, and I'll summarize it for you in no time! 😊"

# initializing chat session in streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "assistant", "content": greeting_msg})
    # st.chat_message("assistant").markdown(greeting_msg)

# displaying chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def summarize_text(text):
    summary_response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a summarization assistant.",
            },
            {"role": "user", "content": f"Summarize the following text: {text}"},
            *st.session_state.chat_history,
        ],
    )

    summary = summary_response.choices[0].message.content
    return summary


# input field for user message
user_prompt = st.chat_input("Paste your query or paragraph to summarize...")

if user_prompt:
    # adding user message to chat and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # sending user message to GPT and get a response
    assistant_response = summarize_text(user_prompt)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_response}
    )

    # displaying GPT's response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)