import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages

import streamlit as st

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# configuring streamlit page settings
st.set_page_config(page_title=" ChatBot", page_icon="💬", layout="centered")

# streamlit page title
st.title("🤖 LangChain GPT ChatBot")

greeting_msg = "Hello👋! I'm your friendly AI assistant. Feel free to ask me anything or just have a chat. I'm here to help! 😊"

# initializing chat session in streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "assistant", "content": greeting_msg})

# displaying chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initializing memory to store conversation history
if "store" not in st.session_state:
    st.session_state.store = {}

def chatbot(input_user_message):
    # creating a prompt template
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant."),
            MessagesPlaceholder(variable_name="history_messages"),
            ("human", "{input_user_message}"),
        ]
    )

    # initializing OpenAI Chat model
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    trimmer = trim_messages(
        max_tokens=100,
        strategy="last",
        token_counter=llm,
        # Usually, we want to keep the SystemMessage
        include_system=True,
        # start_on="human" makes sure we produce a valid chat history
        start_on="human",
    )

    def get_session_history(session_id):
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        else:
            st.session_state.store[session_id].messages = trimmer.invoke(st.session_state.store[session_id].messages)
        return st.session_state.store[session_id]
    
    # Initializing the output parser
    output_parser = StrOutputParser()
 
    # Creating a Chain with the prompt and memory
    conversation_chain = chat_prompt | llm | output_parser

    model_with_memory = RunnableWithMessageHistory(
        conversation_chain,
        get_session_history,
        input_messages_key="input_user_message",
        history_messages_key="history_messages",
    )
    session_id = "1234"

    response = model_with_memory.stream(
        {"input_user_message": input_user_message},
        {"configurable": {"session_id": session_id}},
    )

    return response

# input field for user message
user_prompt = st.chat_input("Ask me a question or chat with me I am here to help....")

if user_prompt:
    # adding user message to chat and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # sending user message to GPT and get displaying response
    with st.chat_message("assistant"):
        assistant_response = st.write_stream(chatbot(user_prompt))

    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_response}
    )