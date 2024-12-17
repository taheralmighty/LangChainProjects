import os
from dotenv import load_dotenv
import requests

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

import streamlit as st

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
WeatherAPI = os.getenv("WEATHER_API_KEY")

# configuring streamlit page settings
st.set_page_config(page_title=" Weather Assistant", page_icon="🌥️", layout="centered")

# streamlit page title
st.title("🌥️ Real-Time Weather Assistant")

greeting_msg = "Hello👋! I'm your Real-Time Weather Assistant 🌞. Feel free to ask me about the weather of any location. I'm here to help! 😊"

# initializing chat session in streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({"role": "assistant", "content": greeting_msg})

# displaying chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def GetWeather(location):
    url = "https://weatherapi-com.p.rapidapi.com/current.json"

    querystring = {"q": location}

    headers = {
        "x-rapidapi-key": WeatherAPI,
        "x-rapidapi-host": "weatherapi-com.p.rapidapi.com",
    }

    response = requests.get(url, headers=headers, params=querystring)
    print(response.json())
    return response.json()

def chatbot(input_user_message):
    # creating a prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Provide a concise weather report for the specified location. Include:
        * Current temperature in degree celsius or fahrenheit.
        * Weather conditions (e.g., sunny, cloudy, rainy)
        * Wind speed
        * Humidity
        * Any relevant alerts or advisories
        """,
            ),
            MessagesPlaceholder(variable_name="history_messages", optional=True),
            ("human", "{input_user_message}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # initializing OpenAI Chat model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", streaming=True
    )  # , callbacks=[StreamingStdOutCallbackHandler()]

    tools = [
        Tool(
            name="GetWeather",
            func=GetWeather,
            description="useful for when you need to check the weather of a particular location. You should specify the location in your query.",
        )
    ]

    st_callback = StreamlitCallbackHandler(st.container())

    # Construct the OpenAI Tools agent
    agent = create_openai_tools_agent(llm, tools, prompt)

    # Create an agent executor by passing in the agent and toolkit
    agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True, callbacks=[st_callback])

    return agent_executor

# input field for user message
user_prompt = st.chat_input("Ask me the weather for a location....")

if user_prompt:
    # adding user message to chat and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # sending user message to GPT and get displaying response
    with st.chat_message("assistant"):        
        assistant_response = ""  # Initialize an empty string to accumulate response
        agent_executor = chatbot(user_prompt)  # Use the chatbot function
        assistant_response = st.write_stream(
            agent_executor.stream({"input_user_message": user_prompt})
        )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_response}
    )
