import os
from dotenv import load_dotenv
import requests

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain.callbacks.base import BaseCallbackHandler
from streamlit.delta_generator import DeltaGenerator

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

class StreamingCustomStreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container: DeltaGenerator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_answer = []  # To accumulate tokens for the final answer
        self.parent_container = (
            container  # Use the passed container for real-time updates
        )
        self.current_stream = None  # A Streamlit element for streaming

    def on_llm_new_token(self, token: str, **kwargs):
        """Streams tokens to the UI in real-time."""
        if self.current_stream is None:
            # Initialize a new container for streaming if not already present
            self.current_stream = self.parent_container.empty()
        self.final_answer.append(token)  # Append token to the final answer
        self.current_stream.write("".join(self.final_answer))  # Stream the updated text

    def on_agent_finish(self, finish: dict, **kwargs):
        """Handles the final output without displaying the dictionary."""
        if self.current_stream:
            self.current_stream.empty()  # Clear the streaming container
        # Extract the clean output from the return_values of the agent
        final_output = finish.get("return_values", {}).get(
            "output", "No output available."
        )
        self.parent_container.write(final_output)  # Write only the final result
        self.final_answer = []  # Reset for future usage

    def on_error(self, error: Exception, **kwargs):
        """Handle any errors."""
        self.final_answer = []  # Reset on error
        if self.current_stream:
            self.current_stream.empty()
        self.parent_container.write(f"Error: {str(error)}")  # Display the error

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

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and toolkit
agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)

# input field for user message
user_prompt = st.chat_input("Ask me the weather for a location....")

if user_prompt:
    # adding user message to chat and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # sending user message to GPT and get displaying response
    with st.chat_message("assistant"):        
        st_callback = StreamingCustomStreamlitCallbackHandler(st.container())
        assistant_response = agent_executor.invoke(
            {"input_user_message": user_prompt}, {"callbacks": [st_callback]}
        )
        st.write(assistant_response["output"])
    assistant_response = assistant_response["output"]

    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_response}
    )