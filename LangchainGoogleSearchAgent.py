import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain.callbacks.base import BaseCallbackHandler
from streamlit.delta_generator import DeltaGenerator
from langchain_community.utilities import GoogleSearchAPIWrapper

import streamlit as st

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")

# configuring streamlit page settings
st.set_page_config(page_title=" Google Search Assistant", page_icon="🔎", layout="centered")

# streamlit page title
st.title("🔎 Real-Time Google Search Assistant")

greeting_msg = "Hello👋! I'm your Personal Google Search Assistant 🔍. Looking for [specific topic]? Let me help you find the latest information. 😊"

# initializing chat session in streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": greeting_msg}]

# displaying chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

class StreamingCustomStreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container: DeltaGenerator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_answer = []  # To accumulate tokens for the final answer
        self.container = container  # A fresh Streamlit container for each query
        self.current_stream = None  # A Streamlit element for streaming

    def reset(self):
        """Reset the handler state for a new query."""
        self.final_answer = []
        self.current_stream = None  # Clear the stream reference

    def on_llm_new_token(self, token: str, **kwargs):
        """Streams tokens to the UI in real-time."""
        if self.current_stream is None:
            # Create a new container for this query
            self.current_stream = self.container.empty()
        self.final_answer.append(token)  # Append token to the final answer
        self.current_stream.markdown(
            "".join(self.final_answer)
        )  # Stream tokens dynamically

    def on_agent_finish(self, finish: dict, **kwargs):
        """Handles the final output."""
        # Extract the clean output from the return_values of the agent
        final_output = finish.get("return_values", {}).get(
            "output", "No output available."
        )
        # Only log the final result; avoid writing it again
        self.final_answer = []  # Reset tokens for safety
        self.reset()

    def on_error(self, error: Exception, **kwargs):
        """Handle any errors."""
        self.final_answer = []  # Reset tokens on error
        if self.current_stream:
            self.current_stream.markdown(f"Error: {str(error)}")  # Display the error

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Google Search agent. Search the web to provide accurate answers to user queries.
    When responding:
    * Include the **Title** of the article where the answer was found.
    * Provide a concise **Answer** based on the information.
    * Add the **Link** to the source for reference.
    * Ensure the information is accurate and relevant to the user's query.
    """,
        ),
        MessagesPlaceholder(variable_name="history_messages", optional=True),
        ("human", "{input_user_message}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# initializing OpenAI Chat model
llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

google_search = GoogleSearchAPIWrapper()
google_tool = Tool(
    func=google_search.run,  # Callable function for the tool
    name="google-search",  # Name of the tool
    description="Search Google for recent results."  # Tool description
)

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, [google_tool], prompt)

# Create an agent executor by passing in the agent and toolkit
agent_executor = AgentExecutor(
    agent=agent,
    tools=[google_tool],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
)

user_prompt = st.chat_input("Ask me a question....")

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
        # st.write(assistant_response["output"])
    assistant_response = assistant_response["output"]

    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_response}
    )
