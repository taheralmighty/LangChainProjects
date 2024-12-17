import requests
import json
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
WeatherAPI = os.getenv("WEATHER_API_KEY")

def query(location):
    url = "https://weatherapi-com.p.rapidapi.com/current.json"

    querystring = {"q": location}
    # querystring = json.dumps({"q": q})

    headers = {
    	"x-rapidapi-key": WeatherAPI,
    	"x-rapidapi-host": "weatherapi-com.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response.json())
    # payload = json.dumps({"q": q})
    # url = "https://google.serper.dev/search"
    # headers = {"X-API-KEY": "YOUR-API-KEY", "Content-Type": "application/json"}

    # response = requests.request("POST", url, headers=headers, data=payload)

    return response.json()

tools = [
    Tool(
        name="search",
        func=query,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    )
]

llm = ChatOpenAI(model="gpt-3.5-turbo")

# prompt
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
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and toolkit
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": "Get me the current weather for Hyderabad"})
print(response["output"])