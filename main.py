import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Missing GOOGLE_GEMINI_API_KEY. Please set it in the .env file.")

search_tool = DuckDuckGoSearchRun()

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)

agent = initialize_agent(
    tools=[Tool(name="Web Search", func=search_tool.run, description="Searches the web for real-time information.")],
    llm=llm,
    agent="zero-shot-react-description",  
    verbose=True
)

def web_search_agent(query):
    result = agent.run(query)
    return result

if __name__ == "__main__":
    query = "Latest advancements in AI"
    print(web_search_agent(query))
