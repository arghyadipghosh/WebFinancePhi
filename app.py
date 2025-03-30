import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
phi_api_key = os.getenv("PHI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if not google_api_key or not phi_api_key or not groq_api_key:
    raise ValueError("API keys not found in environment variables.")

# Configure environment variables
os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["PHI_API_KEY"] = phi_api_key
os.environ["GROQ_API_KEY"] = groq_api_key

# Agent Definitions
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Gemini(id="gemini-1.5-flash"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Gemini(id="gemini-1.5-flash"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=Gemini(id="gemini-1.5-flash"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Main Execution
agent_team.print_response(f"Summarize analyst recommendations and share the latest news for AMZN", stream=True)