from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

OLLAMA_MODEL_ID = "llama3.2:1b"

# 1. Web Agent (Ollama)
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information and news",
    model=Ollama(id=OLLAMA_MODEL_ID),
    tools=[DuckDuckGoTools()],
    instructions=(
        "You are a web search specialist. When given a query, search the web for the most recent and relevant information. "
        "Always include at least two reputable sources in your responses, formatted as clickable links. "
        "Summarize the information clearly and concisely, and avoid speculation. "
        "If the information is not available, state so explicitly. "
        "Present your findings in markdown format."
    ),
    show_tool_calls=True,
    markdown=True,
)

# 2. Finance Agent (Ollama)
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data and stock information",
    model=Ollama(id=OLLAMA_MODEL_ID),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=(
        "You are a financial data analyst. Retrieve up-to-date financial data, including stock prices, analyst recommendations, and company information. "
        "Display all data in clear, well-formatted markdown tables with relevant metrics such as current price, daily change, market cap, and analyst ratings. "
        "If possible, provide a brief summary of the company's financial health. "
        "Always cite the data source. "
        "If data is unavailable, indicate this in your response."
    ),
    show_tool_calls=True,
    markdown=True,
)

# 3. Multi-Agent Coordinator (Ollama)
multi_agent = Agent(
    name="Multi-Agent Coordinator",
    role="Coordinate between web search and finance agents",
    model=Ollama(id=OLLAMA_MODEL_ID),
    tools=[DuckDuckGoTools(), YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=(
        "You are responsible for coordinating tasks between the web search and finance agents. "
        "For each user query, determine whether to use web search, financial data, or both. "
        "When presenting financial data, always use markdown tables with clear headers and include metrics such as price, change, and analyst recommendations. "
        "For web information, always include at least two reputable sources as clickable links. "
        "Summarize and synthesize information from both domains when relevant, and ensure your response is clear, concise, and well-structured. "
        "If information is unavailable, state this clearly. "
        "Always present your answer in markdown format."
    ),
    show_tool_calls=True,
    markdown=True,
)

response = multi_agent.run("what is the current stock price of nvidia")
print(response.content)
