import dspy
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from dspy.adapters.types.tool import Tool
import json
import yfinance as yf

lm = dspy.LM("gemini/gemini-2.5-flash", api_key="API-KEY-HERE")  # Replace with your actual API key
dspy.configure(lm=lm)

yahoo_finance_tool = YahooFinanceNewsTool()

finance_news_tool = Tool.from_langchain(yahoo_finance_tool)

def get_stock_price(ticker: str) -> str:
    """Get current stock price and basic info."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1d")

        if hist.empty:
            return f"Could not retrieve data for {ticker}"

        current_price = hist['Close'].iloc[-1]
        prev_close = info.get('previousClose', current_price)
        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0

        result = {
            "ticker": ticker,
            "price": round(current_price, 2),
            "change_percent": round(change_pct, 2),
            "company": info.get('longName', ticker)
        }

        return json.dumps(result)
    except Exception as e:
        return f"Error: {str(e)}"

def compare_stocks(tickers: str) -> str:
    """Compare multiple stocks (comma-separated)."""
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(',')]
        comparison = []

        for ticker in ticker_list:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1d")

            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = info.get('previousClose', current_price)
                change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0

                comparison.append({
                    "ticker": ticker,
                    "price": round(current_price, 2),
                    "change_percent": round(change_pct, 2)
                })

        return json.dumps(comparison)
    except Exception as e:
        return f"Error: {str(e)}"
    

class FinancialAnalysisAgent(dspy.Module):
    """ReAct agent for financial analysis using Yahoo Finance data."""

    def __init__(self):
        super().__init__()

        # Combine all tools
        self.tools = [
            finance_news_tool,  # LangChain Yahoo Finance News
            get_stock_price,
            compare_stocks
        ]

        # Initialize ReAct
        self.react = dspy.ReAct(
            signature="financial_query -> analysis_response", # Input and output signature
            tools=self.tools,
            max_iters=6
        )

    def forward(self, financial_query: str):
        return self.react(financial_query=financial_query)
    

def run_financial_demo():
    """Demo of the financial analysis agent."""

    # Initialize agent
    agent = FinancialAnalysisAgent()

    # Example queries
    queries = [
        "What's the latest news about Apple (AAPL) and how might it affect the stock price?",
        "Compare AAPL, GOOGL, and MSFT performance",
        "Find recent Tesla news and analyze sentiment"
    ]

    for query in queries:
        print(f"Query: {query}")
        response = agent(financial_query=query)
        print(f"Analysis: {response.analysis_response}")
        print("-" * 50)

# Run the demo
if __name__ == "__main__":
    run_financial_demo()