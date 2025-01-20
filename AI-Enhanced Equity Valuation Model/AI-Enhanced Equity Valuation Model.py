# Import necessary libraries
import openai
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Set up OpenAI API key for GPT-3 usage
openai.api_key = 'your_openai_api_key'

# Function to fetch financial data from Yahoo Finance API
def fetch_financial_data(ticker):
    """
    Fetches financial data for a given ticker from Yahoo Finance.
    Args:
    - ticker (str): The stock ticker symbol.
    Returns:
    - dict: The fetched financial data, including revenue, profit margins, and growth projections.
    """
    url = f"https://finance.yahoo.com/quote/{ticker}/financials"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # Get the HTML content of the financials page
    response = requests.get(url, headers=headers)
    
    # Parsing HTML content using BeautifulSoup to extract the relevant financials
    soup = BeautifulSoup(response.text, "html.parser")
    
    financials_data = {}
    try:
        # Extract Revenue, Net Income, and other key metrics (adapt based on page structure)
        revenue = soup.find("td", {"data-test": "TOTAL_REVENUE-value"}).text
        net_income = soup.find("td", {"data-test": "NET_INCOME_COMMON-value"}).text
        profit_margin = soup.find("td", {"data-test": "PROFIT_MARGIN-value"}).text
        
        financials_data = {
            'Revenue': revenue,
            'Net Income': net_income,
            'Profit Margin': profit_margin
        }
    except Exception as e:
        print(f"Error fetching financial data: {e}")
    
    return financials_data

# Function to apply DCF valuation model
def dcf_valuation(revenue, growth_rate, discount_rate, years=5):
    """
    Applies the Discounted Cash Flow (DCF) model to estimate the stock value.
    Args:
    - revenue (float): The current revenue.
    - growth_rate (float): The projected annual revenue growth rate.
    - discount_rate (float): The discount rate used to account for risk and time value of money.
    - years (int): The number of years for which the projection is made.
    Returns:
    - float: The estimated stock value based on DCF.
    """
    future_revenue = revenue * (1 + growth_rate) ** years  # Projected revenue in the future
    cash_flows = [revenue * (1 + growth_rate) ** i for i in range(1, years + 1)]  # Projected cash flows for each year
    discounted_cash_flows = [cf / ((1 + discount_rate) ** i) for i, cf in enumerate(cash_flows, start=1)]  # Discounted cash flows
    
    # DCF formula: The sum of the discounted cash flows
    dcf_value = sum(discounted_cash_flows) + future_revenue / ((1 + discount_rate) ** years)
    
    return dcf_value

# Function to generate AI-driven insights using GPT-3
def generate_valuation_insights(financial_data):
    """
    Generates AI-driven insights for stock valuation using GPT-3.
    Args:
    - financial_data (dict): The financial data of the company.
    Returns:
    - str: The generated insights into the stock valuation.
    """
    prompt = f"Generate a stock valuation based on the following financial data: {financial_data}. Provide a summary of the valuation and key insights."
    
    response = openai.Completion.create(
        engine="text-davinci-003", 
        prompt=prompt, 
        max_tokens=300, 
        n=1, 
        stop=None, 
        temperature=0.5
    )
    
    return response.choices[0].text.strip()

# Function to perform regression analysis to predict stock prices based on historical data
def predict_stock_price(historical_data):
    """
    Performs a simple linear regression to predict stock prices based on historical revenue data.
    Args:
    - historical_data (DataFrame): The historical revenue and stock price data.
    Returns:
    - float: The predicted stock price.
    """
    # Prepare the data for regression
    X = historical_data[['Revenue']]  # Independent variable (Revenue)
    y = historical_data['Stock Price']  # Dependent variable (Stock Price)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict the stock price based on future revenue projections
    predicted_price = model.predict([[historical_data['Revenue'].iloc[-1] * 1.1]])  # Predicting based on 10% future revenue growth
    return predicted_price[0]

# Example of analyzing financial data for stock ticker 'AAPL' (Apple)
ticker = 'AAPL'
financial_data = fetch_financial_data(ticker)
print("\nFetched Financial Data for", ticker)
print(financial_data)

# Apply DCF valuation model using fetched financial data (adjust values as necessary)
revenue = float(financial_data['Revenue'].replace(',', '').replace('$', ''))
growth_rate = 0.05  # Assumed 5% growth rate
discount_rate = 0.1  # Assumed 10% discount rate
dcf_value = dcf_valuation(revenue, growth_rate, discount_rate)
print(f"\nDiscounted Cash Flow (DCF) Valuation for {ticker}: ${dcf_value:.2f}")

# Generate AI-driven insights for stock valuation
valuation_insights = generate_valuation_insights(financial_data)
print("\nAI-Generated Valuation Insights:")
print(valuation_insights)

# Sample historical data for regression analysis (Revenue and Stock Price data)
historical_data = pd.DataFrame({
    'Revenue': [260174, 265595, 274515, 282880, 294000],  # Example revenue data (in millions)
    'Stock Price': [150, 160, 170, 180, 190]  # Example stock price data (in USD)
})

# Predict stock price based on historical data
predicted_stock_price = predict_stock_price(historical_data)
print(f"\nPredicted Stock Price based on historical data: ${predicted_stock_price:.2f}")

# Visualizing the relationship between revenue and stock price using a scatter plot
plt.scatter(historical_data['Revenue'], historical_data['Stock Price'], color='blue')
plt.plot(historical_data['Revenue'], historical_data['Stock Price'], color='red')  # Linear regression line
plt.title("Stock Price vs Revenue")
plt.xlabel("Revenue (in millions)")
plt.ylabel("Stock Price (USD)")
plt.show()

# Example of saving analysis results to a CSV file
valuation_results = {
    'Ticker': ticker,
    'DCF Valuation': dcf_value,
    'Predicted Stock Price': predicted_stock_price,
    'Valuation Insights': valuation_insights
}

valuation_df = pd.DataFrame([valuation_results])
valuation_df.to_csv("stock_valuation_results.csv", index=False)
print("\nStock valuation results saved to stock_valuation_results.csv")
