"""
Fortune 100 Company Financials Web Application
Displays quarterly financial data for publicly listed Fortune 100 companies.
"""

from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd

app = Flask(__name__)

# Fortune 100 companies (subset of major publicly traded companies)
FORTUNE_100_COMPANIES = {
    "AAPL": "Apple Inc.",
    "",
    "",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc. (Google)",
    "BRK-B": "Berkshire Hathaway Inc.",
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group Inc.",
    "XOM": "Exxon Mobil Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "PG": "Procter & Gamble Co.",
    "MA": "Mastercard Inc.",
    "HD": "The Home Depot Inc.",
    "CVX": "Chevron Corporation",
    "MRK": "Merck & Co. Inc.",
    "ABBV": "AbbVie Inc.",
    "PFE": "Pfizer Inc.",
    "KO": "The Coca-Cola Company",
    "PEP": "PepsiCo Inc.",
    "COST": "Costco Wholesale Corporation",
    "TMO": "Thermo Fisher Scientific Inc.",
    "WMT": "Walmart Inc.",
    "MCD": "McDonald's Corporation",
    "CSCO": "Cisco Systems Inc.",
    "ABT": "Abbott Laboratories",
    "ACN": "Accenture plc",
    "DHR": "Danaher Corporation",
    "NKE": "Nike Inc.",
    "VZ": "Verizon Communications Inc.",
    "T": "AT&T Inc.",
}


def format_large_number(num):
    """Format large numbers to readable format (B for billions, M for millions)."""
    if num is None or pd.isna(num):
        return "N/A"
    if abs(num) >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.2f}M"
    else:
        return f"${num:,.0f}"


def format_percentage(num):
    """Format number as percentage."""
    if num is None or pd.isna(num):
        return "N/A"
    return f"{num:.2f}%"


def get_quarterly_financials(ticker_symbol):
    """
    Fetch last 8 quarters of financial data for a given ticker.
    Returns revenue, profit, margin, and other key metrics.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)

        # Get quarterly financials
        quarterly_financials = ticker.quarterly_financials
        quarterly_balance = ticker.quarterly_balance_sheet

        if quarterly_financials.empty:
            return None, "No financial data available for this company."

        # Get up to 8 quarters of data
        quarters = quarterly_financials.columns[:8]

        financial_data = []

        for quarter in quarters:
            quarter_data = {
                "quarter": quarter.strftime("%Y-Q%q") if hasattr(quarter, 'strftime') else str(quarter)[:10],
                "quarter_display": quarter.strftime("%b %Y") if hasattr(quarter, 'strftime') else str(quarter)[:10],
            }

            # Revenue (Total Revenue)
            revenue = None
            for key in ['Total Revenue', 'Operating Revenue']:
                if key in quarterly_financials.index:
                    revenue = quarterly_financials.loc[key, quarter]
                    break
            quarter_data["revenue"] = revenue
            quarter_data["revenue_display"] = format_large_number(revenue)

            # Gross Profit
            gross_profit = None
            if 'Gross Profit' in quarterly_financials.index:
                gross_profit = quarterly_financials.loc['Gross Profit', quarter]
            quarter_data["gross_profit"] = gross_profit
            quarter_data["gross_profit_display"] = format_large_number(gross_profit)

            # Net Income (Profit)
            net_income = None
            for key in ['Net Income', 'Net Income Common Stockholders']:
                if key in quarterly_financials.index:
                    net_income = quarterly_financials.loc[key, quarter]
                    break
            quarter_data["net_income"] = net_income
            quarter_data["net_income_display"] = format_large_number(net_income)

            # Operating Income
            operating_income = None
            if 'Operating Income' in quarterly_financials.index:
                operating_income = quarterly_financials.loc['Operating Income', quarter]
            quarter_data["operating_income"] = operating_income
            quarter_data["operating_income_display"] = format_large_number(operating_income)

            # Calculate margins
            if revenue and revenue != 0:
                if gross_profit:
                    quarter_data["gross_margin"] = (gross_profit / revenue) * 100
                else:
                    quarter_data["gross_margin"] = None

                if net_income:
                    quarter_data["net_margin"] = (net_income / revenue) * 100
                else:
                    quarter_data["net_margin"] = None

                if operating_income:
                    quarter_data["operating_margin"] = (operating_income / revenue) * 100
                else:
                    quarter_data["operating_margin"] = None
            else:
                quarter_data["gross_margin"] = None
                quarter_data["net_margin"] = None
                quarter_data["operating_margin"] = None

            quarter_data["gross_margin_display"] = format_percentage(quarter_data["gross_margin"])
            quarter_data["net_margin_display"] = format_percentage(quarter_data["net_margin"])
            quarter_data["operating_margin_display"] = format_percentage(quarter_data["operating_margin"])

            # Total Debt (from balance sheet)
            total_debt = None
            if not quarterly_balance.empty and quarter in quarterly_balance.columns:
                for key in ['Total Debt', 'Long Term Debt', 'Total Liabilities Net Minority Interest']:
                    if key in quarterly_balance.index:
                        total_debt = quarterly_balance.loc[key, quarter]
                        break
            quarter_data["total_debt"] = total_debt
            quarter_data["total_debt_display"] = format_large_number(total_debt)

            # Total Assets
            total_assets = None
            if not quarterly_balance.empty and quarter in quarterly_balance.columns:
                if 'Total Assets' in quarterly_balance.index:
                    total_assets = quarterly_balance.loc['Total Assets', quarter]
            quarter_data["total_assets"] = total_assets
            quarter_data["total_assets_display"] = format_large_number(total_assets)

            financial_data.append(quarter_data)

        # Reverse to show oldest first (for charts)
        financial_data.reverse()

        return financial_data, None

    except Exception as e:
        return None, f"Error fetching data: {str(e)}"


@app.route("/")
def index():
    """Render the main page with company dropdown."""
    return render_template("index.html", companies=FORTUNE_100_COMPANIES)


@app.route("/api/financials/<ticker>")
def get_financials(ticker):
    """API endpoint to fetch financial data for a company."""
    if ticker not in FORTUNE_100_COMPANIES:
        return jsonify({"error": "Invalid company ticker"}), 400

    data, error = get_quarterly_financials(ticker)

    if error:
        return jsonify({"error": error}), 500

    return jsonify({
        "ticker": ticker,
        "company_name": FORTUNE_100_COMPANIES[ticker],
        "quarters": data
    })


@app.route("/api/companies")
def get_companies():
    """API endpoint to get list of available companies."""
    return jsonify(FORTUNE_100_COMPANIES)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
