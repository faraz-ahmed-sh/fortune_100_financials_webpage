"""
Fortune 100 Company Financials Web Application
Displays quarterly financial data for publicly listed Fortune 100 companies.
"""

from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd

app = Flask(__name__)

# Fortune 100 companies (publicly traded companies from Fortune 100)
FORTUNE_100_COMPANIES = {
    # Technology
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc. (Google)",
    "META": "Meta Platforms Inc.",
    "CSCO": "Cisco Systems Inc.",
    "ORCL": "Oracle Corporation",
    "IBM": "IBM Corporation",
    "INTC": "Intel Corporation",
    "CRM": "Salesforce Inc.",
    "DELL": "Dell Technologies Inc.",
    "HPQ": "HP Inc.",
    "HPE": "Hewlett Packard Enterprise",
    # Financials & Insurance
    "BRK-B": "Berkshire Hathaway Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "MA": "Mastercard Inc.",
    "BAC": "Bank of America Corp.",
    "WFC": "Wells Fargo & Co.",
    "GS": "Goldman Sachs Group Inc.",
    "MS": "Morgan Stanley",
    "C": "Citigroup Inc.",
    "AIG": "American International Group",
    "MET": "MetLife Inc.",
    "PRU": "Prudential Financial Inc.",
    "TRV": "The Travelers Companies Inc.",
    "ALL": "The Allstate Corporation",
    "AXP": "American Express Company",
    # Healthcare & Pharma
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group Inc.",
    "MRK": "Merck & Co. Inc.",
    "ABBV": "AbbVie Inc.",
    "PFE": "Pfizer Inc.",
    "TMO": "Thermo Fisher Scientific Inc.",
    "ABT": "Abbott Laboratories",
    "LLY": "Eli Lilly and Company",
    "BMY": "Bristol-Myers Squibb Co.",
    "AMGN": "Amgen Inc.",
    "CI": "The Cigna Group",
    "ELV": "Elevance Health Inc.",
    "HUM": "Humana Inc.",
    "CVS": "CVS Health Corporation",
    "MCK": "McKesson Corporation",
    "CAH": "Cardinal Health Inc.",
    "COR": "Cencora Inc.",
    # Energy
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    "COP": "ConocoPhillips",
    "PSX": "Phillips 66",
    "VLO": "Valero Energy Corporation",
    "MPC": "Marathon Petroleum Corp.",
    # Consumer Goods & Retail
    "PG": "Procter & Gamble Co.",
    "KO": "The Coca-Cola Company",
    "PEP": "PepsiCo Inc.",
    "WMT": "Walmart Inc.",
    "COST": "Costco Wholesale Corporation",
    "HD": "The Home Depot Inc.",
    "MCD": "McDonald's Corporation",
    "NKE": "Nike Inc.",
    "TGT": "Target Corporation",
    "LOW": "Lowe's Companies Inc.",
    "KR": "The Kroger Co.",
    "GIS": "General Mills Inc.",
    "SYY": "Sysco Corporation",
    # Industrials & Aerospace
    "GE": "GE Aerospace",
    "HON": "Honeywell International Inc.",
    "CAT": "Caterpillar Inc.",
    "RTX": "RTX Corporation",
    "LMT": "Lockheed Martin Corporation",
    "BA": "The Boeing Company",
    "GD": "General Dynamics Corp.",
    "DE": "Deere & Company",
    "UPS": "United Parcel Service Inc.",
    "FDX": "FedEx Corporation",
    # Telecommunications
    "VZ": "Verizon Communications Inc.",
    "T": "AT&T Inc.",
    "TMUS": "T-Mobile US Inc.",
    "CMCSA": "Comcast Corporation",
    # Automotive
    "F": "Ford Motor Company",
    "GM": "General Motors Company",
    "TSLA": "Tesla Inc.",
    # Services & Consulting
    "ACN": "Accenture plc",
    "DHR": "Danaher Corporation",
    # Diversified / Other
    "DIS": "The Walt Disney Company",
    "NFLX": "Netflix Inc.",
    "TJX": "The TJX Companies Inc.",
    "MMM": "3M Company",
    "DOW": "Dow Inc.",
    "DD": "DuPont de Nemours Inc.",
    "PLD": "Prologis Inc.",
    "NEE": "NextEra Energy Inc.",
    "DUK": "Duke Energy Corporation",
    "SO": "The Southern Company",
    "D": "Dominion Energy Inc.",
    "SLB": "Schlumberger Limited",
    "EOG": "EOG Resources Inc.",
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


def safe_get(df, keys, column):
    """Safely extract a value from a DataFrame given a list of possible row keys."""
    if df is None or df.empty or column not in df.columns:
        return None
    for key in keys:
        if key in df.index:
            val = df.loc[key, column]
            if val is not None and not pd.isna(val):
                return float(val)
    return None


def get_profit_factors(ticker_symbol):
    """
    Fetch quarterly profit factor data for a given ticker.
    Returns R&D spend, SG&A, CapEx, buybacks, interest expense, free cash flow,
    and computed ratios like R&D % of revenue, SG&A % of revenue, etc.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)

        quarterly_financials = ticker.quarterly_financials
        quarterly_cashflow = ticker.quarterly_cashflow
        quarterly_balance = ticker.quarterly_balance_sheet

        if quarterly_financials.empty:
            return None, "No financial data available for this company."

        quarters = quarterly_financials.columns[:8]

        factors = []
        prev_revenue = None

        # Collect all quarter data first (newest to oldest from yfinance)
        raw_data = []
        for quarter in quarters:
            revenue = safe_get(quarterly_financials, ['Total Revenue', 'Operating Revenue'], quarter)
            net_income = safe_get(quarterly_financials, ['Net Income', 'Net Income Common Stockholders'], quarter)
            operating_income = safe_get(quarterly_financials, ['Operating Income', 'EBIT'], quarter)

            rd_spend = safe_get(quarterly_financials, ['Research Development', 'Research And Development'], quarter)
            sga_spend = safe_get(quarterly_financials, ['Selling General Administrative', 'Selling General And Administrative'], quarter)
            interest_expense = safe_get(quarterly_financials, ['Interest Expense', 'Net Interest Income'], quarter)
            total_opex = safe_get(quarterly_financials, ['Total Operating Expenses', 'Operating Expense'], quarter)
            ebitda = safe_get(quarterly_financials, ['EBITDA', 'Normalized EBITDA'], quarter)

            capex = safe_get(quarterly_cashflow, ['Capital Expenditures', 'Capital Expenditure'], quarter)
            buybacks = safe_get(quarterly_cashflow, ['Repurchase Of Stock', 'Common Stock Repurchased'], quarter)
            op_cashflow = safe_get(quarterly_cashflow, ['Total Cash From Operating Activities', 'Operating Cash Flow', 'Cash Flow From Continuing Operating Activities'], quarter)
            dividends_paid = safe_get(quarterly_cashflow, ['Dividends Paid', 'Common Stock Dividend Paid', 'Cash Dividends Paid'], quarter)

            total_debt = safe_get(quarterly_balance, ['Total Debt', 'Long Term Debt'], quarter)
            total_equity = safe_get(quarterly_balance, ['Total Stockholders Equity', 'Stockholders Equity', 'Common Stock Equity'], quarter)
            total_assets = safe_get(quarterly_balance, ['Total Assets'], quarter)
            employees = None  # yfinance provides this on the info object, not per quarter

            # Computed metrics
            rd_pct = (rd_spend / revenue * 100) if (rd_spend and revenue) else None
            sga_pct = (sga_spend / revenue * 100) if (sga_spend and revenue) else None
            operating_margin = (operating_income / revenue * 100) if (operating_income and revenue) else None
            net_margin = (net_income / revenue * 100) if (net_income and revenue) else None

            free_cash_flow = None
            if op_cashflow is not None and capex is not None:
                free_cash_flow = op_cashflow + capex  # capex is typically negative

            fcf_margin = (free_cash_flow / revenue * 100) if (free_cash_flow and revenue) else None

            capex_pct = (abs(capex) / revenue * 100) if (capex and revenue) else None

            debt_to_equity = (total_debt / total_equity) if (total_debt and total_equity and total_equity != 0) else None

            raw_data.append({
                "quarter": quarter.strftime("%Y-Q%q") if hasattr(quarter, 'strftime') else str(quarter)[:10],
                "quarter_display": quarter.strftime("%b %Y") if hasattr(quarter, 'strftime') else str(quarter)[:10],
                "revenue": revenue,
                "net_income": net_income,
                "operating_income": operating_income,
                "rd_spend": rd_spend,
                "rd_spend_display": format_large_number(rd_spend),
                "rd_pct": round(rd_pct, 2) if rd_pct is not None else None,
                "sga_spend": sga_spend,
                "sga_spend_display": format_large_number(sga_spend),
                "sga_pct": round(sga_pct, 2) if sga_pct is not None else None,
                "capex": capex,
                "capex_display": format_large_number(abs(capex)) if capex else "N/A",
                "capex_pct": round(capex_pct, 2) if capex_pct is not None else None,
                "free_cash_flow": free_cash_flow,
                "fcf_display": format_large_number(free_cash_flow),
                "fcf_margin": round(fcf_margin, 2) if fcf_margin is not None else None,
                "buybacks": buybacks,
                "buybacks_display": format_large_number(abs(buybacks)) if buybacks else "N/A",
                "interest_expense": interest_expense,
                "interest_expense_display": format_large_number(abs(interest_expense)) if interest_expense else "N/A",
                "total_debt": total_debt,
                "total_debt_display": format_large_number(total_debt),
                "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity is not None else None,
                "dividends_paid": dividends_paid,
                "dividends_paid_display": format_large_number(abs(dividends_paid)) if dividends_paid else "N/A",
                "op_cashflow": op_cashflow,
                "op_cashflow_display": format_large_number(op_cashflow),
                "ebitda": ebitda,
                "ebitda_display": format_large_number(ebitda),
                "operating_margin": round(operating_margin, 2) if operating_margin is not None else None,
                "net_margin": round(net_margin, 2) if net_margin is not None else None,
                "revenue_growth_qoq": None,  # computed after reversal
            })

        # Reverse to oldest-first for chronological display
        raw_data.reverse()

        # Compute quarter-over-quarter revenue growth
        for i, q in enumerate(raw_data):
            if i > 0 and raw_data[i - 1]["revenue"] and q["revenue"]:
                prev_rev = raw_data[i - 1]["revenue"]
                growth = ((q["revenue"] - prev_rev) / abs(prev_rev)) * 100
                q["revenue_growth_qoq"] = round(growth, 2)

        return raw_data, None

    except Exception as e:
        return None, f"Error fetching profit factors: {str(e)}"


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


@app.route("/api/profit-factors/<ticker>")
def get_profit_factors_api(ticker):
    """API endpoint to fetch profit factor analysis data for a company."""
    if ticker not in FORTUNE_100_COMPANIES:
        return jsonify({"error": "Invalid company ticker"}), 400

    data, error = get_profit_factors(ticker)

    if error:
        return jsonify({"error": error}), 500

    return jsonify({
        "ticker": ticker,
        "company_name": FORTUNE_100_COMPANIES[ticker],
        "factors": data
    })


@app.route("/api/companies")
def get_companies():
    """API endpoint to get list of available companies."""
    return jsonify(FORTUNE_100_COMPANIES)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
