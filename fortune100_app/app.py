"""
Fortune 100 Company Financials Web Application
Displays quarterly financial data for publicly listed Fortune 100 companies.
"""

from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

app = Flask(__name__)

# Cache for all-companies ML data
all_companies_cache = {
    "data": None,
    "model": None,
    "loading": False,
    "progress": 0,
    "total": 0,
    "lock": threading.Lock()
}

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


# ML Feature configuration
ML_FEATURE_KEYS = [
    'rd_pct', 'sga_pct', 'capex_pct', 'fcf_margin', 'operating_margin',
    'debt_to_equity', 'revenue_growth_qoq', 'revenue', 'interest_expense',
    'buybacks', 'dividends_paid', 'ebitda', 'op_cashflow'
]

ML_FEATURE_LABELS = {
    'revenue': 'Revenue',
    'rd_pct': 'R&D Intensity (% Rev)',
    'sga_pct': 'SG&A (% Rev)',
    'capex_pct': 'CapEx Intensity (% Rev)',
    'fcf_margin': 'Free Cash Flow Margin',
    'operating_margin': 'Operating Margin',
    'debt_to_equity': 'Debt-to-Equity Ratio',
    'revenue_growth_qoq': 'Revenue Growth (QoQ)',
    'interest_expense': 'Interest Expense',
    'buybacks': 'Share Buybacks',
    'dividends_paid': 'Dividends Paid',
    'ebitda': 'EBITDA',
    'op_cashflow': 'Operating Cash Flow'
}


def generate_explanation(factor_key, factor_label, rank, mode, coefficient_or_importance,
                         correlation, company_name, percentile=None):
    """Generate a plain-language explanation for why a factor was selected."""
    direction = "positively" if correlation > 0 else "negatively"
    corr_abs = abs(correlation)

    if corr_abs > 0.7:
        strength = "strongly"
    elif corr_abs > 0.4:
        strength = "moderately"
    else:
        strength = "weakly"

    # Avoid double periods (e.g., "Apple Inc..")
    name = company_name.rstrip('.')

    if mode == "single":
        explanation = (
            f"{factor_label} was the #{rank} predictor of net profit for {name}. "
            f"It is {strength} {direction} correlated with net income "
            f"(correlation: {correlation:+.2f}). "
            f"The Lasso model assigned it a coefficient of {coefficient_or_importance:+.3f}, "
            f"meaning it remained a significant driver even after accounting for other factors."
        )
    else:
        pct_importance = coefficient_or_importance * 100
        explanation = (
            f"{factor_label} is the #{rank} driver of net profit across all "
            f"{len(FORTUNE_100_COMPANIES)} Fortune 100 companies. "
            f"It accounts for {pct_importance:.1f}% of the Random Forest model's "
            f"predictive power and is {strength} {direction} correlated with net income "
            f"(correlation: {correlation:+.2f})."
        )
        if percentile is not None:
            ordinal = "th"
            p = int(percentile)
            if p % 10 == 1 and p != 11:
                ordinal = "st"
            elif p % 10 == 2 and p != 12:
                ordinal = "nd"
            elif p % 10 == 3 and p != 13:
                ordinal = "rd"
            explanation += (
                f" {company_name}'s value ranks in the {p}{ordinal} percentile among peers."
            )

    return explanation


def build_ml_dataframe(factors_data):
    """Build a clean DataFrame from profit factors data for ML analysis."""
    df = pd.DataFrame(factors_data)
    return df


def analyze_single_company(ticker_symbol):
    """Run Lasso regression on a single company's quarterly data to identify top profit drivers."""
    factors_data, error = get_profit_factors(ticker_symbol)
    if error:
        return None, error

    if not factors_data or len(factors_data) < 3:
        return None, "Insufficient quarterly data for analysis (need at least 3 quarters)."

    company_name = FORTUNE_100_COMPANIES.get(ticker_symbol, ticker_symbol)
    df = build_ml_dataframe(factors_data)

    # Extract target
    if 'net_income' not in df.columns or df['net_income'].notna().sum() < 3:
        return None, "Insufficient net income data for analysis."

    target = df['net_income'].copy()

    # Select available features
    available_features = []
    for key in ML_FEATURE_KEYS:
        if key in df.columns:
            non_null = df[key].notna().sum()
            if non_null >= len(df) // 2:
                available_features.append(key)

    if len(available_features) < 3:
        return None, "Insufficient feature data for ML analysis."

    X = df[available_features].copy()

    # Fill NaNs
    X = X.ffill().bfill().fillna(0)
    target = target.ffill().bfill().fillna(0)

    # Drop zero-variance columns
    variances = X.var()
    zero_var = variances[variances == 0].index.tolist()
    X = X.drop(columns=zero_var)
    available_features = [f for f in available_features if f not in zero_var]

    if len(available_features) < 3:
        return None, "Insufficient feature variance for ML analysis."

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    target_scaled = (target - target.mean()) / (target.std() + 1e-10)

    # Lasso regression
    n_samples = len(X_scaled)
    cv_folds = min(n_samples, 5)
    try:
        lasso = LassoCV(cv=cv_folds, max_iter=10000, random_state=42)
        lasso.fit(X_scaled, target_scaled.values)
        coefficients = lasso.coef_
    except Exception:
        # Fallback: use correlations only
        coefficients = np.zeros(len(available_features))

    # Correlations
    feature_correlations = {}
    for i, feat in enumerate(available_features):
        corr = np.corrcoef(X_scaled[:, i], target_scaled.values)[0, 1]
        feature_correlations[feat] = float(corr) if not np.isnan(corr) else 0.0

    # PCA for variance visualization
    n_components = min(len(available_features), n_samples - 1, 10)
    if n_components >= 1:
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        explained_variance = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    else:
        explained_variance = []

    # Build factor scores - use absolute Lasso coefficient as primary, correlation as tiebreaker
    all_lasso_zero = np.all(coefficients == 0)
    factor_scores = []
    for i, feat in enumerate(available_features):
        coef = float(coefficients[i])
        corr = feature_correlations[feat]
        if all_lasso_zero:
            importance = abs(corr)
        else:
            importance = abs(coef)
        factor_scores.append({
            'key': feat,
            'label': ML_FEATURE_LABELS.get(feat, feat),
            'coefficient': round(coef, 4),
            'correlation': round(corr, 4),
            'importance': round(importance, 4),
        })

    factor_scores.sort(key=lambda x: x['importance'], reverse=True)

    # Generate explanations for top 3
    top_3 = factor_scores[:3]
    for rank_idx, factor in enumerate(top_3):
        factor['rank'] = rank_idx + 1
        factor['explanation'] = generate_explanation(
            factor['key'], factor['label'], rank_idx + 1, "single",
            factor['coefficient'], factor['correlation'], company_name
        )

    # Net income trend
    net_income_trend = [
        {'quarter': q['quarter_display'], 'value': q.get('net_income')}
        for q in factors_data
    ]

    latest = factors_data[-1] if factors_data else {}

    methodology = (
        f"Lasso Regression (L1-regularized) analysis on {n_samples} quarters of data "
        f"across {len(available_features)} financial metrics. "
        f"Lasso automatically selects the most important features by zeroing out "
        f"less relevant ones. "
    )
    if all_lasso_zero:
        methodology += (
            "Note: Lasso zeroed out all coefficients (strong regularization), "
            "so rankings are based on direct correlation with net income."
        )

    return {
        'mode': 'single',
        'ml_technique': 'Lasso Regression (L1)',
        'top_factors': top_3,
        'all_factor_scores': factor_scores,
        'net_income_latest': latest.get('net_income'),
        'net_income_latest_display': format_large_number(latest.get('net_income')),
        'net_income_trend': net_income_trend,
        'pca_explained_variance': explained_variance,
        'n_quarters': n_samples,
        'features_used': [ML_FEATURE_LABELS.get(f, f) for f in available_features],
        'methodology_note': methodology,
    }, None


def fetch_company_factors(ticker_symbol):
    """Fetch profit factors for a single company (used in parallel fetching)."""
    try:
        data, error = get_profit_factors(ticker_symbol)
        if error or not data:
            return ticker_symbol, None
        return ticker_symbol, data
    except Exception:
        return ticker_symbol, None


def analyze_all_companies(selected_ticker):
    """Run Random Forest on all companies' data to identify global profit drivers."""
    global all_companies_cache

    company_name = FORTUNE_100_COMPANIES.get(selected_ticker, selected_ticker)

    # Check cache
    with all_companies_cache["lock"]:
        if all_companies_cache["loading"]:
            return None, "Data is still loading. Please check progress and try again."

        if all_companies_cache["data"] is not None:
            cached_df = all_companies_cache["data"]
            cached_model = all_companies_cache["model"]
            return _run_all_companies_analysis(
                cached_df, cached_model, selected_ticker, company_name
            ), None

        all_companies_cache["loading"] = True
        all_companies_cache["progress"] = 0
        all_companies_cache["total"] = len(FORTUNE_100_COMPANIES)

    # Fetch all companies in parallel
    all_rows = []
    tickers = list(FORTUNE_100_COMPANIES.keys())

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_company_factors, t): t for t in tickers
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            all_companies_cache["progress"] = completed

            ticker_sym, data = future.result()
            if data:
                for quarter in data:
                    quarter['_ticker'] = ticker_sym
                    all_rows.append(quarter)

    if not all_rows:
        with all_companies_cache["lock"]:
            all_companies_cache["loading"] = False
        return None, "Failed to fetch data for any companies."

    combined_df = pd.DataFrame(all_rows)

    # Train model
    model = _train_all_companies_model(combined_df)

    # Cache results
    with all_companies_cache["lock"]:
        all_companies_cache["data"] = combined_df
        all_companies_cache["model"] = model
        all_companies_cache["loading"] = False

    return _run_all_companies_analysis(
        combined_df, model, selected_ticker, company_name
    ), None


def _train_all_companies_model(df):
    """Train a Random Forest model on the combined dataset."""
    available_features = []
    for key in ML_FEATURE_KEYS:
        if key in df.columns:
            non_null = df[key].notna().sum()
            if non_null >= len(df) * 0.3:
                available_features.append(key)

    X = df[available_features].copy()
    target = df['net_income'].copy()

    # Drop rows where target is NaN
    valid_mask = target.notna()
    X = X[valid_mask]
    target = target[valid_mask]

    # Fill NaNs in features
    X = X.ffill().bfill().fillna(0)

    # Drop zero-variance columns
    variances = X.var()
    zero_var = variances[variances == 0].index.tolist()
    X = X.drop(columns=zero_var)
    available_features = [f for f in available_features if f not in zero_var]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, target.values)

    # Correlations
    target_scaled = (target - target.mean()) / (target.std() + 1e-10)
    correlations = {}
    for i, feat in enumerate(available_features):
        corr = np.corrcoef(X_scaled[:, i], target_scaled.values)[0, 1]
        correlations[feat] = float(corr) if not np.isnan(corr) else 0.0

    # PCA
    n_components = min(len(available_features), 10)
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    return {
        'rf': rf,
        'scaler': scaler,
        'available_features': available_features,
        'correlations': correlations,
        'pca_explained_variance': [round(float(v), 4) for v in pca.explained_variance_ratio_],
        'feature_importances': dict(zip(available_features, rf.feature_importances_)),
    }


def _run_all_companies_analysis(df, model, selected_ticker, company_name):
    """Produce analysis results for a selected company using the pre-trained model."""
    available_features = model['available_features']
    importances = model['feature_importances']
    correlations = model['correlations']

    # Build factor scores
    factor_scores = []
    for feat in available_features:
        imp = float(importances.get(feat, 0))
        corr = float(correlations.get(feat, 0))
        factor_scores.append({
            'key': feat,
            'label': ML_FEATURE_LABELS.get(feat, feat),
            'importance': round(imp, 4),
            'correlation': round(corr, 4),
        })

    factor_scores.sort(key=lambda x: x['importance'], reverse=True)

    # Get selected company data for percentile calculation
    company_df = df[df['_ticker'] == selected_ticker]

    top_3 = factor_scores[:3]
    for rank_idx, factor in enumerate(top_3):
        # Calculate percentile for selected company
        percentile = None
        if not company_df.empty and factor['key'] in company_df.columns:
            company_vals = company_df[factor['key']].dropna()
            all_vals = df[factor['key']].dropna()
            if len(company_vals) > 0 and len(all_vals) > 0:
                company_median = company_vals.median()
                percentile = float((all_vals < company_median).sum() / len(all_vals) * 100)
                percentile = round(percentile, 0)
                factor['percentile'] = percentile

        factor['rank'] = rank_idx + 1
        factor['explanation'] = generate_explanation(
            factor['key'], factor['label'], rank_idx + 1, "all",
            factor['importance'], factor['correlation'], company_name,
            percentile=percentile
        )

    # Net income trend for selected company
    net_income_trend = []
    if not company_df.empty:
        for _, row in company_df.iterrows():
            net_income_trend.append({
                'quarter': row.get('quarter_display', ''),
                'value': row.get('net_income'),
            })

    latest_ni = None
    if net_income_trend:
        latest_ni = net_income_trend[-1].get('value')

    n_companies = df['_ticker'].nunique()
    n_observations = len(df[df['net_income'].notna()])

    methodology = (
        f"Random Forest analysis trained on {n_observations} quarterly observations "
        f"across {n_companies} Fortune 100 companies and {len(available_features)} "
        f"financial metrics. Random Forest captures non-linear relationships and "
        f"feature interactions that linear models miss."
    )

    return {
        'mode': 'all',
        'ml_technique': 'Random Forest',
        'top_factors': top_3,
        'all_factor_scores': factor_scores,
        'net_income_latest': latest_ni,
        'net_income_latest_display': format_large_number(latest_ni),
        'net_income_trend': net_income_trend,
        'pca_explained_variance': model['pca_explained_variance'],
        'n_companies': n_companies,
        'n_observations': n_observations,
        'features_used': [ML_FEATURE_LABELS.get(f, f) for f in available_features],
        'methodology_note': methodology,
    }


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


@app.route("/api/ml-analysis/<ticker>")
def get_ml_analysis(ticker):
    """API endpoint for single-company Lasso ML analysis."""
    if ticker not in FORTUNE_100_COMPANIES:
        return jsonify({"error": "Invalid company ticker"}), 400

    result, error = analyze_single_company(ticker)

    if error:
        return jsonify({"error": error}), 500

    return jsonify({
        "ticker": ticker,
        "company_name": FORTUNE_100_COMPANIES[ticker],
        "analysis": result
    })


@app.route("/api/ml-analysis-all/<ticker>")
def get_ml_analysis_all(ticker):
    """API endpoint for all-companies Random Forest ML analysis."""
    if ticker not in FORTUNE_100_COMPANIES:
        return jsonify({"error": "Invalid company ticker"}), 400

    result, error = analyze_all_companies(ticker)

    if error:
        return jsonify({"error": error}), 500

    return jsonify({
        "ticker": ticker,
        "company_name": FORTUNE_100_COMPANIES[ticker],
        "analysis": result
    })


@app.route("/api/ml-analysis-all/status")
def get_ml_loading_status():
    """API endpoint to check all-companies data loading progress."""
    return jsonify({
        "loading": all_companies_cache["loading"],
        "progress": all_companies_cache["progress"],
        "total": all_companies_cache["total"],
        "cached": all_companies_cache["data"] is not None
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
