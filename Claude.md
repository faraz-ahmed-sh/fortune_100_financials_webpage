# Fortune 100 Financials Web Application

## Project Overview

An interactive web application that displays quarterly financial data for publicly traded Fortune 100 companies. Users can select a company from a dropdown and view financial metrics, profitability margins, balance sheet data, and interactive charts showing trends over the last 8 quarters.

## Tech Stack

**Backend:**
- Flask (Python web framework)
- yfinance (Yahoo Finance API wrapper)
- pandas (data processing)

**Frontend:**
- HTML5 with Jinja2 templating
- Vanilla JavaScript
- Chart.js for interactive charts
- CSS3 with responsive design

## Project Structure

```
fortune100_app/
├── app.py              # Flask application entry point
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Main HTML template with embedded JS
└── static/
    └── css/
        └── style.css   # Application styling
```

## Running the Application

```bash
# Install dependencies
pip install -r fortune100_app/requirements.txt

# Run the server
cd fortune100_app
python app.py
```

The app runs at `http://localhost:5000`

## API Endpoints

- `GET /` - Main page with company selector
- `GET /api/companies` - List of available Fortune 100 companies
- `GET /api/financials/<ticker>` - Quarterly financial data for a company

## Key Files

| File | Description |
|------|-------------|
| `app.py` | Flask routes, 30+ company tickers, data fetching/processing logic |
| `index.html` | UI with dropdown, data table, and Chart.js visualizations |
| `style.css` | Responsive styling with gradient header and mobile support |

## Development Notes

- Data is fetched live from Yahoo Finance (last 8 quarters)
- Numbers formatted as billions (B) or millions (M) USD
- Debug mode enabled in development
- Responsive design: desktop (1400px max), tablet, and mobile
