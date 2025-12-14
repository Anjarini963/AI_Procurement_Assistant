# AI Procurement Assistant

An AI-powered procurement assistant that answers natural-language questions about procurement data using LLM-powered agents and MongoDB. The assistant provides both historical analysis and predictive forecasting capabilities, enabling users to gain insights into procurement patterns, trends, and future projections.

## Overview

This application combines LangChain agents with MongoDB to create an intelligent procurement analytics system. Users can ask questions in natural language, and the agent automatically selects and executes the appropriate tools to analyze procurement data, identify trends, and make predictions about future spending, demand, pricing, and more.

### Key Features

- **Natural Language Querying**: Ask questions in plain English about procurement data
- **Historical Analysis**: Query spending, orders, suppliers, departments, and commodities
- **Predictive Analytics**: Forecast future spending, identify growth trends, predict seasonal patterns
- **Price Analysis**: Compare supplier prices, forecast price trends, identify cost-saving opportunities
- **Strategic Recommendations**: Get suggestions for statewide contracts, declining items, and optimization opportunities

## Prediction & Forecasting Capabilities

The assistant includes **8 projection tools** that enable comprehensive forecasting:

1. **predict_spending_trends**: Predict future spending (overall or by department/commodity/supplier) using linear regression
2. **predict_trends_by_category**: General-purpose tool to predict trends for any category over time periods (year/quarter/month)
3. **identify_growth_commodities**: Identify commodities with positive growth trends
4. **identify_declining_items**: Find items with declining usage/spending that need attention
5. **predict_seasonal_patterns**: Identify seasonal spending patterns and predict future seasonal trends
6. **forecast_price_trends**: Forecast future unit price trends for items or commodities
7. **recommend_suppliers_by_price**: Recommend suppliers with historically lower unit prices
8. **recommend_statewide_contracts**: Identify items suitable for statewide contracts based on volume and cross-department usage


The prediction tools use statistical methods including linear regression, CAGR calculations, seasonal analysis, and time series forecasting, all working with cleaned and normalized data from MongoDB.

## Prerequisites

- Python 3.12
- MongoDB instance (local or cloud, e.g. MongoDB Atlas)
- OpenAI API key (must support tools/function calling and work with LangChain)

## Quick Start

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Set up environment variables**

Create a `.env` file in the project root:

```bash
MONGODB_URI=your_uri_here (ex: mongodb+srv:/.....)
MONGODB_DB_NAME=db_name
MONGODB_COLLECTION_NAME=collection_name
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=openai/gpt-4.1 (or whichever model you choose to use)
```

**Note**: This project was developed and tested with **GPT-4.1**, which is guaranteed to work. Ensure your OpenAI API key has access to models that support tools/function calling (required for LangChain integration).

3. **Load CSV data into MongoDB**

Run the data loading script to populate MongoDB with your procurement data:

```bash
python scripts/load_data.py --csv-path "path\to\california_procurements.csv"
```

The script will:
- Read the CSV file in chunks (5000 records at a time)
- Convert currency and numeric fields appropriately
- Insert all records into the MongoDB collection specified in your `.env` file

**Note**: The script will clear existing data before loading, so it's safe to run multiple times.

4. **Start the server**

Double-click `start_server.bat` or run:

```bash
start_server.bat
```

The server will start at `http://127.0.0.1:8000`

## Using the Assistant

Once the server is running, you can:

- **Access the web interface**: Open `http://127.0.0.1:8000` in your browser
- **Chat with the agent**: Ask questions in natural language

### Example Questions

**High-Level Spending Analysis:**

"What is the total spending per fiscal year?"

"Which fiscal year had the highest total spending?"

"How did overall spending change from 2012 to 2015?"

"What is the average spending per purchase order over time?"

"Which department spent the most in a given fiscal year?"

**Procurement Methods & Purchasing Channels:**

"Which acquisition methods were used most frequently?"

"Which acquisition method accounts for the highest total spending?"

"How many purchases were made using CalCard versus standard purchase orders?"

"Are certain departments more likely to use specific acquisition methods?"

"Which Leveraged Procurement Agreements (LPAs) were used most often?"

**Commodity & Classification Analysis:**

"Which commodity titles represent the highest total spending?"

"Which UNSPSC segments dominate procurement spending?"

"How does spending vary across families and classes?"

"Which commodities experienced the largest year-to-year spending changes?"

"What are the most commonly purchased items?"

**Supplier-Focused Analysis:**

"Which suppliers received the highest total payments?"

"What is the average order value for a specific supplier?"

"Which suppliers received the largest single purchase orders?"

"Which departments rely most heavily on a single supplier?"

"What geographic patterns exist based on supplier ZIP codes?"

**Operational & Lookup Queries:**

"Retrieve a purchase order by Purchase Order Number"

"List all purchase orders for a specific department and year"

"Find all purchases created or purchased on a specific date"

"Show all purchases where the item description contains a specific keyword"

"List all purchases with a unit price above a defined threshold"


## Project Structure

- `app/` – FastAPI application and agent logic
- `scripts/` – Data loading utilities
- `static/` – Web interface files
- `start_server.bat` – Server startup script
