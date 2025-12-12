# AI Procurement Assistant

An AI-powered procurement assistant that answers natural-language questions about procurement data using LLM-powered agents and MongoDB.

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
- **Chat with the agent**: Ask questions like:
  - "How many orders were created in Q1 2023?"
  - "Which quarter has the highest total spending?"
  - "What are the top 10 most frequently ordered line items?"

## Project Structure

- `app/` – FastAPI application and agent logic
- `scripts/` – Data loading utilities
- `static/` – Web interface files
- `start_server.bat` – Server startup script
