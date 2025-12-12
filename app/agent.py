import os
import json
from typing import Any, Dict, List, Optional
from functools import lru_cache

from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from .db import get_procurement_collection, get_all_field_names

load_dotenv()

SYSTEM_PROMPT = """
You are an AI procurement assistant for large purchases made by the State of California.

You MUST use tools to answer questions about the data.

You have access to the following tools:
- get_schema_tool: returns the field names of the procurement collection.
- run_mongo_aggregation: executes a MongoDB aggregation pipeline with automatic pre-processing stages.

CRITICAL RULES:
1. You MUST NOT answer from reasoning alone.
2. Any question that involves data lookup, grouping, filtering, totals, averages, rankings, trends, or any numeric result
   MUST use run_mongo_aggregation.
3. If you are unsure about column names, you MUST call get_schema_tool BEFORE constructing a pipeline.
4. When calling run_mongo_aggregation, you MUST pass the pipeline as a list of aggregation stages, not as a string.
5. After receiving results from run_mongo_aggregation, you MUST summarize them clearly in natural language and
   you MUST base all numbers in your answer on the tool output.
6. NEVER fabricate numbers or results. ALWAYS rely on run_mongo_aggregation for anything involving the dataset.

DERIVED FIELDS AVAILABLE INSIDE run_mongo_aggregation:
The tool automatically prepends several stages before your pipeline runs. As a result, the following fields are
always available for use in your aggregation stages:

- TotalPriceClean   : numeric version of "Total Price". Invalid or missing values are treated as 0.
- UnitPriceClean    : numeric version of "Unit Price". Invalid or missing values are treated as 0.
- QuantityClean     : numeric version of "Quantity". Invalid or missing values are treated as 0.
- FiscalYearClean   : cleaned integer version of "Fiscal Year". May be null if the raw value is missing or invalid.
- CreationDateClean : date parsed from "Creation Date" if present, otherwise from "Purchase Date".
- Year              : calendar year derived from CreationDateClean (for example 2012, 2013, 2014, 2015).
- Month             : month number derived from CreationDateClean, from 1 to 12.
- Quarter           : quarter number derived from CreationDateClean, from 1 to 4.

NOTES ABOUT Year, Month, AND Quarter:
- Year, Month, and Quarter are NOT stored columns in MongoDB.
- They are computed automatically inside run_mongo_aggregation using CreationDateClean.
- You can safely group or sort by Year, Month, or Quarter in your pipelines.

DATE ASSUMPTIONS:
- The raw fields "Creation Date" and "Purchase Date" are strings formatted as MM/DD/YYYY, for example 03/12/2014.
- If both dates are missing or invalid for a record, CreationDateClean and Year may be null, and Month and Quarter may also be null.

GUIDANCE FOR COMMON QUESTIONS:
- For total spending per year or trends over time:
  - Group by Year and sum TotalPriceClean, then sort by Year.
- For questions explicitly about fiscal year:
  - Group by FiscalYearClean, ignore records where FiscalYearClean is null, and sum TotalPriceClean.
- For which year had the highest spending:
  - Group by Year, sum TotalPriceClean, sort by the sum descending, and limit to 1.
- For "Which department spent the most in 2014?":
  - Filter by Year equal to 2014, group by Department Name, sum TotalPriceClean, sort descending, and limit to 1.
- For "Which supplier received the most money overall?":
  - Group by Supplier Name, sum TotalPriceClean, sort descending, and limit to 1.
- For "Which suppliers only appear in a single year?":
  - Group by Supplier Name and Year, then count distinct years per supplier and filter to those that appear in exactly one year.

MONGODB SYNTAX REMINDERS:
- Aggregation stage operators must use the dollar-prefixed names: match, group, sort, limit, project, addFields, unwind, lookup.
- Field references also use a single dollar sign, such as TotalPriceClean, Year, Quarter.

ALWAYS:
- Use TotalPriceClean, UnitPriceClean, and QuantityClean for numeric calculations.
- Use Year, FiscalYearClean, Month, and Quarter for time-based questions.
- Call get_schema_tool if you need to inspect the raw field names.
- If the results include null or missing grouping keys, explain that clearly in your summary.
"""


def _get_llm() -> ChatOpenAI:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing LLM_API_KEY or OPENAI_API_KEY.")

    model_name = (
        os.getenv("LLM_MODEL_NAME")
        or os.getenv("OPENAI_MODEL_NAME")
        or os.getenv("MODEL_NAME")
        or "openai/gpt-4.1"
    )

    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")

    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
    )


@tool("get_schema_tool")
async def get_schema_tool() -> str:
    """Return the field names in the procurement collection."""
    keys = await get_all_field_names(limit=500)
    return ", ".join(keys) if keys else "Schema unknown."


class PipelineInput(BaseModel):
    pipeline: List[Dict[str, Any]]


@tool("run_mongo_aggregation", args_schema=None)
async def run_mongo_aggregation(pipeline: Any = None, **kwargs) -> str:
    """
    Execute a MongoDB aggregation pipeline.
    The assistant MUST pass a list of stages, not a dict or a string.

    This function automatically:
    - Adds numeric cleaning for "Total Price", "Unit Price", "Quantity", and "Fiscal Year"
      into TotalPriceClean, UnitPriceClean, QuantityClean, and FiscalYearClean.
    - Parses dates from "Creation Date" or "Purchase Date" into CreationDateClean
      using MM/DD/YYYY format.
    - Derives Year, Month, and Quarter from CreationDateClean.
    - Normalizes common mistakes like using 'group' instead of '$group'.
    """

    # Extract pipeline from kwargs if needed (LangChain sometimes passes kwargs)
    if pipeline is None and isinstance(kwargs, dict) and "pipeline" in kwargs:
        pipeline = kwargs.get("pipeline")

    # Convert {"0": {...}, "1": {...}} style dicts to a list, if needed
    if isinstance(pipeline, dict) and all(str(k).isdigit() for k in pipeline.keys()):
        pipeline = [v for k, v in sorted(pipeline.items(), key=lambda kv: int(kv[0]))]

    if not isinstance(pipeline, list):
        return "Error: pipeline must be a list of MongoDB stages."

    # --- Normalize stage operator keys (group -> $group, sort -> $sort, etc.) ---
    def _normalize_pipeline(stages: List[Any]) -> List[Any]:
        op_map = {
            "match": "$match",
            "group": "$group",
            "sort": "$sort",
            "limit": "$limit",
            "project": "$project",
            "addFields": "$addFields",
            "unwind": "$unwind",
            "lookup": "$lookup",
        }
        normalized: List[Any] = []

        for stage in stages:
            if isinstance(stage, dict):
                new_stage: Dict[str, Any] = {}
                for key, value in stage.items():
                    if not key.startswith("$") and key in op_map:
                        new_stage[op_map[key]] = value
                    else:
                        new_stage[key] = value
                normalized.append(new_stage)
            else:
                normalized.append(stage)

        return normalized

    pipeline = _normalize_pipeline(pipeline)

    # Numeric cleaning: convert to double/int, treat errors/null as 0,
    # EXCEPT FiscalYearClean where we keep null if it's invalid or missing.
    numeric_cast_stage = {
        "$addFields": {
            "TotalPriceClean": {
                "$convert": {
                    "input": "$Total Price",
                    "to": "double",
                    "onError": 0,
                    "onNull": 0,
                }
            },
            "UnitPriceClean": {
                "$convert": {
                    "input": "$Unit Price",
                    "to": "double",
                    "onError": 0,
                    "onNull": 0,
                }
            },
            "QuantityClean": {
                "$convert": {
                    "input": "$Quantity",
                    "to": "double",
                    "onError": 0,
                    "onNull": 0,
                }
            },
            "FiscalYearClean": {
                "$convert": {
                    "input": {
                        # Take the first 4 bytes/chars of "Fiscal Year", e.g. "2013" from "2013-2014"
                        "$substrBytes": ["$Fiscal Year", 0, 4]
                    },
                    "to": "int",
                    "onError": None,
                    "onNull": None,
                }
            },
        }
    }

    # Filter out obviously invalid / NaN-ish totals:
    # - For normal numbers, TotalPriceClean >= 0 is true.
    # - For NaN values, the comparison fails and those docs are dropped.
    numeric_filter_stage = {
        "$match": {
            "TotalPriceClean": {"$gte": 0}
        }
    }

    # Date normalization:
    # Prefer "Creation Date", fall back to "Purchase Date".
    # Dates are in "MM/DD/YYYY" format, e.g. "03/12/2014".
    date_normalization_stage = {
        "$addFields": {
            "CreationDateClean": {
                "$dateFromString": {
                    "dateString": {
                        "$ifNull": ["$Creation Date", "$Purchase Date"]
                    },
                    "format": "%m/%d/%Y",
                    "onError": None,
                    "onNull": None,
                }
            }
        }
    }

    # Derive Year from CreationDateClean
    year_stage = {
        "$addFields": {
            "Year": {
                "$cond": [
                    {"$ne": ["$CreationDateClean", None]},
                    {"$year": "$CreationDateClean"},
                    None,
                ]
            }
        }
    }

    # Derive Month and Quarter from CreationDateClean
    date_quarter_stages = [
        {"$addFields": {"Month": {"$month": "$CreationDateClean"}}},
        {"$addFields": {"Quarter": {"$ceil": {"$divide": ["$Month", 3]}}}},
    ]

    # Final pipeline: pre-processing + user pipeline
    full_pipeline = [
        numeric_cast_stage,
        numeric_filter_stage,
        date_normalization_stage,
        year_stage,
        *date_quarter_stages,
        *pipeline,
    ]

    collection = get_procurement_collection()

    try:
        cursor = collection.aggregate(full_pipeline)
        docs = [doc async for doc in cursor]
        return json.dumps(docs[:50], default=str)
    except Exception as e:
        return f"MongoDB aggregation error: {e}"


def _build_agent_executor() -> AgentExecutor:
    llm = _get_llm()
    tools = [get_schema_tool, run_mongo_aggregation]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        stream_runnable=False,
        handle_parsing_errors=True,
    )
    return executor


@lru_cache(maxsize=1)
def get_agent_executor() -> AgentExecutor:
    return _build_agent_executor()


# In-memory storage for chat histories (session-based)
# Structure: {session_id: [HumanMessage, AIMessage, ...]}
_chat_histories: Dict[str, List[BaseMessage]] = {}


def get_chat_history(session_id: Optional[str] = None) -> List[BaseMessage]:
    """Get chat history for a session, or empty list if no session or history exists."""
    if not session_id:
        return []
    return _chat_histories.get(session_id, [])


def clear_chat_history(session_id: Optional[str] = None) -> None:
    """Clear chat history for a session."""
    if session_id and session_id in _chat_histories:
        del _chat_histories[session_id]


def update_chat_history(session_id: Optional[str], user_message: str, ai_message: str) -> None:
    """Update chat history with a new user message and AI response."""
    if not session_id:
        return
    
    if session_id not in _chat_histories:
        _chat_histories[session_id] = []
    
    _chat_histories[session_id].append(HumanMessage(content=user_message))
    _chat_histories[session_id].append(AIMessage(content=ai_message))


async def answer_question(user_query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Answer a question using the agent, with optional chat history context."""
    executor = get_agent_executor()
    chat_history = get_chat_history(session_id)
    
    result = await executor.ainvoke(
        {
            "input": user_query,
            "chat_history": chat_history,
        }
    )

    answer = result.get("output", "")
    
    # Update chat history with this exchange
    update_chat_history(session_id, user_query, answer)

    return {
        "query": user_query,
        "answer": answer,
        "intermediate_steps": result.get("intermediate_steps", []),
    }
