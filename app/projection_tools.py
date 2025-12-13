"""
Projection and predictive analysis tools for the procurement dataset.

This module provides tools for:
- Predicting future spending based on historical trends
- Identifying commodities with growth potential
- Recommending suppliers based on price history
- Identifying items suitable for statewide contracts based on volume
"""

import json
import numpy as np
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from .db import get_procurement_collection


async def _get_historical_spending_by_year() -> Dict[int, float]:
    """Get historical total spending by year from the database."""
    collection = get_procurement_collection()
    
    pipeline = [
        {
            "$addFields": {
                "TotalPriceClean": {
                    "$convert": {
                        "input": "$Total Price",
                        "to": "double",
                        "onError": 0,
                        "onNull": 0,
                    }
                },
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
        },
        {
            "$match": {
                "TotalPriceClean": {"$gte": 0},
                "CreationDateClean": {"$ne": None}
            }
        },
        {
            "$addFields": {
                "Year": {"$year": "$CreationDateClean"}
            }
        },
        {
            "$group": {
                "_id": "$Year",
                "totalSpending": {"$sum": "$TotalPriceClean"}
            }
        },
        {
            "$sort": {"_id": 1}
        }
    ]
    
    result = {}
    async for doc in collection.aggregate(pipeline):
        year = doc.get("_id")
        if year:
            result[year] = doc.get("totalSpending", 0)
    
    return result


async def _get_commodity_spending_trends(commodity_field: str = "Commodity") -> Dict[str, List[Dict[str, float]]]:
    """Get spending trends by commodity over years."""
    collection = get_procurement_collection()
    
    # Build the group ID with proper field reference
    # Use a dictionary to construct the field reference dynamically
    group_id = {
        "commodity": f"${commodity_field}",
        "year": "$Year"
    }
    
    pipeline = [
        {
            "$addFields": {
                "TotalPriceClean": {
                    "$convert": {
                        "input": "$Total Price",
                        "to": "double",
                        "onError": 0,
                        "onNull": 0,
                    }
                },
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
        },
        {
            "$match": {
                "TotalPriceClean": {"$gte": 0},
                "CreationDateClean": {"$ne": None},
                commodity_field: {"$ne": None, "$exists": True}
            }
        },
        {
            "$addFields": {
                "Year": {"$year": "$CreationDateClean"}
            }
        },
        {
            "$group": {
                "_id": group_id,
                "totalSpending": {"$sum": "$TotalPriceClean"}
            }
        },
        {
            "$sort": {
                "_id.commodity": 1,
                "_id.year": 1
            }
        }
    ]
    
    result: Dict[str, List[Dict[str, float]]] = {}
    async for doc in collection.aggregate(pipeline):
        commodity = doc.get("_id", {}).get("commodity", "Unknown")
        year = doc.get("_id", {}).get("year")
        spending = doc.get("totalSpending", 0)
        
        if commodity and year:
            if commodity not in result:
                result[commodity] = []
            result[commodity].append({"year": year, "spending": spending})
    
    # Sort each commodity's data by year
    for commodity in result:
        result[commodity].sort(key=lambda x: x["year"])
    
    return result


async def _get_historical_spending_by_group_and_year(group_by_field: str, filter_value: Optional[str] = None) -> Dict[str, Dict[int, float]]:
    """Get historical spending grouped by a field and year."""
    collection = get_procurement_collection()
    
    match_conditions: Dict[str, Any] = {
        "TotalPriceClean": {"$gte": 0},
        "CreationDateClean": {"$ne": None},
        group_by_field: {"$ne": None, "$exists": True}
    }
    
    if filter_value:
        match_conditions[group_by_field] = filter_value
    
    pipeline = [
        {
            "$addFields": {
                "TotalPriceClean": {
                    "$convert": {
                        "input": "$Total Price",
                        "to": "double",
                        "onError": 0,
                        "onNull": 0,
                    }
                },
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
        },
        {
            "$match": match_conditions
        },
        {
            "$addFields": {
                "Year": {"$year": "$CreationDateClean"}
            }
        },
        {
            "$group": {
                "_id": {
                    "group": f"${group_by_field}",
                    "year": "$Year"
                },
                "totalSpending": {"$sum": "$TotalPriceClean"}
            }
        },
        {
            "$sort": {
                "_id.group": 1,
                "_id.year": 1
            }
        }
    ]
    
    result: Dict[str, Dict[int, float]] = {}
    async for doc in collection.aggregate(pipeline):
        group_key = doc.get("_id", {}).get("group", "Unknown")
        year = doc.get("_id", {}).get("year")
        spending = doc.get("totalSpending", 0)
        
        if group_key and year:
            if group_key not in result:
                result[group_key] = {}
            result[group_key][year] = spending
    
    return result


@tool("predict_spending_trends")
async def predict_spending_trends(
    next_year: Optional[int] = None,
    group_by: Optional[str] = None,
    filter_value: Optional[str] = None
) -> str:
    """
    Predict future spending based on historical trends using linear regression.
    
    Can predict overall spending or spending for specific groups (departments, commodities, suppliers, etc.).
    
    Args:
        next_year: The year to predict for. If None, predicts for the year after the latest data.
        group_by: Optional field to group by (e.g., "Department Name", "Commodity", "Supplier Name"). 
                  If None, predicts overall spending.
        filter_value: Optional value to filter by when group_by is specified (e.g., specific department name).
    
    Returns:
        A JSON string with prediction results including:
        - predicted_spending: The predicted total spending
        - historical_data: List of {year, spending} pairs (and group if group_by is used)
        - growth_rate: Average annual growth rate as a percentage
        - confidence: Notes on prediction confidence based on data quality
    """
    try:
        # If group_by is specified, get data grouped by that field
        if group_by:
            historical = await _get_historical_spending_by_group_and_year(group_by, filter_value)
        else:
            historical = await _get_historical_spending_by_year()
        
        # Handle grouped results differently
        if group_by:
            results = []
            for group_key, year_data in historical.items():
                if len(year_data) < 2:
                    continue
                
                years = sorted(year_data.keys())
                spendings = [year_data[y] for y in years]
                
                # Determine target year
                target_year = next_year if next_year else max(years) + 1
                
                # Linear regression
                X = np.array(years)
                y = np.array(spendings)
                coefficients = np.polyfit(X, y, 1)
                predicted = coefficients[0] * target_year + coefficients[1]
                
                # Calculate growth rate
                growth_rates = []
                for i in range(1, len(spendings)):
                    if spendings[i-1] > 0:
                        growth = ((spendings[i] - spendings[i-1]) / spendings[i-1]) * 100
                        growth_rates.append(growth)
                avg_growth = np.mean(growth_rates) if growth_rates else 0
                
                results.append({
                    group_by: group_key,
                    "predicted_spending": float(predicted),
                    "predicted_year": target_year,
                    "historical_data": [{"year": int(y), "spending": float(s)} for y, s in zip(years, spendings)],
                    "growth_rate": float(avg_growth),
                    "data_points": len(years)
                })
            
            return json.dumps({
                "predictions": results,
                "group_by": group_by,
                "method": "Linear regression on historical spending data"
            }, indent=2)
        
        # Original logic for overall spending
        if len(historical) < 2:
            return json.dumps({
                "error": "Insufficient historical data for prediction. Need at least 2 years of data.",
                "historical_data": [{"year": k, "spending": v} for k, v in sorted(historical.items())]
            })
        
        # Sort by year
        years = sorted(historical.keys())
        spendings = [historical[y] for y in years]
        
        # Determine target year
        if next_year is None:
            next_year = max(years) + 1
        
        # Prepare data for linear regression
        X = np.array(years)
        y = np.array(spendings)
        
        # Calculate linear regression (y = mx + b)
        # Using numpy's polyfit for simple linear regression
        coefficients = np.polyfit(X, y, 1)
        slope = coefficients[0]
        intercept = coefficients[1]
        
        # Predict next year
        predicted_spending = slope * next_year + intercept
        
        # Calculate average growth rate
        if len(spendings) > 1:
            # Calculate year-over-year growth rates
            growth_rates = []
            for i in range(1, len(spendings)):
                if spendings[i-1] > 0:
                    growth = ((spendings[i] - spendings[i-1]) / spendings[i-1]) * 100
                    growth_rates.append(growth)
            
            avg_growth_rate = np.mean(growth_rates) if growth_rates else 0
        else:
            avg_growth_rate = 0
        
        # Calculate confidence indicators
        data_points = len(historical)
        recent_trend_consistent = True
        if len(years) >= 3:
            recent_slope = (spendings[-1] - spendings[-3]) / (years[-1] - years[-3])
            overall_slope = slope
            # Check if recent trend matches overall trend (within 50%)
            recent_trend_consistent = abs(recent_slope - overall_slope) / abs(overall_slope) < 0.5 if overall_slope != 0 else True
        
        confidence = "High" if data_points >= 5 and recent_trend_consistent else "Moderate" if data_points >= 3 else "Low"
        
        result = {
            "predicted_spending": float(predicted_spending),
            "predicted_year": next_year,
            "historical_data": [{"year": int(y), "spending": float(s)} for y, s in zip(years, spendings)],
            "growth_rate": float(avg_growth_rate),
            "confidence": confidence,
            "data_points": data_points,
            "method": "Linear regression on historical spending data"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error predicting spending trends: {str(e)}"})


@tool("identify_growth_commodities")
async def identify_growth_commodities(
    top_n: int = 10,
    min_years: int = 2,
    commodity_field: str = "Commodity"
) -> str:
    """
    Identify commodities likely to grow in spending based on historical trends.
    
    Analyzes spending trends by commodity and identifies those with positive growth trajectories.
    
    Args:
        top_n: Number of top growing commodities to return (default: 10)
        min_years: Minimum number of years of data required (default: 2)
        commodity_field: The field name for commodity (default: "Commodity")
    
    Returns:
        A JSON string with commodities ranked by growth potential, including:
        - growth_rankings: List of {commodity, avg_growth_rate, total_spending, years_analyzed}
    """
    try:
        trends = await _get_commodity_spending_trends(commodity_field)
        
        if not trends:
            return json.dumps({
                "error": f"No commodity data found. Check if '{commodity_field}' field exists in the dataset.",
                "suggestion": "Use get_schema_tool to check available field names."
            })
        
        growth_rankings = []
        
        for commodity, year_data in trends.items():
            if len(year_data) < min_years:
                continue
            
            # Calculate growth rate
            spendings = [d["spending"] for d in year_data]
            years = [d["year"] for d in year_data]
            
            # Calculate compound annual growth rate (CAGR) if we have multiple years
            if len(spendings) >= 2:
                first_spending = spendings[0]
                last_spending = spendings[-1]
                num_years = years[-1] - years[0]
                
                if first_spending > 0 and num_years > 0:
                    cagr = ((last_spending / first_spending) ** (1 / num_years) - 1) * 100
                else:
                    cagr = 0
                
                # Also calculate average year-over-year growth
                yoy_growths = []
                for i in range(1, len(spendings)):
                    if spendings[i-1] > 0:
                        yoy = ((spendings[i] - spendings[i-1]) / spendings[i-1]) * 100
                        yoy_growths.append(yoy)
                
                avg_yoy_growth = np.mean(yoy_growths) if yoy_growths else 0
                total_spending = sum(spendings)
                
                growth_rankings.append({
                    "commodity": commodity,
                    "cagr": float(cagr),
                    "avg_yoy_growth": float(avg_yoy_growth),
                    "total_spending": float(total_spending),
                    "years_analyzed": len(year_data),
                    "year_range": f"{years[0]}-{years[-1]}",
                    "recent_spending": float(spendings[-1])
                })
        
        # Sort by CAGR (descending)
        growth_rankings.sort(key=lambda x: x["cagr"], reverse=True)
        
        # Return top N
        top_rankings = growth_rankings[:top_n]
        
        result = {
            "growth_rankings": top_rankings,
            "total_commodities_analyzed": len(growth_rankings),
            "criteria": {
                "min_years": min_years,
                "commodity_field": commodity_field
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error identifying growth commodities: {str(e)}"})


@tool("recommend_suppliers_by_price")
async def recommend_suppliers_by_price(
    item_description: Optional[str] = None,
    commodity: Optional[str] = None,
    top_n: int = 5,
    min_orders: int = 3
) -> str:
    """
    Recommend suppliers with historically lower unit prices for an item or commodity.
    
    Analyzes supplier pricing history and recommends those with consistently lower unit prices.
    
    Args:
        item_description: Optional item description to filter by (partial match supported)
        commodity: Optional commodity to filter by
        top_n: Number of top suppliers to return (default: 5)
        min_orders: Minimum number of orders required from a supplier (default: 3)
    
    Returns:
        A JSON string with supplier recommendations including:
        - recommendations: List of {supplier_name, avg_unit_price, total_orders, total_spending, price_rank}
    """
    try:
        collection = get_procurement_collection()
        
        # Build match stage
        match_conditions: Dict[str, Any] = {
            "Total Price": {"$exists": True},
            "Unit Price": {"$exists": True},
            "Supplier Name": {"$ne": None, "$exists": True}
        }
        
        if item_description:
            match_conditions["Item Description"] = {"$regex": item_description, "$options": "i"}
        
        if commodity:
            match_conditions["Commodity"] = commodity
        
        pipeline = [
            {
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
                    }
                }
            },
            {
                "$match": {
                    **match_conditions,
                    "UnitPriceClean": {"$gt": 0},
                    "TotalPriceClean": {"$gt": 0}
                }
            },
            {
                "$group": {
                    "_id": "$Supplier Name",
                    "avg_unit_price": {"$avg": "$UnitPriceClean"},
                    "min_unit_price": {"$min": "$UnitPriceClean"},
                    "max_unit_price": {"$max": "$UnitPriceClean"},
                    "total_orders": {"$sum": 1},
                    "total_spending": {"$sum": "$TotalPriceClean"},
                    "total_quantity": {"$sum": "$QuantityClean"}
                }
            },
            {
                "$match": {
                    "total_orders": {"$gte": min_orders}
                }
            },
            {
                "$sort": {"avg_unit_price": 1}
            },
            {
                "$limit": top_n
            }
        ]
        
        recommendations = []
        rank = 1
        async for doc in collection.aggregate(pipeline):
            recommendations.append({
                "supplier_name": doc.get("_id", "Unknown"),
                "avg_unit_price": float(doc.get("avg_unit_price", 0)),
                "min_unit_price": float(doc.get("min_unit_price", 0)),
                "max_unit_price": float(doc.get("max_unit_price", 0)),
                "total_orders": int(doc.get("total_orders", 0)),
                "total_spending": float(doc.get("total_spending", 0)),
                "total_quantity": float(doc.get("total_quantity", 0)),
                "price_rank": rank
            })
            rank += 1
        
        # Also get overall average for context
        overall_pipeline = [
            {
                "$addFields": {
                    "UnitPriceClean": {
                        "$convert": {
                            "input": "$Unit Price",
                            "to": "double",
                            "onError": 0,
                            "onNull": 0,
                        }
                    }
                }
            },
            {
                "$match": {
                    **match_conditions,
                    "UnitPriceClean": {"$gt": 0}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "overall_avg_unit_price": {"$avg": "$UnitPriceClean"}
                }
            }
        ]
        
        overall_avg = None
        async for doc in collection.aggregate(overall_pipeline):
            overall_avg = float(doc.get("overall_avg_unit_price", 0))
            break
        
        result = {
            "recommendations": recommendations,
            "overall_avg_unit_price": overall_avg,
            "filter_criteria": {
                "item_description": item_description,
                "commodity": commodity,
                "min_orders": min_orders
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error recommending suppliers: {str(e)}"})


@tool("recommend_statewide_contracts")
async def recommend_statewide_contracts(
    top_n: int = 20,
    min_volume_threshold: Optional[float] = None,
    min_departments: int = 2
) -> str:
    """
    Identify items that should be put into statewide contracts based on volume and cross-department usage.
    
    Analyzes purchase volume, number of departments using items, and total spending to identify
    items suitable for statewide contracting.
    
    Args:
        top_n: Number of top candidates to return (default: 20)
        min_volume_threshold: Minimum total quantity/volume threshold (if None, calculates from data)
        min_departments: Minimum number of departments that must use the item (default: 2)
    
    Returns:
        A JSON string with contract recommendations including:
        - recommendations: List of items with volume, department count, total spending, and contract_score
    """
    try:
        collection = get_procurement_collection()
        
        pipeline = [
            {
                "$addFields": {
                    "TotalPriceClean": {
                        "$convert": {
                            "input": "$Total Price",
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
                    }
                }
            },
            {
                "$match": {
                    "Item Description": {"$ne": None, "$exists": True},
                    "Department Name": {"$ne": None, "$exists": True},
                    "TotalPriceClean": {"$gt": 0}
                }
            },
            {
                "$group": {
                    "_id": "$Item Description",
                    "total_quantity": {"$sum": "$QuantityClean"},
                    "total_spending": {"$sum": "$TotalPriceClean"},
                    "total_orders": {"$sum": 1},
                    "departments": {"$addToSet": "$Department Name"},
                    "suppliers": {"$addToSet": "$Supplier Name"}
                }
            },
            {
                "$addFields": {
                    "department_count": {"$size": "$departments"},
                    "supplier_count": {"$size": "$suppliers"}
                }
            },
            {
                "$match": {
                    "department_count": {"$gte": min_departments}
                }
            }
        ]
        
        # Collect all items first to calculate threshold if needed
        all_items = []
        async for doc in collection.aggregate(pipeline):
            all_items.append({
                "item_description": doc.get("_id", "Unknown"),
                "total_quantity": float(doc.get("total_quantity", 0)),
                "total_spending": float(doc.get("total_spending", 0)),
                "total_orders": int(doc.get("total_orders", 0)),
                "department_count": int(doc.get("department_count", 0)),
                "supplier_count": int(doc.get("supplier_count", 0)),
                "departments": list(doc.get("departments", [])),
            })
        
        if not all_items:
            return json.dumps({
                "error": "No items found matching the criteria.",
                "suggestion": "Try lowering min_departments or check field names with get_schema_tool."
            })
        
        # Calculate threshold if not provided (use median)
        if min_volume_threshold is None:
            quantities = [item["total_quantity"] for item in all_items if item["total_quantity"] > 0]
            if quantities and len(quantities) > 0:
                min_volume_threshold = float(np.median(quantities))
            else:
                min_volume_threshold = 0.0
        
        # Filter by volume threshold
        filtered_items = [item for item in all_items if item["total_quantity"] >= min_volume_threshold]
        
        # Calculate contract score (combination of volume, spending, and department count)
        # Higher score = better candidate for statewide contract
        for item in filtered_items:
            # Normalize and combine factors
            # Score = (normalized_quantity * 0.3) + (normalized_spending * 0.4) + (normalized_dept_count * 0.3)
            quantities = [i["total_quantity"] for i in filtered_items]
            spendings = [i["total_spending"] for i in filtered_items]
            dept_counts = [i["department_count"] for i in filtered_items]
            
            max_qty = max(quantities) if quantities else 1
            max_spending = max(spendings) if spendings else 1
            max_depts = max(dept_counts) if dept_counts else 1
            
            norm_qty = item["total_quantity"] / max_qty if max_qty > 0 else 0
            norm_spending = item["total_spending"] / max_spending if max_spending > 0 else 0
            norm_depts = item["department_count"] / max_depts if max_depts > 0 else 0
            
            item["contract_score"] = float(
                (norm_qty * 0.3) + (norm_spending * 0.4) + (norm_depts * 0.3)
            )
        
        # Sort by contract score
        filtered_items.sort(key=lambda x: x["contract_score"], reverse=True)
        
        # Return top N
        top_items = filtered_items[:top_n]
        
        result = {
            "recommendations": top_items,
            "total_candidates": len(filtered_items),
            "criteria": {
                "min_volume_threshold": float(min_volume_threshold),
                "min_departments": min_departments
            },
            "scoring_method": "Combined score based on volume (30%), spending (40%), and department count (30%)"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error recommending statewide contracts: {str(e)}"})


@tool("predict_trends_by_category")
async def predict_trends_by_category(
    search_term: Optional[str] = None,
    category_field: Optional[str] = None,
    time_period: str = "Year",
    metric: str = "spending",
    top_n: Optional[int] = None,
    min_data_points: int = 2
) -> str:
    """
    Predict trends for categories (departments, suppliers, items, commodities, etc.) over time.
    
    Intelligent search that tries multiple fields automatically. If search_term is provided, it searches
    across multiple text fields. If category_field is provided, it uses that specific field.
    
    Args:
        search_term: Optional search term to find in various fields (e.g., "office supplies", "health department").
                    If provided, searches across: Item Description, Commodity, Department Name, Supplier Name.
        category_field: Optional specific field to analyze (e.g., "Department Name", "Supplier Name", "Item Description", "Commodity").
                       If not provided and search_term is provided, will try multiple fields automatically.
        time_period: Time grouping - "Year", "Quarter", or "Month" (default: "Year")
        metric: What to predict - "spending" or "quantity" or "orders" (default: "spending")
        top_n: Optional limit on number of top categories to return (if None, returns all)
        min_data_points: Minimum time periods required for prediction (default: 2)
    
    Returns:
        JSON with predictions for each category, including growth rates and forecasts, and which field(s) were used
    """
    try:
        collection = get_procurement_collection()
        
        # Define text fields to search
        text_fields = [
            "Item Description",
            "Commodity",
            "Department Name",
            "Supplier Name"
        ]
        
        # If category_field is specified, use it directly
        if category_field:
            fields_to_try = [category_field]
        elif search_term:
            # Try each field with the search term
            fields_to_try = text_fields
        else:
            # If neither specified, analyze all categories (no filter)
            fields_to_try = [None]
        
        # Build aggregation based on metric
        metric_field = "TotalPriceClean" if metric == "spending" else "QuantityClean" if metric == "quantity" else 1
        
        time_group = {
            "Year": "$Year",
            "Quarter": "$Quarter",
            "Month": "$Month"
        }.get(time_period, "$Year")
        
        all_predictions = []
        fields_used = []
        all_category_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        
        # Try each field in sequence
        for field in fields_to_try:
            if field is None:
                # Analyze all categories without filtering
                match_conditions: Dict[str, Any] = {
                    "CreationDateClean": {"$ne": None}
                }
                group_field = "$Item Description"  # Default grouping
            elif search_term:
                # Use regex to search for the term in this field
                match_conditions = {
                    field: {
                        "$regex": search_term,
                        "$options": "i"  # Case-insensitive
                    },
                    "CreationDateClean": {"$ne": None}
                }
                group_field = f"${field}"
            else:
                # Use the field directly without search term
                match_conditions = {
                    field: {"$ne": None, "$exists": True},
                    "CreationDateClean": {"$ne": None}
                }
                group_field = f"${field}"
            
            pipeline = [
                {
                    "$addFields": {
                        "TotalPriceClean": {
                            "$convert": {
                                "input": "$Total Price",
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
                },
                {
                    "$match": match_conditions
                },
                {
                    "$addFields": {
                        "Year": {"$year": "$CreationDateClean"},
                        "Month": {"$month": "$CreationDateClean"},
                        "Quarter": {"$ceil": {"$divide": [{"$month": "$CreationDateClean"}, 3]}}
                    }
                },
                {
                    "$match": {
                        "Year": {"$ne": None}
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "category": group_field,
                            "time": time_group
                        },
                        "value": {"$sum": metric_field if isinstance(metric_field, str) else metric_field},
                        "count": {"$sum": 1}
                    }
                },
                {
                    "$sort": {
                        "_id.category": 1,
                        "_id.time": 1
                    }
                }
            ]
            
            # Collect data grouped by category
            category_data: Dict[str, List[Dict[str, Any]]] = {}
            async for doc in collection.aggregate(pipeline):
                category = doc.get("_id", {}).get("category", "Unknown")
                time_val = doc.get("_id", {}).get("time")
                value = float(doc.get("value", 0))
                
                if category and time_val is not None:
                    if category not in category_data:
                        category_data[category] = []
                    category_data[category].append({"time": int(time_val), "value": value})
            
            # If we found data with this field, use it
            if category_data:
                fields_used.append(field if field else "all_fields")
                all_category_data[field if field else "all"] = category_data
                
                # If we have enough data from one field, we can break (or continue to collect from all)
                # For now, we'll collect from all fields that have data
        
        # If no data found with search_term, try exact match or broader search
        if not all_category_data and search_term:
            # Try exact match (case-insensitive)
            for field in text_fields:
                match_conditions = {
                    field: {"$regex": f"^{search_term}$", "$options": "i"},
                    "CreationDateClean": {"$ne": None}
                }
                
                # Reuse pipeline structure but with exact match
                pipeline = [
                    {
                        "$addFields": {
                            "TotalPriceClean": {
                                "$convert": {
                                    "input": "$Total Price",
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
                    },
                    {
                        "$match": match_conditions
                    },
                    {
                        "$addFields": {
                            "Year": {"$year": "$CreationDateClean"},
                            "Month": {"$month": "$CreationDateClean"},
                            "Quarter": {"$ceil": {"$divide": [{"$month": "$CreationDateClean"}, 3]}}
                        }
                    },
                    {
                        "$match": {
                            "Year": {"$ne": None}
                        }
                    },
                    {
                        "$group": {
                            "_id": {
                                "category": f"${field}",
                                "time": time_group
                            },
                            "value": {"$sum": metric_field if isinstance(metric_field, str) else metric_field},
                            "count": {"$sum": 1}
                        }
                    },
                    {
                        "$sort": {
                            "_id.category": 1,
                            "_id.time": 1
                        }
                    }
                ]
                
                category_data: Dict[str, List[Dict[str, Any]]] = {}
                async for doc in collection.aggregate(pipeline):
                    category = doc.get("_id", {}).get("category", "Unknown")
                    time_val = doc.get("_id", {}).get("time")
                    value = float(doc.get("value", 0))
                    
                    if category and time_val is not None:
                        if category not in category_data:
                            category_data[category] = []
                        category_data[category].append({"time": int(time_val), "value": value})
                
                if category_data:
                    fields_used.append(f"{field}_exact")
                    all_category_data[field] = category_data
                    break  # Found data, can stop
        
        # Process all collected category data
        for field_name, category_data in all_category_data.items():
            # Sort each category's data by time
            for cat in category_data:
                category_data[cat].sort(key=lambda x: x["time"])
            
            # Calculate predictions for each category
            for category, time_data in category_data.items():
                if len(time_data) < min_data_points:
                    continue
                
                times = [d["time"] for d in time_data]
                values = [d["value"] for d in time_data]
                
                # Linear regression
                X = np.array(times)
                y = np.array(values)
                coefficients = np.polyfit(X, y, 1)
                slope = coefficients[0]
                
                # Predict next period
                next_time = max(times) + 1
                predicted = slope * next_time + coefficients[1]
                
                # Calculate growth
                growth_rates = []
                for i in range(1, len(values)):
                    if values[i-1] > 0:
                        growth = ((values[i] - values[i-1]) / values[i-1]) * 100
                        growth_rates.append(growth)
                
                avg_growth = np.mean(growth_rates) if growth_rates else 0
                total_value = sum(values)
                
                all_predictions.append({
                    "category": category,
                    "predicted_value": float(predicted),
                    "predicted_time_period": next_time,
                    "avg_growth_rate": float(avg_growth),
                    "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                    "total_value": float(total_value),
                    "data_points": len(time_data),
                    "time_range": f"{min(times)}-{max(times)}",
                    "recent_value": float(values[-1]),
                    "matched_field": field_name
                })
        
        if not all_predictions:
            searched_fields = ", ".join(fields_to_try) if fields_to_try != [None] else "all fields"
            return json.dumps({
                "error": f"No data found matching '{search_term}' in fields: {searched_fields}" if search_term else "No data found",
                "searched_fields": fields_to_try if fields_to_try != [None] else text_fields,
                "suggestion": "Try a different search term or use get_schema_tool to check available field names."
            }, indent=2)
        
        # Sort by predicted value (descending)
        all_predictions.sort(key=lambda x: x["predicted_value"], reverse=True)
        
        if top_n:
            all_predictions = all_predictions[:top_n]
        
        return json.dumps({
            "predictions": all_predictions,
            "search_term": search_term,
            "category_field_used": category_field,
            "fields_searched": fields_used if search_term else [category_field] if category_field else ["all_fields"],
            "time_period": time_period,
            "metric": metric,
            "total_categories": len(all_predictions)
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error predicting trends by category: {str(e)}"})


@tool("identify_declining_items")
async def identify_declining_items(
    item_field: str = "Item Description",
    min_years: int = 2,
    top_n: int = 10
) -> str:
    """
    Identify items with declining usage/spending that might need attention or discontinuation.
    
    Args:
        item_field: Field to analyze (default: "Item Description")
        min_years: Minimum years of data required (default: 2)
        top_n: Number of top declining items to return (default: 10)
    
    Returns:
        JSON with items showing declining trends
    """
    try:
        collection = get_procurement_collection()
        
        pipeline = [
            {
                "$addFields": {
                    "TotalPriceClean": {
                        "$convert": {
                            "input": "$Total Price",
                            "to": "double",
                            "onError": 0,
                            "onNull": 0,
                        }
                    },
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
            },
            {
                "$match": {
                    item_field: {"$ne": None, "$exists": True},
                    "CreationDateClean": {"$ne": None}
                }
            },
            {
                "$addFields": {
                    "Year": {"$year": "$CreationDateClean"}
                }
            },
            {
                "$group": {
                    "_id": {
                        "item": f"${item_field}",
                        "year": "$Year"
                    },
                    "spending": {"$sum": "$TotalPriceClean"},
                    "orders": {"$sum": 1}
                }
            },
            {
                "$sort": {
                    "_id.item": 1,
                    "_id.year": 1
                }
            }
        ]
        
        item_data: Dict[str, List[Dict[str, Any]]] = {}
        async for doc in collection.aggregate(pipeline):
            item = doc.get("_id", {}).get("item", "Unknown")
            year = doc.get("_id", {}).get("year")
            spending = doc.get("spending", 0)
            
            if item and year:
                if item not in item_data:
                    item_data[item] = []
                item_data[item].append({"year": year, "spending": spending})
        
        # Find declining items
        declining = []
        for item, year_data in item_data.items():
            if len(year_data) < min_years:
                continue
            
            year_data.sort(key=lambda x: x["year"])
            years = [d["year"] for d in year_data]
            spendings = [d["spending"] for d in year_data]
            
            # Calculate trend
            X = np.array(years)
            y = np.array(spendings)
            if len(years) >= 2:
                slope = np.polyfit(X, y, 1)[0]
                
                # Only include declining items
                if slope < 0:
                    first_spending = spendings[0]
                    last_spending = spendings[-1]
                    decline_pct = ((first_spending - last_spending) / first_spending * 100) if first_spending > 0 else 0
                    
                    declining.append({
                        "item": item,
                        "decline_rate": float(-slope),
                        "decline_percentage": float(decline_pct),
                        "first_year_spending": float(first_spending),
                        "last_year_spending": float(last_spending),
                        "years_analyzed": len(year_data),
                        "year_range": f"{years[0]}-{years[-1]}"
                    })
        
        # Sort by decline rate
        declining.sort(key=lambda x: x["decline_rate"], reverse=True)
        declining = declining[:top_n]
        
        return json.dumps({
            "declining_items": declining,
            "total_declining": len(declining)
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error identifying declining items: {str(e)}"})


@tool("predict_seasonal_patterns")
async def predict_seasonal_patterns(
    item_field: Optional[str] = None,
    category_field: Optional[str] = None,
    min_years: int = 2
) -> str:
    """
    Identify and predict seasonal patterns in spending or usage.
    
    Analyzes spending by month/quarter to identify seasonal trends and predict future seasonal patterns.
    
    Args:
        item_field: Optional item field to filter by (e.g., "Item Description")
        category_field: Optional category to analyze (e.g., "Commodity", "Department Name")
        min_years: Minimum years of data required (default: 2)
    
    Returns:
        JSON with seasonal patterns, peak months/quarters, and predictions
    """
    try:
        collection = get_procurement_collection()
        
        match_conditions: Dict[str, Any] = {
            "CreationDateClean": {"$ne": None}
        }
        
        if item_field:
            match_conditions[item_field] = {"$ne": None, "$exists": True}
        if category_field:
            match_conditions[category_field] = {"$ne": None, "$exists": True}
        
        pipeline = [
            {
                "$addFields": {
                    "TotalPriceClean": {
                        "$convert": {
                            "input": "$Total Price",
                            "to": "double",
                            "onError": 0,
                            "onNull": 0,
                        }
                    },
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
            },
            {
                "$match": {
                    **match_conditions,
                    "TotalPriceClean": {"$gt": 0}
                }
            },
            {
                "$addFields": {
                    "Year": {"$year": "$CreationDateClean"},
                    "Month": {"$month": "$CreationDateClean"},
                    "Quarter": {"$ceil": {"$divide": [{"$month": "$CreationDateClean"}, 3]}}
                }
            },
            {
                "$match": {
                    "Year": {"$ne": None},
                    "Month": {"$ne": None, "$gte": 1, "$lte": 12},
                    "Quarter": {"$ne": None, "$gte": 1, "$lte": 4}
                }
            },
            {
                "$group": {
                    "_id": {
                        "year": "$Year",
                        "month": "$Month",
                        "quarter": "$Quarter"
                    },
                    "spending": {"$sum": "$TotalPriceClean"}
                }
            },
            {
                "$sort": {
                    "_id.year": 1,
                    "_id.month": 1
                }
            }
        ]
        
        # Aggregate by month across years
        monthly_totals: Dict[int, List[float]] = {}
        quarterly_totals: Dict[int, List[float]] = {}
        
        async for doc in collection.aggregate(pipeline):
            month = doc.get("_id", {}).get("month")
            quarter = doc.get("_id", {}).get("quarter")
            spending = doc.get("spending", 0)
            
            if month:
                if month not in monthly_totals:
                    monthly_totals[month] = []
                monthly_totals[month].append(spending)
            
            if quarter:
                if quarter not in quarterly_totals:
                    quarterly_totals[quarter] = []
                quarterly_totals[quarter].append(spending)
        
        # Calculate averages by month/quarter (handle empty lists to avoid NaN)
        monthly_avg = {
            m: float(np.mean(vals)) if len(vals) > 0 else 0.0 
            for m, vals in monthly_totals.items()
        }
        quarterly_avg = {
            q: float(np.mean(vals)) if len(vals) > 0 else 0.0 
            for q, vals in quarterly_totals.items()
        }
        
        # Find peak periods
        peak_month = max(monthly_avg.items(), key=lambda x: x[1])[0] if monthly_avg else None
        peak_quarter = max(quarterly_avg.items(), key=lambda x: x[1])[0] if quarterly_avg else None
        
        # Month names
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        return json.dumps({
            "monthly_patterns": {
                m: {
                    "average_spending": float(monthly_avg.get(m, 0)),
                    "month_name": month_names[m-1] if 1 <= m <= 12 else f"Month {m}"
                }
                for m in range(1, 13)
            },
            "quarterly_patterns": {
                q: {"average_spending": float(quarterly_avg.get(q, 0))}
                for q in range(1, 5)
            },
            "peak_month": peak_month,
            "peak_quarter": peak_quarter,
            "seasonal_variation": float(np.std(list(monthly_avg.values()))) if monthly_avg and len(monthly_avg) > 0 else 0.0
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error predicting seasonal patterns: {str(e)}"})


@tool("forecast_price_trends")
async def forecast_price_trends(
    item_description: Optional[str] = None,
    commodity: Optional[str] = None,
    forecast_periods: int = 4
) -> str:
    """
    Forecast future unit price trends for items or commodities.
    
    Analyzes historical unit prices and predicts future pricing trends.
    
    Args:
        item_description: Optional item to analyze (partial match)
        commodity: Optional commodity to analyze
        forecast_periods: Number of quarters to forecast ahead (default: 4)
    
    Returns:
        JSON with price forecasts and trend analysis
    """
    try:
        collection = get_procurement_collection()
        
        match_conditions: Dict[str, Any] = {
            "Unit Price": {"$exists": True},
            "CreationDateClean": {"$ne": None}
        }
        
        if item_description:
            match_conditions["Item Description"] = {"$regex": item_description, "$options": "i"}
        if commodity:
            match_conditions["Commodity"] = commodity
        
        pipeline = [
            {
                "$addFields": {
                    "UnitPriceClean": {
                        "$convert": {
                            "input": "$Unit Price",
                            "to": "double",
                            "onError": 0,
                            "onNull": 0,
                        }
                    },
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
            },
            {
                "$match": {
                    **match_conditions,
                    "UnitPriceClean": {"$gt": 0}
                }
            },
            {
                "$addFields": {
                    "Year": {"$year": "$CreationDateClean"},
                    "Month": {"$month": "$CreationDateClean"},
                    "Quarter": {"$ceil": {"$divide": [{"$month": "$CreationDateClean"}, 3]}}
                }
            },
            {
                "$match": {
                    "Year": {"$ne": None},
                    "Month": {"$ne": None, "$gte": 1, "$lte": 12},
                    "Quarter": {"$ne": None, "$gte": 1, "$lte": 4}
                }
            },
            {
                "$group": {
                    "_id": {
                        "year": "$Year",
                        "quarter": "$Quarter"
                    },
                    "avg_price": {"$avg": "$UnitPriceClean"},
                    "min_price": {"$min": "$UnitPriceClean"},
                    "max_price": {"$max": "$UnitPriceClean"}
                }
            },
            {
                "$sort": {
                    "_id.year": 1,
                    "_id.quarter": 1
                }
            }
        ]
        
        time_series = []
        async for doc in collection.aggregate(pipeline):
            year = doc.get("_id", {}).get("year")
            quarter = doc.get("_id", {}).get("quarter")
            avg_price = doc.get("avg_price", 0)
            
            if year and quarter:
                # Create a simple time index (year * 4 + quarter)
                time_idx = year * 4 + quarter
                time_series.append({
                    "time_index": time_idx,
                    "year": year,
                    "quarter": quarter,
                    "avg_price": avg_price
                })
        
        # If we don't have enough quarterly data, try grouping by year instead
        if len(time_series) < 2:
            # Try yearly grouping as fallback
            yearly_pipeline = [
                {
                    "$addFields": {
                        "UnitPriceClean": {
                            "$convert": {
                                "input": "$Unit Price",
                                "to": "double",
                                "onError": 0,
                                "onNull": 0,
                            }
                        },
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
                },
                {
                    "$match": {
                        **match_conditions,
                        "UnitPriceClean": {"$gt": 0},
                        "CreationDateClean": {"$ne": None}
                    }
                },
                {
                    "$addFields": {
                        "Year": {"$year": "$CreationDateClean"}
                    }
                },
                {
                    "$match": {
                        "Year": {"$ne": None}
                    }
                },
                {
                    "$group": {
                        "_id": "$Year",
                        "avg_price": {"$avg": "$UnitPriceClean"},
                        "min_price": {"$min": "$UnitPriceClean"},
                        "max_price": {"$max": "$UnitPriceClean"}
                    }
                },
                {
                    "$sort": {
                        "_id": 1
                    }
                }
            ]
            
            yearly_time_series = []
            async for doc in collection.aggregate(yearly_pipeline):
                year = doc.get("_id")
                avg_price = doc.get("avg_price", 0)
                
                if year:
                    yearly_time_series.append({
                        "time_index": year,
                        "year": year,
                        "quarter": None,
                        "avg_price": avg_price
                    })
            
            if len(yearly_time_series) >= 2:
                # Use yearly data instead
                time_series = yearly_time_series
            else:
                return json.dumps({
                    "error": f"Insufficient data for price forecasting. Found {len(time_series)} quarterly period(s) and {len(yearly_time_series)} yearly period(s). Need at least 2 time periods.",
                    "suggestion": "Try a commodity or item with more historical data across multiple time periods."
                })
        
        time_series.sort(key=lambda x: x["time_index"])
        
        # Prepare for regression
        X = np.array([t["time_index"] for t in time_series])
        y = np.array([t["avg_price"] for t in time_series])
        coefficients = np.polyfit(X, y, 1)
        
        # Check if we're using quarterly or yearly data
        is_yearly = time_series[0].get("quarter") is None
        
        # Forecast future periods
        last_time = time_series[-1]["time_index"]
        forecasts = []
        
        for i in range(1, forecast_periods + 1):
            if is_yearly:
                # For yearly data, increment by 1 year
                future_time = last_time + i
                future_year = future_time
                future_quarter = None
                predicted_price = coefficients[0] * future_time + coefficients[1]
            else:
                # For quarterly data, increment by quarters
                future_time = last_time + i
                future_year = future_time // 4
                future_quarter = ((future_time - 1) % 4) + 1
                predicted_price = coefficients[0] * future_time + coefficients[1]
            
            forecast_item = {
                "year": future_year,
                "predicted_avg_price": float(predicted_price)
            }
            if future_quarter is not None:
                forecast_item["quarter"] = future_quarter
            
            forecasts.append(forecast_item)
        
        # Calculate price trend
        price_change = ((y[-1] - y[0]) / y[0] * 100) if y[0] > 0 else 0
        
        result = {
            "historical_data": time_series,
            "forecasts": forecasts,
            "trend": "increasing" if coefficients[0] > 0 else "decreasing" if coefficients[0] < 0 else "stable",
            "historical_price_change_pct": float(price_change),
            "time_granularity": "yearly" if is_yearly else "quarterly"
        }
        
        if is_yearly:
            result["slope_per_year"] = float(coefficients[0])
        else:
            result["slope_per_quarter"] = float(coefficients[0])
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error forecasting price trends: {str(e)}"})


