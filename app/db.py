import os
from functools import lru_cache
from typing import List, Set, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from dotenv import load_dotenv


load_dotenv()


@lru_cache(maxsize=1)
def get_mongo_client() -> AsyncIOMotorClient:
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    return AsyncIOMotorClient(uri)


def get_database() -> AsyncIOMotorDatabase:
    client = get_mongo_client()
    db_name = os.getenv("MONGODB_DB_NAME", "procurement_db")
    return client[db_name]


def get_procurement_collection() -> AsyncIOMotorCollection:
    db = get_database()
    collection_name = os.getenv("MONGODB_COLLECTION_NAME", "ca_procurements")
    return db[collection_name]


async def ping_database() -> bool:
    client = get_mongo_client()
    try:
        await client.admin.command("ping")
        return True
    except Exception:
        return False


async def get_all_field_names(limit: Optional[int] = None) -> List[str]:
    """
    Aggregate the collection to gather all distinct field names across documents.
    Note: This can be expensive on very large collections; limit can bound the scan.
    """
    collection = get_procurement_collection()
    pipeline = [
        *([{"$limit": limit}] if limit and limit > 0 else []),
        {"$project": {"k": {"$objectToArray": "$$ROOT"}}},
        {"$unwind": "$k"},
        {"$group": {"_id": None, "keys": {"$addToSet": "$k.k"}}},
    ]
    keys: Set[str] = set()
    async for doc in collection.aggregate(pipeline):
        keys.update(doc.get("keys", []))

    # Always drop Mongo's internal id
    keys.discard("_id")
    return sorted(keys)


