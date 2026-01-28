from config import MONGO_URI, MONGO_DB, MONGO_COLLECTION
from pymongo import MongoClient

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
try:
    db = client[MONGO_DB]
    coll = db[MONGO_COLLECTION]
    print("Connected to MongoDB collection", coll.full_name)
    print("Total documents:", coll.count_documents({}))
    print("Wanted-face documents:", coll.count_documents({"type": "wanted_face"}))
    print("Sample docs:")
    for d in coll.find().sort("timestamp", -1).limit(5):
        print(d)
finally:
    client.close()
