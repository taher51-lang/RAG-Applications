import chromadb

client = chromadb.PersistentClient(path="./nyayasetu_db")
col = client.get_collection("landmark_judgments")
print(col.get())
