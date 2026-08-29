import chromadb


client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="enterprise_documents"
)

print("Collection:", collection.name)
print("Number of records:", collection.count())

results = collection.peek(3)

print("\nSample records:")

for i, document in enumerate(results["documents"]):
    print("\nRecord", i + 1)
    print("Document:", results["metadatas"][i]["document"])
    print("Page:", results["metadatas"][i]["page"])
    print("Text:")
    print(document[:500])