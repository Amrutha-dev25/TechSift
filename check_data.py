import chromadb

client = chromadb.PersistentClient(path='./chroma_db')
collections = client.list_collections()
for c in collections:
    if c.name == 'technology_reaction_documents':
        print(f'Collection count: {c.count()}')
        # Get all data
        all_data = c.get(include=['metadatas', 'documents'])
        print(f'Got all data: {len(all_data["ids"])} items')
        if all_data['metadatas']:
            print('First metadata:', all_data['metadatas'][0])
            print('First document:', all_data['documents'][0][:80] if all_data['documents'] else 'none')