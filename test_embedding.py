from app.nlp.embedding import EmbeddingModel

model = EmbeddingModel(batch_size=2)
embeds = model.embed(["Developers are concerned about AI coding assistants", "Positive reaction to GitHub Copilot"])
print(f"Embedding count: {len(embeds)}")
print(f"Dimension: {len(embeds[0])}")
print(f"First embedding sample: {embeds[0][:5]}...")