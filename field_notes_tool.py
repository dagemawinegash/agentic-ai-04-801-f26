# field_notes_tool.py, Week 3: retrieval is just another tool (Chroma)
# Core idea: RAG is not magic. It is a tool that searches a local vector
# store instead of calling a remote API. Same @tool contract as weather/soil.

import chromadb

# ---- Synthetic field notes (Challenge B works with data like this) ----------

NOTES = [
    "Field K3, 12 May: maize leaves turning yellow at the edges, "
    "mostly on the lower leaves. No pests visible.",
    "Field K3, 19 May: yellowing has spread upward. Neighbour suggests "
    "nitrogen deficiency. Soil was not fertilized this season.",
    "Field N1, 12 May: heavy rain last night, standing water between "
    "ridges. Some seedlings washed out on the slope side.",
    "Field N1, 26 May: replanted washed-out section. Growth uneven.",
    "Field K7, 02 June: small holes in young leaves, larvae found in "
    "two whorls. Suspect fall armyworm, applied ash mixture.",
    "Field K7, 09 June: larvae count lower after ash treatment, but "
    "damage visible on about a quarter of plants.",
]

client = chromadb.Client()                      # in-memory for the lab
collection = client.create_collection("field_notes")
collection.add(
    documents=NOTES,
    ids=[f"note_{i}" for i in range(len(NOTES))],
)

# ---- The retrieval function -------------------------------------------------

def search_notes(query: str, n: int = 2) -> str:
    """Find the field notes most relevant to a question."""
    hits = collection.query(query_texts=[query], n_results=n)
    return "\n---\n".join(hits["documents"][0])

# ---- Quick test: semantic, not keyword --------------------------------------

if __name__ == "__main__":
    for q in ["Which field has a pest problem?",
              "Where did flooding damage the crop?",
              "chlorosis symptoms"]:
        print(f"\nQUERY: {q}")
        print(search_notes(q))
