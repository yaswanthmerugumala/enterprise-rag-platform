import requests

queries = [
    "What encryption standard is required?",
    "What are the KPIs of the AI strategy?",
    "What cloud services are used?"
]

for q in queries:
    response = requests.post(
        "http://127.0.0.1:8000/chat",
        json={"query": q}
    )

    print("\nQuery:", q)
    print("Answer:", response.json()["answer"])
