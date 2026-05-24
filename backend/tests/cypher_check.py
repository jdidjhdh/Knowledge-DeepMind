import httpx, json

r = httpx.post("http://localhost:8000/api/graph/cypher", json={"query": "MATCH (n) DELETE n"})
print(f"DELETE: HTTP {r.status_code}")

r2 = httpx.post("http://localhost:8000/api/graph/cypher", json={"query": "MATCH (n) SET n.name='hack'"})
print(f"SET:    HTTP {r2.status_code}")

r3 = httpx.post("http://localhost:8000/api/graph/cypher", json={"query": "MATCH (n) RETURN n LIMIT 3"})
print(f"READ:   HTTP {r3.status_code} rows={len(r3.json()['results'])}")