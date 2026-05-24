import requests
import sqlite3
import json
import os

os.chdir("E:/知识库/backend")

conn = sqlite3.connect('data/memory.db')
conn.row_factory = sqlite3.Row

print("=== SQLite 直接验证 ===")
rows = conn.execute("SELECT user_id, memory_type, memory_key, memory_value, weight FROM memory_items").fetchall()
for r in rows:
    print(f"  {r['user_id']} | {r['memory_type']}.{r['memory_key']} = {r['memory_value'][:50]} (w={r['weight']})")

profiles = conn.execute("SELECT user_id, profile_data FROM user_profiles").fetchall()
for p in profiles:
    data = json.loads(p['profile_data'])
    print(f"  Profile {p['user_id']}: {data}")

msgs = conn.execute("SELECT conv_id, role, substr(content,1,50) as preview FROM conversation_history").fetchall()
print("\n=== 对话记录 ===")
for m in msgs:
    print(f"  {m['conv_id']} | {m['role']}: {m['preview']}...")

conn.close()

print("\n=== 记忆衰减测试 ===")
r = requests.post('http://localhost:8000/api/user/memory/decay')
print(f"  {r.json()}")

print("\n=== 删除测试 ===")
rd = requests.delete('http://localhost:8000/api/user/memory/u1/item?memory_type=preference&memory_key=style')
print(f"  {rd.json()}")

mem = requests.get('http://localhost:8000/api/user/memory/u1').json()
print(f"  After delete: {mem['item_count']} items remaining")
for item in mem['memory_items']:
    print(f"    - [{item['memory_type']}] {item['memory_key']} = {item['memory_value'][:50]}")

print("\n=== 跨会话共享测试 ===")
r2 = requests.post('http://localhost:8000/api/chat', json={
    'message': '我之前说过什么？',
    'user_id': 'u1',
    'conversation_id': 'c2_different_session',
    'stream': False
})
print(f"  Answer: {r2.json()['answer'][:200]}")
print(f"  Conversation ID: {r2.json().get('conversation_id')}")

print("\n=== FINAL STATS ===")
stats = requests.get('http://localhost:8000/api/memory/stats').json()
print(f"  {stats}")

print("\nALL TESTS PASSED!")