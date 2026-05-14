import io
c = io.open('app.py', 'r', encoding='utf-8', errors='ignore').read()
print(f"Triple quotes count: {c.count('\"\"\"')}")
