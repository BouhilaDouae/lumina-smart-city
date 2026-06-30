import os

terms = ["kruskal", "bellman", "nlp", "maintenance", "cabling", "routing"]
root_dir = r"c:\Users\HP\Downloads\Lumina-Smart-Grid-GNN"

for root, dirs, files in os.walk(root_dir):
    if "venv" in root or ".git" in root or "__pycache__" in root:
        continue
    for file in files:
        if file.endswith((".py", ".md", ".txt")):
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        for term in terms:
                            if term in line.lower():
                                print(f"{file}:{i} ({term}): {line.strip()[:100]}")
            except Exception as e:
                pass
