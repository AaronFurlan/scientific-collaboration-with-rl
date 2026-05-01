import json

def load_jsonl(file_path: str) -> list:
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data