import json

def load_json_with_comments(path: str):
    """Load JSON file ignoring comment lines starting with '#' or '//'"""
    lines = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('//'):
                continue
            lines.append(line)
    return json.loads(''.join(lines))
