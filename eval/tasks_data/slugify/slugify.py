import re


def slugify(text):
    """Turn arbitrary text into a URL slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text
