import re


def sanitize_filename(name: str) -> str:
    """Entfernt Zeichen, die in Dateinamen nicht erlaubt sind."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    return name.strip().strip(".") or "activity"
