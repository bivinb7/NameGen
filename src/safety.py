import re

DEFAULT_BLOCKLIST = {"fuck", "shit", "bitch", "asshole"}

def is_valid_name(name, blocklist=None):
    blocklist = DEFAULT_BLOCKLIST if blocklist is None else blocklist
    candidate = re.sub(r"\\s+", " ", str(name).strip().lower())
    if not 3 <= len(candidate) <= 30:
        return False
    if not re.fullmatch(r"[a-z]+(?: [a-z]+)*", candidate):
        return False
    if any(term in candidate for term in blocklist):
        return False
    return True
