import re

SHORT_TOKEN_WHITELIST = {
    "up",        # warm up
    "db",        # dumbbell
    "bb",        # barbell
    "ez",        # ez-bar
    "kg",
    "lb",
    "rdl",
    "ohp",
    "bw",
    "ab",        # ab wheel
    "jm",        # jm press
    "t",         # t-bar (as in t-bar row)
}

BAD_CHARS_RE = re.compile(r"[|`'\"=@°]")
DROP_LINE_RE = re.compile(r"\b(google|apple|download|cohevy|created)\b", re.I)
WORD_RE = re.compile(r"[A-Za-z]")

def normalize_token(tok: str) -> str:
    # Remove leading/trailing punctuation (OCR noise)
    tok = tok.strip("‘’'\"`.,:;()[]{}")
    return tok

def clean_text(text: str) -> str:
    # 1. Remove specific unwanted characters
    text = BAD_CHARS_RE.sub("", text)

    # 2. Replace non-breaking spaces and weird Unicode with normal spaces
    text = text.replace("\xa0", " ")

    cleaned_lines = []
    for raw_line in text.splitlines():
        # Strip once for checks
        line = raw_line.strip()

        # DROP entire line if it contains store / UI noise
        if DROP_LINE_RE.search(line):
            continue

        if not line:
            continue

        tokens = line.split()
        filtered = []

        for tok in tokens:
            tok = normalize_token(tok)
            if not tok:
                continue

            tok_l = tok.lower()

            # Keep numbers
            if tok.isdigit():
                filtered.append(tok)
                continue

            # Keep rep ranges
            if re.fullmatch(r"\d+[-–]\d+", tok):
                filtered.append(tok)
                continue

            # Keep whitelisted short tokens
            if tok_l in SHORT_TOKEN_WHITELIST:
                filtered.append(tok)
                continue

            # Keep alphabetic tokens >= 3 chars
            if len(tok) >= 3 and WORD_RE.search(tok):
                filtered.append(tok)
                continue

            # Otherwise drop token (1–2 char junk)

        new_line = " ".join(filtered).strip()
        if new_line:
            cleaned_lines.append(new_line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = cleaned_text.strip()

    return cleaned_text