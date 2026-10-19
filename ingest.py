import ast
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import requests

REPO_NAME = "fastapi"
ZIP_URL = "https://github.com/fastapi/fastapi/archive/refs/heads/master.zip"
INCLUDE_PREFIXES = ("fastapi/", "docs/en/docs/")   # source code + English docs only
MAX_LINES = 80          # longest chunk before it is split into windows
MAX_FILE_BYTES = 300_000
MIN_CHARS = 40          # skip near-empty chunks

DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / f"{REPO_NAME}.zip"
REPO_DIR = DATA_DIR / REPO_NAME
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"


# ---------- Download ----------

def download_repo():
    """Download the repo zip (once) and extract only the files we need."""
    if REPO_DIR.exists():
        print(f"Using existing files in {REPO_DIR}")
        return
    DATA_DIR.mkdir(exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"Downloading {ZIP_URL} ...")
        resp = requests.get(ZIP_URL, timeout=120)
        resp.raise_for_status()
        ZIP_PATH.write_bytes(resp.content)
    print("Extracting source and English docs ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        for name in z.namelist():
            rel = name.split("/", 1)[1] if "/" in name else ""   # drop 'fastapi-master/'
            if not rel or name.endswith("/"):
                continue
            if rel.startswith(INCLUDE_PREFIXES) and rel.endswith((".py", ".md")):
                target = REPO_DIR / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))


# ---------- Chunking ----------

def make_chunk(file, lines, start, end, kind, name):
    """Build one chunk from lines[start-1:end] (line numbers are 1-based, inclusive)."""
    text = "\n".join(lines[start - 1:end])
    if len(text.strip()) < MIN_CHARS:
        return None
    return {"file": file, "start_line": start, "end_line": end,
            "kind": kind, "name": name, "text": text}

def add_span(chunks, file, lines, start, end, kind, name):
    """Add a span as one chunk, or as several windows if it is too long."""
    for s in range(start, end + 1, MAX_LINES):
        chunk = make_chunk(file, lines, s, min(s + MAX_LINES - 1, end), kind, name)
        if chunk:
            chunks.append(chunk)

def node_start(node):
    """First line of a function/class, including its decorators."""
    return min([node.lineno] + [d.lineno for d in node.decorator_list])

def chunk_python(file, text):
    lines = text.splitlines()
    chunks = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        add_span(chunks, file, lines, 1, len(lines), "code", file)
        return chunks
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_span(chunks, file, lines, node_start(node), node.end_lineno, "function", node.name)
        elif isinstance(node, ast.ClassDef):
            start, end = node_start(node), node.end_lineno
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if end - start + 1 <= MAX_LINES or not methods:
                add_span(chunks, file, lines, start, end, "class", node.name)
                continue
            # big class: one chunk for the header, then one per method
            add_span(chunks, file, lines, start, node_start(methods[0]) - 1, "class", node.name)
            for m in methods:
                add_span(chunks, file, lines, node_start(m), m.end_lineno, "method", f"{node.name}.{m.name}")
    return chunks

def chunk_markdown(file, text):
    lines = text.splitlines()
    chunks, sections = [], []
    start, title, in_code = 1, Path(file).stem, False
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"#{1,4}\s", line):
            if i > start:
                sections.append((start, i - 1, title))
            start, title = i, line.lstrip("#").strip()
    sections.append((start, len(lines), title))
    for s, e, t in sections:
        add_span(chunks, file, lines, s, e, "doc", t)
    return chunks


# ---------- Main ----------

def is_noise(path):
    """Skip doc pages that would pollute search: internal test pages snd release notes."""
    return path.suffix == ".md" and (path.name.startswith("_") or path.name == "release-notes.md")

def main():
    download_repo()
    chunks = []
    for path in sorted(REPO_DIR.rglob("*")):
        if path.suffix not in (".py",".md") or path.stat().st_size > MAX_FILE_BYTES or is_noise(path):
            continue
        rel = path.relative_to(REPO_DIR).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks += chunk_python(rel,text) if path.suffix == ".py" else chunk_markdown(rel, text)
        
    for n, c in enumerate(chunks):
        c["id"] = f"c{n:05d}"
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            
    kinds = Counter(c["kind"] for c in chunks)
    avg = sum(c["end_line"] - c["start_line"] + 1 for c in chunks)/len(chunks)
    print(f"\nWrote {len(chunks)} chunks to {CHUNKS_PATH}")
    print("By kind:", dict(kinds))
    print(f"Average chunk length : {avg:.0f} lines")
    
if __name__ == "__main__":
    main()