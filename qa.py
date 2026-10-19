import re

import ollama

from retrieval import KeywordRetriever, load_chunks

MODEL = "qwen3:1.7b"   # match whatever `ollama list` shows
TOP_K = 4              # how many snippets the model reads
MAX_CHARS = 1500       # per snippet, to keep the prompt small on a CPU

SYSTEM_PROMPT = """You answer questions about the FastAPI codebase using ONLY the numbered snippets provided.
- Cite every claim with its location in the form [path:start-end], copied exactly from the snippet header.
- If the snippets do not contain the answer, reply exactly: The retrieved snippets do not cover this.
- Never guess or use outside knowledge. Be concise: at most 6 sentences, plus a short code line if it helps."""

CITATION = re.compile(r"([\w./\-]+\.(?:py|md)):(\d+)(?:-(\d+))?")

def strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()

def format_snippets(hits: list) -> str:
    parts = []
    for i, (_, c) in enumerate(hits, 1):
        header = f"[{c['file']}:{c['start_line']}-{c['end_line']}]"
        parts.append(f"Snippet {i} {header} ({c['kind']}: {c['name']})\n{c['text'][:MAX_CHARS]}")
    return "\n\n".join(parts)

def generate(question: str, hits: list) -> str:
    """Ask the model to answer from the retrieved snippets only."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{format_snippets(hits)}\n\nQuestion: {question}"},
        ],
        options={"num_ctx": 4096, "temperature": 0},
        think=False,
    )
    return strip_thinking(response.message.content)

def check_citations(answer: str, hits: list) -> tuple:
    """Split the answer's citations into those matching a retrieved snippet and those that don't."""
    ranges = [(c["file"], c["start_line"], c["end_line"]) for _, c in hits]
    good, bad = [], []
    for m in CITATION.finditer(answer):
        file, start = m.group(1), int(m.group(2))
        end = int(m.group(3) or start)
        label = f"{file}:{start}-{end}"
        ok = any(file == rf and start <= r_end and end >= r_start for rf, r_start, r_end in ranges)
        if label not in good + bad:
            (good if ok else bad).append(label)
    return good, bad

def main():
    chunks = load_chunks()
    retriever = KeywordRetriever(chunks)
    print(f"Loaded {len(chunks)} chunks. Ask about the FastAPI code ('quit' to exit).")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("quit", "exit", ""):
            break
        hits = retriever.search(question, TOP_K)
        if not hits:
            print("No matching snippets found. Try different keywords.")
            continue
        print("\nRetrieved:")
        for score, c in hits:
            print(f"  {score:6.2f}  {c['file']}:{c['start_line']}-{c['end_line']}  ({c['name']})")
        print("\nGenerating answer...")
        answer = generate(question, hits)
        print("\n" + answer)
        good, bad = check_citations(answer, hits)
        print(f"\n[Citations: {len(good)} valid, {len(bad)} not matching a retrieved snippet]")
        if bad:
            print("  Unverified:", ", ".join(bad))

if __name__ == "__main__":
    main()