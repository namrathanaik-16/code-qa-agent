# Code Q&A Agent

A local agent that answers questions about a codebase, citing the exact file and line numbers. It runs on an open-source LLM through [Ollama](https://ollama.com), so no code leaves your machine. The project includes an **evaluation harness** so every design choice is measured, not guessed.

The first target codebase is [FastAPI](https://github.com/fastapi/fastapi) (source and English docs).

## Status

- [x] Ingest and chunk a repository (functions, classes, doc sections)
- [x] Keyword search and answers with citations
- [ ] Embeddings and hybrid search
- [ ] Evaluation set and scoring script
- [ ] Experiments (chunk size, search method, model size)
- [ ] Streamlit interface

## How it works

1. **Ingest:** download the repo and split it into chunks along natural boundaries (Python functions, classes and methods via the `ast` module; Markdown sections by heading). Each chunk records its file and line range.
2. **Retrieve:** find the few most relevant chunks for a question.
3. **Answer:** the local model answers only from those chunks and cites `file:line`.
4. **Evaluate:** score retrieval and answers against a hand-written question set.

## Index statistics

| Kind | Chunks |
|---|---|
| doc | 1201 |
| function | 216 |
| class | 116 |
| method | 241 |
| **Total** | **1774** |

## Run the ingestion

## Usage

```bash
pip install -r requirements.txt
ollama pull qwen3:1.7b
python ingest.py     # downloads FastAPI and writes data/chunks.jsonl
python qa.py         # ask questions about the code
```

`qa.py` prints the retrieved snippets first, then a cited answer, then a check that every citation matches a snippet the model was actually shown.

## License

MIT
