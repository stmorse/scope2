# Conversation Planning with Monte Carlo Tree Search

Usage:

- On K8S, Dockerfile sets PYTHONPATH and you can run everything from project root as usual: `python main.py`, `python scripts/script.py`, etc.  Run from root for `config.ini` indexing
- On non-K8S, `pyproject.toml` sets it as an editable install so you can do `uv pip install -r requirements.txt` then `uv pip install -e .` then `python -m main` or `python -m scripts.script` etc