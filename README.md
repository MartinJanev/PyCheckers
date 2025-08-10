# Python Checkers (Pygame GUI + Minimax + Opening Book via draughts.js)

This is the **Option A** pipeline: we keep the Python engine/UI and generate a JSON opening book offline using **draughts.js**.

## What’s inside
- **Pygame GUI**
- **Minimax + alpha–beta**, 3 difficulties, TT in Hard
- **15+ heuristics**
- **Opening book** loader (`checkers/core/opening_book.py`) that reads `book.json` (generated from PDN by Node tool)
- **Node tool** `tools/pdn_to_book.mjs` to convert PDN → JSON FEN transitions
- **FEN support** in `CheckersState` (`to_fen` / `from_fen`)
- **Version control helpers**: `.gitignore`, `scripts/setup_git.sh`, and an optional GitHub Actions lint job

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m checkers.main
```

## Build an opening book (PDN → JSON)
```bash
# 1) Install Node deps for the converter
cd tools
npm init -y
npm i draughts
# 2) Convert PDN (first 10 plies) into book.json used by Python
node pdn_to_book.mjs ../data/your_games.pdn ../checkers/core/book.json --plies=10
```

> The book format is **FEN_current → [FEN_next, ...]**. Python maps legal moves by applying them and checking if the resulting FEN is in the set.

## Run
```bash
python -m checkers.main
```

## Notes
- Make sure your PDN is **English checkers (8×8)** compatible.
- If you change rules/variant, regenerate the book with matching PDN.
