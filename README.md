# Python Checkers (PyGame GUI + Minimax + Opening Book via OCA)

This project implements English checkers (draughts) with a Python engine and a Pygame graphical interface.

The game features AI using minimax with alpha-beta pruning, multiple difficulty levels, and an opening book for move selection.

The opening book is generated offline from PDN files using a Python script and loaded in Python for improved AI play. 

The game supports FEN notation for board states and includes various helpers for version control and setup.

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m checkers.main
```

## Build an opening book (PDN → JSON)
```bash
# 1) Create data directory
mkdir data

# 2) Download PDN files into data directory
from https://www.fierz.ch/OCA_2.0.zip

# 3) execute the python script to generate the book

python -m tools.pdn_to_book data/*.pdn checkers/book/book.json --plies 8
```

**Notes:**

- Replace `../data/your_games.pdn` with the actual path to your PDN file.
- `--plies=x` means we store the first 10 half-moves; adjust as you wish.


## Run
```bash
python -m checkers.main
```

## Notes
- Make sure your PDN is **English checkers (8×8)** compatible.
- If you change rules/variant, regenerate the book with matching PDN.



