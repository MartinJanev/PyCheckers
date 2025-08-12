# PyCheckers

### Adversarial Game - English Checkers (Draughts) using Python and Pygame

This project is a Python implementation of the classic game of English Checkers (Draughts) using Pygame for the
graphical interface.

It features an AI opponent that can play at various difficulty levels, leveraging an opening book for optimal moves in
the early game.
The way the AI plays is based on the minimax algorithm with alpha-beta pruning, allowing it to make strategic decisions
based on the current board state.
The game supports the standard 8x8 board and follows the rules of English Checkers, including forced captures and
multi-jumps.

The project includes a tool for building an opening book from PDN files - collections of checkers games.
The book is stored in JSON format and is used to provide the AI with optimal opening moves.
The game supports FEN notation for board states and includes various helpers for version control and setup.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate   # On Windows 
pip install -r requirements.txt # to install dependencies
```

**In-game controls:**

- You play as **White** or **Black**, whichever you choose.
- Click your piece, then click a highlighted square to move.
- Forced captures, multi-jumps, and undo/redo are supported.
- The GUI has the ability to change difficulty (Easy / Medium / Hard).
- Press **CTRL + Z** to undo and **CTRL + Y** to redo your last move.
- It saves the game state at the end of the game.

**AI behavior:**

- On the first several moves, if the position is in the book, the AI plays instantly.
- After leaving the book, the AI uses minimax search with alpha-beta pruning.

---

## 3. Folder Structure Overview

```
pycheckers/
├── checkers/
│   ├── core/                   #  rules/AI/heuristics/PDN
│   │   ├── constants.py        # game constants
│   │   ├── engine.py           # game engine
│   │   ├── heuristics.py       #  heuristics for AI
│   │   ├── move.py         [pygame_app.py](checkers/gui/pygame_app.py)    # move logic
│   │   ├── opening_book.py     # AI opening book
│   │   ├── pdn.py              # PDN parsing and handling
│   │   ├── state.py            # game state management
│   │   └── util.py             # utility functions
│   ├── gui/                    
│   │   ├── pygame_app.py
│   │   └── icon.jpg
│   └── main.py
├── tools/
│   ├── pdn_to_book.py          # tool to convert PDN files to opening book JSON - you can use this to build your own book
│   └── book.json
├── requirements.txt            # (your existing)
├── .gitignore
└── README.md

```

### Build an opening book (PDN → JSON)

If you want to build an opening book from PDN files, follow these steps:

```bash
# 1) Create data directory
mkdir data

# 2) Download PDN files into data directory (example: OCA 2.0)
# You can download PDN files from Internet sources that provide collections of checkers games.
# For example, you can download the OCA 2.0 PDN files and extract them into the `data` directory.
# Download link: https://www.fierz.ch/OCA

# 3) execute the python script to generate the book
python -m tools.pdn_to_book data/your_games.pdn checkers/book/book.json --plies 8
```

**Notes:**

- Replace `../data/your_games.pdn` with the actual path to your PDN file.
- `--plies=x` means we store the first x half-moves; adjust as you wish.

## Run

To run the game, execute the following command:

```bash
# to run the game
python -m checkers.main 
```

## Notes

- Make sure your PDN is **English checkers (8×8)** compatible.



