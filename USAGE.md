# How to Use Python Checkers with Opening Book

This guide explains how to **build the opening book**, adn **run the game**.

---

## 1. Build an Opening Book from PDN

The opening book is stored as `checkers/core/book.json`.  
We generate it from a **PDN** game collection using the Python tool in `tools/pdn_to_book.py`.

**Steps:**

```bash
# Go to the tools directory

cd tools

touch book.json  # Create an empty book file

cd ../tools # Change to the tools directory

nano pdn_to_book.py  # Create the script if it doesn't exist and edit it as needed

python -m tools.pdn_to_book data/*.pdn checkers/book/book.json --plies 8
```

**Notes:**

- Replace `../data/your_games.pdn` with the actual path to your PDN file.
- `--plies=x` means we store the first 10 half-moves; adjust as you wish.

---

## 2. Run the Game

```bash
# Install Python requirements
pip install -r requirements.txt

# Launch the Pygame GUI
python -m checkers.main
```

**In-game controls:**

- You play as **White** or **Black**, whichever you choose.
- Click your piece, then click a highlighted square to move.
- Forced captures, multi-jumps, and undo/redo are supported.
- The GUI has the ability to change difficulty (Easy / Medium / Hard).
- Press **U** to undo and **R** to redo your last move.

**AI behavior:**

- On the first several moves, if the position is in the book, the AI plays instantly.
- After leaving the book, the AI uses minimax search with alpha-beta pruning.

---

## 3. Folder Structure Overview

```
pycheckers/
├── checkers/
│   ├── core/               # Game rules, AI, heuristics, opening book loader
│   │   ├── constants.py    # Game constants
│   │   ├── engine.py       # Game engine logic
│   │   ├── heuristics.py   # Heuristic evaluation functions
│   │   ├── opening_book.py # Opening book loader
│   │   ├── move.py         # Move representation and validation
│   │   ├── state.py        # Game state management
│   │   ├── util.py         # Utility functions
│   ├── gui/                # Pygame UI
│   │   ├── pygame_app.py   # Main Pygame application
│   │   └── icon.jpg        # Application icon
│   └── main.py
|── data/                   # PDN files for training
├── tools/                  # PDN to book converter
│   ├── pdn_to_book.py      # Script to convert PDN to opening book
│   └── book.json           # Output book file - YOU CREATE THIS
├── requirements.txt
|── .gitignore
├── README.md
└── USAGE.md                 # This file
```

---

## Sample PDN

- A small example is included at `data/sample_english_checkers.pdn`.
  You can run the converter like:

```bash
cd tools
node pdn_to_book.mjs ../data/sample_english_checkers.pdn ../checkers/core/book.json --plies=8
```
