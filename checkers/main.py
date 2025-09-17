from .gui.gui_app import PygameUI


def main():
    """
    Entry point for the checkers application with a graphical user interface (GUI).
    Initializes the PygameUI with a specified difficulty level and starts the main event loop.
    """
    ui = PygameUI(difficulty="Amateur")
    ui.run()


if __name__ == "__main__":
    main()
