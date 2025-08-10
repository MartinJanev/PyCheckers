from .gui.pygame_app import PygameUI


def main():
    ui = PygameUI(difficulty="Amateur")
    ui.run()


if __name__ == "__main__":
    main()
