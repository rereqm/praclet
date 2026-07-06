from src.dataset.converter import GTSDBConverter


def main():
    converter = GTSDBConverter()
    converter.run()


if __name__ == "__main__":
    main()