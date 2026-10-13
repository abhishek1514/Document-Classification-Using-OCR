from create_dataset import main


if __name__ == "__main__":
    print(
        "The old merger duplicated receipt rows and accepted synthetic SQL as bill data. "
        "Rebuilding only from labeled local documents instead."
    )
    main()
