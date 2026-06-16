from src.models.ae import build_argparser, main

if __name__ == "__main__":
    main(build_argparser().parse_args())
