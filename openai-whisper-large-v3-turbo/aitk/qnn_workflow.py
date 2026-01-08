import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to input config file")
    parser.add_argument("--runtime", required=True, help="runtime")
    return parser.parse_args()

def main():
    args = parse_arguments()
    # Generate original model
    # Generate dataset
    # Generate quantized model

if __name__ == "__main__":
    main()
