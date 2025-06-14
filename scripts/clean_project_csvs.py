import os
from src.csv_validator import validate_and_convert_csv


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    for name in ["XAUUSD_M1.csv", "XAUUSD_M15.csv"]:
        path = os.path.join(base_dir, name)
        cleaned_path = path
        validate_and_convert_csv(path, cleaned_path)
        print(f"Cleaned {name} -> {cleaned_path}")


if __name__ == "__main__":
    main()
