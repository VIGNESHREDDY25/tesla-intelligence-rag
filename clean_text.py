import os
import re

input_folder = "data/processed"
output_folder = "data/cleaned"

os.makedirs(output_folder, exist_ok=True)

text_files = [
    "tesla_2023_10k.txt",
    "tesla_2024_10k.txt",
    "tesla_2025_10k.txt"
]


def clean_text(text):

    # Remove SEC timestamp + URL + PDF page counter
    text = re.sub(
        r"\d{2}/\d{2}/\d{4}, \d{2}:\d{2}\s*"
        r"tsla-\d+\s*"
        r"https://www\.sec\.gov/\S+\s*"
        r"\d+/\d+",
        "",
        text
    )

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Remove spaces at the beginning/end of lines
    text = "\n".join(line.strip() for line in text.splitlines())

    return text.strip()


for text_file in text_files:

    input_path = os.path.join(input_folder, text_file)

    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()

    cleaned = clean_text(text)

    output_path = os.path.join(output_folder, text_file)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(cleaned)

    print(f"Cleaned: {text_file}")