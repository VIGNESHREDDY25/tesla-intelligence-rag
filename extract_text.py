import pymupdf
import os

data_folder = "data/raw"
output_folder = "data/processed"

pdf_files = [
    "tesla_2023_10k.pdf",
    "tesla_2024_10k.pdf",
    "tesla_2025_10k.pdf"
]

for pdf_file in pdf_files:

    pdf_path = os.path.join(data_folder, pdf_file)

    document = pymupdf.open(pdf_path)

    all_text = ""

    for page_number, page in enumerate(document):

        text = page.get_text()

        all_text += f"\n\n--- PAGE {page_number + 1} ---\n\n"
        all_text += text

    output_filename = pdf_file.replace(".pdf", ".txt")
    output_path = os.path.join(output_folder, output_filename)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(all_text)

    print(f"Saved: {output_path}")