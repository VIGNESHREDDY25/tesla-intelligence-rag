import os
import re
import json

# Where our cleaned text files are stored
input_folder = "data/cleaned"

# Where the new chunks will be stored
output_folder = "data/chunks_v3"

os.makedirs(output_folder, exist_ok=True)

# Tesla reports we want to process
text_files = [
    "tesla_2023_10k.txt",
    "tesla_2024_10k.txt",
    "tesla_2025_10k.txt"
]

# Maximum approximate size of each chunk
CHUNK_SIZE = 1500

# We are intentionally using NO overlap for this version.
OVERLAP_SENTENCES = 0


def split_long_paragraph(paragraph):
    """
    If a paragraph is larger than CHUNK_SIZE,
    split it into smaller pieces using sentences.
    """

    sentences = re.split(r'(?<=[.!?])\s+', paragraph)

    chunks = []
    current = ""

    for sentence in sentences:

        # If adding this sentence stays within the limit
        if len(current) + len(sentence) + 1 <= CHUNK_SIZE:

            current += (" " if current else "") + sentence

        else:

            # Save the current chunk
            if current:
                chunks.append(current)

            # If one sentence itself is too large,
            # split it by characters as a last resort.
            if len(sentence) > CHUNK_SIZE:

                for i in range(0, len(sentence), CHUNK_SIZE):
                    chunks.append(
                        sentence[i:i + CHUNK_SIZE]
                    )

                current = ""

            else:
                current = sentence

    # Save anything remaining
    if current:
        chunks.append(current)

    return chunks


def create_chunks(text, year, source):

    # Separate the document into pages
    pages = re.split(r'--- PAGE (\d+) ---', text)

    chunks = []
    chunk_id = 1

    # pages has this structure:
    #
    # ["",
    #  "1", "page 1 text",
    #  "2", "page 2 text",
    #  "3", "page 3 text", ...]
    #
    # Therefore we move through it two items at a time.

    for i in range(1, len(pages), 2):

        page_number = int(pages[i])
        page_text = pages[i + 1].strip()

        if not page_text:
            continue

        # Split page into paragraphs
        paragraphs = re.split(
            r'\n\s*\n',
            page_text
        )

        processed_paragraphs = []

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # If paragraph is too large,
            # split it into smaller pieces.
            if len(paragraph) > CHUNK_SIZE:

                smaller_chunks = split_long_paragraph(
                    paragraph
                )

                processed_paragraphs.extend(
                    smaller_chunks
                )

            else:

                processed_paragraphs.append(
                    paragraph
                )

        # Build chunks from the processed paragraphs
        current_paragraphs = []

        for paragraph in processed_paragraphs:

            candidate = "\n\n".join(
                current_paragraphs + [paragraph]
            )

            # If the candidate fits inside our limit,
            # keep adding paragraphs.
            if len(candidate) <= CHUNK_SIZE:

                current_paragraphs.append(
                    paragraph
                )

            else:

                # Save the current chunk first
                if current_paragraphs:

                    chunks.append({
                        "chunk_id": chunk_id,
                        "year": year,
                        "source": source,
                        "page": page_number,
                        "text": "\n\n".join(
                            current_paragraphs
                        )
                    })

                    chunk_id += 1

                # IMPORTANT:
                # No overlap.
                current_paragraphs = [paragraph]

        # Save the final chunk from this page
        if current_paragraphs:

            chunks.append({
                "chunk_id": chunk_id,
                "year": year,
                "source": source,
                "page": page_number,
                "text": "\n\n".join(
                    current_paragraphs
                )
            })

            chunk_id += 1

    return chunks


# --------------------------------------------------
# Process all Tesla reports
# --------------------------------------------------

for text_file in text_files:

    input_path = os.path.join(
        input_folder,
        text_file
    )

    # Extract year from filename
    year = re.search(
        r'20\d{2}',
        text_file
    ).group()

    # Read cleaned text
    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    # Create chunks
    chunks = create_chunks(
        text=text,
        year=year,
        source=text_file
    )

    # Output filename
    output_file = os.path.join(
        output_folder,
        text_file.replace(
            ".txt",
            "_chunks.json"
        )
    )

    # Save JSON
    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"{year}: {len(chunks)} chunks created"
    )