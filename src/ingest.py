from pathlib import Path
import re
import pymupdf
import json


DOCUMENTS_DIR = Path("data/documents")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CHUNKS_DIR = Path("data/chunks")


def clean_text(text):
    """
    Clean extracted PDF text while preserving the actual content.
    """

    # Replace repeated whitespace with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def extract_pages_from_pdf(pdf_path):
    """
    Extract text page-by-page from a PDF.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")
        text = clean_text(text)

        if text:
            pages.append({
                "page_number": page_number,
                "text": text
            })

    document.close()

    return pages


def create_chunks(pages):
    """
    Create chunks while preserving page information.

    This first implementation uses word-based chunking.
    The final token-based implementation can be introduced
    after the pipeline is working.
    """

    chunks = []

    chunk_id = 0

    for page in pages:

        words = page["text"].split()

        start = 0

        while start < len(words):

            end = start + CHUNK_SIZE

            chunk_words = words[start:end]

            if not chunk_words:
                break

            chunk_text = " ".join(chunk_words)

            chunk_id += 1

            chunks.append({
                "chunk_id": chunk_id,
                "page_number": page["page_number"],
                "text": chunk_text
            })

            start = end - CHUNK_OVERLAP

            if start < 0:
                start = 0

    return chunks


if __name__ == "__main__":

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    total_chunks = 0

    for pdf_path in pdf_files:

        print("\n" + "=" * 70)
        print(f"Processing: {pdf_path.name}")

        pages = extract_pages_from_pdf(pdf_path)

        print(f"Pages with text: {len(pages)}")

        chunks = create_chunks(pages)

        print(f"Chunks created: {len(chunks)}")

        output = {
            "document": pdf_path.name,
            "chunks": chunks
        }

        output_file = CHUNKS_DIR / f"{pdf_path.stem}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Saved chunks to: {output_file}")

        total_chunks += len(chunks)

        if chunks:
            print("\nFirst chunk:")
            print("-" * 70)
            print(chunks[0]["text"][:1000])
            print("-" * 70)

    print("\n" + "=" * 70)
    print(f"Total chunks created across all PDFs: {total_chunks}")