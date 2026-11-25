from pathlib import Path
from pypdf import PdfReader
from pypdf.errors import PdfReadError

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks of roughly `chunk_size` characters."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Collapse messy PDF newlines/extra spaces into single spaces for clean chunks
    clean_text = " ".join(text.split())
    if not clean_text:
        return []

    chunks = []
    start = 0
    text_len = len(clean_text)

    while start < text_len:
        # Take a window of up to `chunk_size` characters from `start`
        end = min(start + chunk_size, text_len)

        # If we aren't at the end of the text, try to end on a space so we don't cut a word in half
        if end < text_len:
            last_space = clean_text.rfind(" ", start, end)
            # Only snap back to the space if the chunk stays larger than the overlap
            if last_space > start + overlap:
                end = last_space

        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Stop once we've reached the end of the page text
        if end == text_len:
            break

        # Step forward, rewinding by `overlap` characters so context carries over
        start = end - overlap

    return chunks


def inspect_pdf(
    pdf_path: str | Path,
    chunk_size: int = 500,
    overlap: int = 100,
) -> dict:
    """Validate a PDF and return page count, chunk count, and status ('ok', 'empty_or_scanned', or 'corrupted')."""
    path = Path(pdf_path)
    if not path.exists() or path.stat().st_size == 0:
        return {"status": "corrupted", "num_pages": 0, "num_chunks": 0, "error": "File is empty (0 bytes)."}

    try:
        reader = PdfReader(path)
        num_pages = len(reader.pages)
        if num_pages == 0:
            return {"status": "empty_or_scanned", "num_pages": 0, "num_chunks": 0, "error": "PDF has 0 pages."}

        num_chunks = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            num_chunks += len(chunk_text(page_text, chunk_size=chunk_size, overlap=overlap))

        if num_chunks == 0:
            return {
                "status": "empty_or_scanned",
                "num_pages": num_pages,
                "num_chunks": 0,
                "error": "No extractable text found (likely a scanned/image-only PDF).",
            }

        return {"status": "ok", "num_pages": num_pages, "num_chunks": num_chunks, "error": ""}
    except (PdfReadError, Exception) as exc:
        return {"status": "corrupted", "num_pages": 0, "num_chunks": 0, "error": str(exc)}


def load_pdfs(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """Load every readable PDF in `data_dir` page by page and return chunk dicts."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    all_chunks = []
    pdf_files = sorted(
        p for p in data_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
    )

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                for chunk in chunk_text(page_text, chunk_size=chunk_size, overlap=overlap):
                    all_chunks.append(
                        {
                            "text": chunk,
                            "source": pdf_path.name,
                            "page": page_num,
                        }
                    )
        except (PdfReadError, Exception):
            # Skip corrupted/unreadable PDFs instead of crashing indexing for valid files
            continue

    return all_chunks
