"""
ingest.py
Extract ADA Standards of Care 2025 PDF into clean Markdown files (one per section).
Run from any directory: python src/ingest.py
"""

import fitz
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PDF = BASE_DIR / "data/raw/standards-of-care-2025.pdf"
OUT_DIR = BASE_DIR / "data/clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Define the expected section titles (from the TOC)
SECTION_TITLES = [
    "Introduction and Methodology",
    "Summary of Revisions",
    "Improving Care and Promoting Health in Populations",
    "Diagnosis and Classification of Diabetes",
    "Prevention or Delay of Diabetes and Associated Comorbidities",
    "Comprehensive Medical Evaluation and Assessment of Comorbidities",
    "Facilitating Positive Health Behaviors and Well-being to Improve Health Outcomes",
    "Glycemic Goals and Hypoglycemia",
    "Diabetes Technology",
    "Obesity and Weight Management for the Prevention and Treatment of Type 2 Diabetes",
    "Pharmacologic Approaches to Glycemic Treatment",
    "Cardiovascular Disease and Risk Management",
    "Chronic Kidney Disease and Risk Management",
    "Retinopathy, Neuropathy, and Foot Care",
    "Older Adults",
    "Children and Adolescents",
    "Management of Diabetes in Pregnancy",
    "Diabetes Care in the Hospital",
    "Diabetes Advocacy",
]

def clean_text(txt: str) -> str:
    """Basic cleaning: strip extra spaces, normalize line breaks."""
    txt = re.sub(r'\s+\n', '\n', txt)        # remove trailing spaces
    txt = re.sub(r'\n{2,}', '\n\n', txt)     # collapse multiple newlines
    txt = txt.replace("‐", "-")              # fix hyphen char
    return txt.strip()

def extract_sections(pdf_path: Path, section_titles: list[str]):
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text("text") for page in doc])

    # Normalise horizontal whitespace in the extracted text so multi-space
    # gaps from PDF column layout don't break title matching.
    full_text = re.sub(r'[^\S\n]+', ' ', full_text)

    # Find ALL occurrences of every section title (TOC + actual chapter headings).
    # Build patterns that tolerate any whitespace between words so titles like
    # "Facilitating Positive Health Behaviors and Well-being  to Improve…" match.
    all_occurrences: dict[str, list[int]] = {}
    for title in section_titles:
        words = title.split()
        pattern = r'\s+'.join(re.escape(w) for w in words)
        matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
        all_occurrences[title] = [m.start() for m in matches]
        if not matches:
            print(f"⚠️ Could not find: {title}")

    sections = {}
    for i, title in enumerate(section_titles):
        starts = all_occurrences.get(title, [])
        if not starts:
            continue

        # Collect all start positions of every subsequent section title
        all_next_starts = []
        for j in range(i + 1, len(section_titles)):
            all_next_starts.extend(all_occurrences.get(section_titles[j], []))

        # For each occurrence of this title, compute the span to the nearest
        # subsequent title occurrence. The TOC entry will produce a tiny span;
        # the actual chapter heading will produce the full chapter body.
        # We keep the longest span.
        best_text = ""
        for s in starts:
            valid_ends = sorted(p for p in all_next_starts if p > s)
            end = valid_ends[0] if valid_ends else len(full_text)
            span = full_text[s:end]
            if len(span) > len(best_text):
                best_text = span

        if best_text:
            sections[title] = clean_text(best_text)
            print(f"  ✓ {title}: {len(best_text.split()):,} words")

    return sections

def save_sections(sections: dict, out_dir: Path):
    for i, (title, text) in enumerate(sections.items(), start=1):
        # Sanitize filename
        safe_title = re.sub(r'[^a-zA-Z0-9]+', '_', title).strip("_")
        fname = f"ADA2025_{i:02d}_{safe_title}.md"
        out_path = out_dir / fname
        out_path.write_text(text, encoding="utf-8")
        print(f"✅ Saved {out_path} ({len(text.split())} words)")

if __name__ == "__main__":
    sections = extract_sections(RAW_PDF, SECTION_TITLES)
    save_sections(sections, OUT_DIR)
    print("🎉 Extraction complete!")
