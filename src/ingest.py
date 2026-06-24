"""
ingest.py
Extract ADA Standards of Care 2025 PDF into clean Markdown files (one per section).
Uses PDF TOC page numbers for reliable section boundaries.
Run from any directory: python src/ingest.py
"""

import fitz
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PDF = BASE_DIR / "data/raw/standards-of-care-2025.pdf"
OUT_DIR = BASE_DIR / "data/clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    txt = re.sub(r'[^\S\n]+', ' ', txt)   # collapse horizontal whitespace
    txt = re.sub(r'\n{3,}', '\n\n', txt)  # max 2 consecutive newlines
    txt = txt.replace("‐", "-").replace("‑", "-")  # fix hyphens
    return txt.strip()


def extract_sections(pdf_path: Path) -> dict[str, str]:
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()  # [(level, title, page_1indexed), ...]

    # The first 19 TOC entries map 1:1 to SECTION_TITLES.
    # Entries 20+ are Disclosures and Index — skip them.
    content_entries = [(title, page) for _, title, page in toc if page > 0][:19]

    if len(content_entries) != 19:
        print(f"⚠️  Expected 19 TOC sections, found {len(content_entries)}")

    sections: dict[str, str] = {}
    for i, (_, start_page) in enumerate(content_entries):
        title = SECTION_TITLES[i]
        # end page is the start of the next section (exclusive), or EOF
        end_page = content_entries[i + 1][1] - 1 if i + 1 < len(content_entries) else doc.page_count

        pages_text = []
        for pg in range(start_page - 1, end_page):  # fitz is 0-indexed
            pages_text.append(doc[pg].get_text("text"))

        text = clean_text("\n".join(pages_text))
        sections[title] = text
        print(f"  ✓ [{i+1:02d}] {title}: {len(text.split()):,} words  (PDF pp {start_page}–{end_page})")

    return sections


def save_sections(sections: dict, out_dir: Path) -> None:
    for i, (title, text) in enumerate(sections.items(), start=1):
        safe_title = re.sub(r'[^a-zA-Z0-9]+', '_', title).strip("_")
        fname = f"ADA2025_{i:02d}_{safe_title}.md"
        out_path = out_dir / fname
        out_path.write_text(text, encoding="utf-8")
        print(f"  ✅ Saved {fname} ({len(text.split()):,} words)")


if __name__ == "__main__":
    print("Extracting ADA 2025 Standards of Care sections…")
    sections = extract_sections(RAW_PDF)
    print("\nSaving…")
    save_sections(sections, OUT_DIR)
    print("\n🎉 Extraction complete!")
