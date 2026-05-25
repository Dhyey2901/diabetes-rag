"""
ingest.py
Extract ADA Standards of Care 2025 PDF into clean Markdown files (one per section).
"""

import fitz
from pathlib import Path
import re


# Paths
RAW_PDF = Path("data/raw/standards-of-care-2025.pdf")
OUT_DIR = Path("data/clean/")
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

    sections = {}
    for i, title in enumerate(section_titles):
        # Escape special regex chars
        title_pattern = re.escape(title)
        # Find start
        start_match = re.search(title_pattern, full_text, re.IGNORECASE)
        if not start_match:
            print(f"⚠️ Could not find: {title}")
            continue
        start_idx = start_match.start()

        # End = start of next section (or end of doc)
        if i + 1 < len(section_titles):
            next_title = section_titles[i + 1]
            next_match = re.search(re.escape(next_title), full_text, re.IGNORECASE)
            end_idx = next_match.start() if next_match else len(full_text)
        else:
            end_idx = len(full_text)

        section_text = full_text[start_idx:end_idx]
        sections[title] = clean_text(section_text)

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
