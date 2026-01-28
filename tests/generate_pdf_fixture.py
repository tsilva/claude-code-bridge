"""Generate PDF test fixture for PDF extraction tests."""

from pathlib import Path

from fpdf import FPDF


def generate_test_pdf(output_path: Path) -> None:
    """Generate a test PDF with known text content.

    Content:
        CLAUDE CODE BRIDGE
        PDF Extraction Test Document
        This document tests PDF extraction accuracy.
        Version: 2025-01
        Status: ACTIVE
    """
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "CLAUDE CODE BRIDGE", ln=True, align="C")

    # Subtitle
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, "PDF Extraction Test Document", ln=True, align="C")

    # Spacing
    pdf.ln(10)

    # Body text
    pdf.cell(0, 10, "This document tests PDF extraction accuracy.", ln=True)
    pdf.cell(0, 10, "Version: 2025-01", ln=True)
    pdf.cell(0, 10, "Status: ACTIVE", ln=True)

    pdf.output(str(output_path))


# Expected text content for validation
EXPECTED_TEXT = """CLAUDE CODE BRIDGE
PDF Extraction Test Document
This document tests PDF extraction accuracy.
Version: 2025-01
Status: ACTIVE"""


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    output_path = fixtures_dir / "pdf_test_document.pdf"
    generate_test_pdf(output_path)
    print(f"Generated: {output_path}")
