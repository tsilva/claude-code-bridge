"""Pytest configuration for claudebridge tests."""

from pathlib import Path

import pytest
import httpx

from claudebridge.url_utils import resolve_bridge_url

SERVER_URL = resolve_bridge_url()
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _generate_pdf_fixture():
    """Generate PDF test fixture if it doesn't exist."""
    pdf_path = FIXTURES_DIR / "pdf_test_document.pdf"
    if pdf_path.exists():
        return

    try:
        from fpdf import FPDF
    except ImportError:
        # fpdf2 not installed, skip generation
        return

    FIXTURES_DIR.mkdir(exist_ok=True)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "CLAUDE CODE BRIDGE", ln=True, align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, "PDF Extraction Test Document", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(0, 10, "This document tests PDF extraction accuracy.", ln=True)
    pdf.cell(0, 10, "Version: 2025-01", ln=True)
    pdf.cell(0, 10, "Status: ACTIVE", ln=True)
    pdf.output(str(pdf_path))


def pytest_configure(config):
    """Register custom markers and generate fixtures."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (no server required)"
    )
    # Generate PDF fixture if needed
    _generate_pdf_fixture()


def pytest_collection_modifyitems(config, items):
    """Check server availability unless all tests are unit tests."""
    # Check if all collected tests are unit tests
    all_unit = all(item.get_closest_marker("unit") for item in items)
    if all_unit:
        return  # Skip server check for pure unit test runs

    try:
        response = httpx.get(f"{SERVER_URL}/health", timeout=5.0)
        if response.status_code != 200 or response.json().get("status") != "ok":
            pytest.exit(
                f"\n\nServer not responding correctly at {SERVER_URL}\n"
                f"Start the server with: claudebridge\n",
                returncode=1
            )
    except (httpx.ConnectError, httpx.TimeoutException):
        pytest.exit(
            f"\n\nServer not available at {SERVER_URL}\n"
            f"Start the server with: claudebridge\n",
            returncode=1
        )
