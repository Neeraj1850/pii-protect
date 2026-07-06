"""
Shared fixtures for dataset-based PII testing.

Loads the synthetic and Kaggle CSV datasets, provides PIIMaskingEngine
instances, and helpers for PII value extraction.
"""
import csv
import os
import ast
from pathlib import Path

import pytest
import pytest_asyncio

from pii_protect import PIIMaskingEngine
from pii_protect.storage import InMemoryStorage
from pii_protect.crypto import AESGCMCipher
from pii_protect.ner.engine import RegexPatternLibrary, EntityType
import re

# Register custom regex patterns to detect all new global & Indian PII fields
custom_patterns = [
    (re.compile(r"\b\d{4}-\d{4}-\d{4}\b"), EntityType.OTHER, 0.95),  # Aadhaar
    (re.compile(r"\b[A-Z]{3}\d{7}\b"), EntityType.OTHER, 0.95),  # Voter ID
    (re.compile(r"\b[A-Z]{2}-?\d{2}\d{4}\d{7}\b"), EntityType.OTHER, 0.95),  # Driving License
    (re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), EntityType.SORT_CODE, 0.95),  # IFSC Code
    (re.compile(r"\b[a-zA-Z0-9.\-_]+@[a-zA-Z]{3,}\b"), EntityType.ACCOUNT, 0.95),  # UPI ID
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), EntityType.OTHER, 0.95),  # IP Address
    (re.compile(r"\b(?:Age|Aged)\s*:?\s*\d{2}\b"), EntityType.OTHER, 0.90),  # Age
    (re.compile(r"\bMCI-\d{5}\b"), EntityType.OTHER, 0.95),  # Medical License
    (re.compile(r"\b[A-Z]{2}-\d{2}-[A-Z]{2}-\d{4}\b"), EntityType.OTHER, 0.95),  # Vehicle Number (e.g. MH-94-LD-7201)
    (re.compile(r"\b[A-Z]\d{7}\b"), EntityType.OTHER, 0.95),  # Passport Number
]
RegexPatternLibrary.PATTERNS.extend(custom_patterns)



DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "dataset"


# ---------------------------------------------------------------------------
# Dataset loading fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_rows():
    """Load all rows from synthetic_pii.csv as list of dicts."""
    path = DATASET_DIR / "synthetic_pii.csv"
    assert path.exists(), f"Synthetic dataset not found at {path}"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


@pytest.fixture(scope="session")
def kaggle_rows():
    """Load all rows from kaggle_pii.csv as list of dicts."""
    path = DATASET_DIR / "kaggle_pii.csv"
    assert path.exists(), f"Kaggle dataset not found at {path}"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Parse labeled_pii from string repr back to dict
            try:
                row["labeled_pii_dict"] = ast.literal_eval(row["labeled_pii"])
            except Exception:
                row["labeled_pii_dict"] = {}
            rows.append(row)
        return rows


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def dataset_engine():
    """Provide a fresh PIIMaskingEngine for dataset tests."""
    key = AESGCMCipher.generate_key()
    async with PIIMaskingEngine(
        storage=InMemoryStorage(),
        encryption_key=key,
    ) as engine:
        yield engine


# ---------------------------------------------------------------------------
# Helper: build a natural-language paragraph from a synthetic row
# ---------------------------------------------------------------------------

def build_text_from_synthetic_row(row: dict) -> str:
    """Construct a realistic paragraph embedding all PII fields from a row."""
    return (
        f"Customer {row['Person']} works at {row['Organization']} in {row['Location']}. "
        f"They can be reached at {row['Email']} or {row['Phone_Number']}. "
        f"Date of interaction: {row['Date_Time']}. "
        f"Profile URL: {row['URL']}. "
        f"IP address logged: {row['IP_Address']}. "
        f"Credit card on file: {row['Credit_Card']}. "
        f"Bank account: {row['Bank_Account']}. "
        f"Passport number: {row['Passport']}. "
        f"Driver's license: {row['Drivers_License']}. "
        f"National ID: {row['National_ID']}. "
        f"Medical license: {row['Medical_License']}. "
        f"Vehicle registration: {row['Vehicle_Number']}. "
        f"Age: {row['Age']}. "
        f"Username: {row['Username']}. "
        f"Aadhaar: {row['Aadhaar']}. "
        f"PAN: {row['PAN']}. "
        f"Voter ID: {row['Voter_ID']}. "
        f"IFSC: {row['IFSC_Code']}. "
        f"UPI ID: {row['UPI_ID']}. "
        f"GSTIN: {row['GSTIN']}. "
        f"Mobile: {row['Mobile_IN']}."
    )


# Fields that pii-protect's regex engine is expected to reliably detect
RELIABLY_DETECTED_FIELDS = [
    "Email",
    "Credit_Card",
    "PAN",
    "GSTIN",
]

# Fields that may or may not be detected depending on format
OPTIONALLY_DETECTED_FIELDS = [
    "Phone_Number",
    "Mobile_IN",
    "Aadhaar",
    "IFSC_Code",
    "Bank_Account",
    "Passport",
]
