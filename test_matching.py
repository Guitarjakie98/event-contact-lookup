"""
Rigorous tests for contact lookup and account matching functions.

Uses the real ContactDataApp2.1.parquet data. Tests the shared matching.py
module directly — no Streamlit or FastAPI required.
"""

import os
import sys

import pytest
import pandas as pd

# Ensure the project directory is on sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from matching import (
    normalize_name,
    extract_abbreviation,
    prepare_contacts,
    find_contact_matches,
    find_account_matches,
    GENERIC_DOMAINS,
)

# ---------------------------------------------------------------------------
# Load the real data once for all tests
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(PROJECT_DIR, "ContactDataApp2.1.parquet")
_contacts = prepare_contacts(pd.read_parquet(DATA_PATH))


# ===================================================================
# normalize_name
# ===================================================================

class TestNormalizeName:
    """Tests for the normalize_name utility."""

    # -- Legal suffix removal --

    def test_removes_inc(self):
        assert normalize_name("Amgen Inc") == "amgen"

    def test_removes_llc(self):
        assert normalize_name("Acme LLC") == "acme"

    def test_removes_corporation(self):
        assert normalize_name("TOYOTA MOTOR CORPORATION") == "toyota motor"

    def test_removes_ltd(self):
        assert normalize_name("Tokyo Co., Ltd.") == "tokyo"

    def test_removes_gmbh(self):
        assert normalize_name("BASF Business Services GmbH") == "basf business services"

    def test_removes_plc(self):
        assert normalize_name("Barclays PLC") == "barclays"

    def test_removes_sa(self):
        assert normalize_name("Renault SA") == "renault"

    def test_removes_multiple_suffixes(self):
        assert normalize_name("Foo Corp Inc") == "foo"

    # -- Preserves things it should --

    def test_preserves_parentheses(self):
        assert normalize_name("JACOBS(KLING STUBBINS)") == "jacobs(kling stubbins)"

    def test_preserves_digits(self):
        assert normalize_name("3M Company") == "3m company"

    # -- Word boundary protection --

    def test_costco_not_mangled(self):
        assert normalize_name("Costco") == "costco"

    def test_incognito_not_mangled(self):
        assert normalize_name("Incognito Systems") == "incognito systems"

    # -- Punctuation and whitespace --

    def test_strips_punctuation(self):
        assert normalize_name("O'Reilly & Associates!") == "oreilly associates"

    def test_collapses_whitespace(self):
        assert normalize_name("  Too   Many   Spaces  ") == "too many spaces"

    # -- Edge cases --

    def test_empty_string(self):
        assert normalize_name("") == ""

    def test_whitespace_only(self):
        assert normalize_name("   ") == ""

    def test_none(self):
        assert normalize_name(None) == ""

    def test_integer(self):
        assert normalize_name(123) == ""


# ===================================================================
# extract_abbreviation
# ===================================================================

class TestExtractAbbreviation:

    def test_extracts_from_parens(self):
        assert extract_abbreviation("JACOBS(KLING STUBBINS)") == "kling stubbins"

    def test_no_parens(self):
        assert extract_abbreviation("Securitas Danmark") == ""

    def test_none(self):
        assert extract_abbreviation(None) == ""


# ===================================================================
# find_contact_matches
# ===================================================================

class TestFindContactMatches:
    """Tests for the contact lookup function using real data."""

    # -- Exact match --

    def test_exact_match(self):
        df = find_contact_matches(_contacts, ["christoffer.moilanen@securitas.com"])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["match type"] == "Exact Match"
        assert row["match score"] == 100
        assert row["customer_name"] == "Securitas Danmark"

    def test_exact_match_case_insensitive(self):
        df = find_contact_matches(_contacts, ["Christoffer.Moilanen@Securitas.com"])
        assert len(df) == 1
        assert df.iloc[0]["match type"] == "Exact Match"

    # -- Domain fallback --

    def test_domain_match(self):
        df = find_contact_matches(_contacts, ["nonexistent.person@basf.com"])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["match type"] == "Domain Match"
        assert row["match score"] == 90

    def test_domain_match_blanks_personal_fields(self):
        df = find_contact_matches(_contacts, ["nonexistent.person@basf.com"])
        row = df.iloc[0]
        for col in ["first_name", "last_name", "job_title", "email_address"]:
            if col in df.columns:
                assert row[col] == "", f"{col} should be blank on domain match"

    # -- Personal email skip --

    @pytest.mark.parametrize("email", [
        "someone@gmail.com",
        "user@yahoo.com",
        "person@outlook.com",
        "test@hotmail.com",
        "me@icloud.com",
    ])
    def test_personal_email_skipped(self, email):
        df = find_contact_matches(_contacts, [email])
        assert len(df) == 1
        assert df.iloc[0]["match type"] == "Skipped - Personal Email"

    # -- No match --

    def test_no_match(self):
        df = find_contact_matches(_contacts, ["nobody@xyzzynonexistent99.com"])
        assert len(df) == 1
        assert df.iloc[0]["match type"] == "No Match"
        assert df.iloc[0]["match score"] == 0

    # -- Empty / edge inputs --

    def test_empty_string(self):
        df = find_contact_matches(_contacts, [""])
        assert len(df) == 1
        assert df.iloc[0]["match type"] == "No Match"

    def test_no_at_symbol(self):
        df = find_contact_matches(_contacts, ["notanemail"])
        assert len(df) == 1
        assert df.iloc[0]["match type"] == "No Match"

    # -- Multiple inputs & order preservation --

    def test_multiple_mixed_inputs(self):
        emails = [
            "christoffer.moilanen@securitas.com",  # Exact
            "someone@gmail.com",                     # Personal skip
            "nobody@xyzzynonexistent99.com",         # No match
            "fake@basf.com",                         # Domain match
        ]
        df = find_contact_matches(_contacts, emails)
        assert len(df) == 4
        assert df.iloc[0]["match type"] == "Exact Match"
        assert df.iloc[1]["match type"] == "Skipped - Personal Email"
        assert df.iloc[2]["match type"] == "No Match"
        assert df.iloc[3]["match type"] == "Domain Match"

    def test_one_row_per_input(self):
        emails = ["a@b.com", "c@d.com", "e@f.com"]
        df = find_contact_matches(_contacts, emails)
        assert len(df) == len(emails)

    def test_preserves_input_value(self):
        raw = "Christoffer.Moilanen@Securitas.com"
        df = find_contact_matches(_contacts, [raw])
        assert df.iloc[0]["input"] == raw

    # -- Output shape --

    def test_returns_dataframe(self):
        result = find_contact_matches(_contacts, ["test@test.com"])
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        df = find_contact_matches(_contacts, ["test@test.com"])
        for col in ["input", "match type", "match score"]:
            assert col in df.columns, f"Missing column: {col}"

    # -- Duplicate email in input --

    def test_duplicate_email_returns_two_rows(self):
        email = "christoffer.moilanen@securitas.com"
        df = find_contact_matches(_contacts, [email, email])
        assert len(df) == 2
        assert df.iloc[0]["match type"] == "Exact Match"
        assert df.iloc[1]["match type"] == "Exact Match"


# ===================================================================
# find_account_matches
# ===================================================================

class TestFindAccountMatches:
    """Tests for the account matching function using real data."""

    # -- Normalized exact match --

    def test_normalized_exact_match(self):
        df = find_account_matches(_contacts, ["Securitas Danmark"])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["match type"] == "Normalized Exact Match"
        assert row["match score"] == 100.0
        assert row["customer_name"] == "Securitas Danmark"

    def test_exact_match_case_insensitive(self):
        df = find_account_matches(_contacts, ["securitas danmark"])
        assert len(df) == 1
        assert df.iloc[0]["match type"] == "Normalized Exact Match"

    def test_exact_match_suffix_stripped(self):
        df = find_account_matches(_contacts, ["Amgen Inc"])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["match type"] == "Normalized Exact Match"
        assert row["match score"] == 100.0

    # -- Abbreviation match --

    def test_abbreviation_match(self):
        df = find_account_matches(_contacts, ["Kling Stubbins"])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["match type"] == "Abbreviation Match"
        assert row["match score"] == 100.0
        assert row["customer_name"] == "JACOBS(KLING STUBBINS)"

    # -- Fuzzy high confidence match --

    @pytest.mark.slow
    def test_fuzzy_high_confidence(self):
        df = find_account_matches(_contacts, ["Securitas Danmrk"])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["match type"] == "High Confidence Match"
        assert row["match score"] >= 90

    @pytest.mark.slow
    def test_fuzzy_with_extra_suffix(self):
        df = find_account_matches(_contacts, ["Securitas Danmark A/S"])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["match type"] in ("High Confidence Match", "Normalized Exact Match")
        assert row["match score"] >= 90

    # -- Possible match (70-89%) --

    @pytest.mark.slow
    def test_possible_match(self):
        # "Arrow" scores ~71% against "Arrow ECS" — below 90 but above 70
        df = find_account_matches(_contacts, ["Arrow"])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["match type"] == "Possible Match"
        assert 70 <= row["match score"] < 90
        # Account fields should be populated so the user can verify
        assert row["customer_name"] != ""

    # -- No match --

    @pytest.mark.slow
    def test_no_match(self):
        df = find_account_matches(_contacts, ["XYZZY Nonexistent Corp"])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["match type"] == "No Match"
        assert row["match score"] < 70
        assert row["customer_name"] == ""

    # -- Empty input --

    def test_empty_string(self):
        df = find_account_matches(_contacts, [""])
        assert len(df) == 1
        assert df.iloc[0]["match type"] == "No Match"
        assert df.iloc[0]["match score"] == 0

    def test_whitespace_only(self):
        df = find_account_matches(_contacts, ["   "])
        assert len(df) == 1
        assert df.iloc[0]["match type"] == "No Match"

    # -- Multiple inputs & order --

    def test_multiple_mixed_inputs(self):
        inputs = ["Securitas Danmark", "Kling Stubbins", "", "XYZZY"]
        df = find_account_matches(_contacts, inputs)
        assert len(df) == 4
        assert df.iloc[0]["match type"] == "Normalized Exact Match"
        assert df.iloc[1]["match type"] == "Abbreviation Match"
        assert df.iloc[2]["match type"] == "No Match"
        assert df.iloc[3]["match type"] == "No Match"

    def test_one_row_per_input(self):
        inputs = ["A", "B", "C"]
        df = find_account_matches(_contacts, inputs)
        assert len(df) == len(inputs)

    # -- Output shape --

    def test_returns_dataframe(self):
        result = find_account_matches(_contacts, ["test"])
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        df = find_account_matches(_contacts, ["Securitas Danmark"])
        for col in ["input", "match type", "match score", "customer_name", "customer_id"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_preserves_input_value(self):
        raw = "  Securitas Danmark  "
        df = find_account_matches(_contacts, [raw])
        assert df.iloc[0]["input"] == raw.strip()

    # -- Special characters --

    def test_special_chars_only(self):
        df = find_account_matches(_contacts, ["@#$%^"])
        assert len(df) == 1
        assert df.iloc[0]["match type"] == "No Match"
