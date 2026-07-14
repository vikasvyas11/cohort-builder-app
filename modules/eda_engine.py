# =============================================================================
# modules/eda_engine.py
# PURPOSE: Automated EDA and data cleaning pipeline for user-uploaded datasets.
#          All functions return the modified DataFrame plus a log entry so the
#          Streamlit page can display a clear before/after report.
# =============================================================================

import re
import string
from typing import Optional

import numpy as np
import pandas as pd

# =============================================================================
# ── FIELD TYPE KEYWORD MAPS ───────────────────────────────────────────────────
# Used to infer the semantic type of a column from its cleaned name.
# Each set contains common column-name patterns for that type of field.
# =============================================================================

_FIRST_NAME = {'first_name','forename','given_name','fname','christian_name',
               'firstname','first','given','first_initial','nome','prenom'}
_SURNAME    = {'surname','last_name','lastname','family_name','lname',
               'last','family','second_name','cognome'}
_FULL_NAME  = {'name','full_name','fullname','whole_name','complete_name',
               'first_and_surname','both_names','full_nm'}
_DOB        = {'dob','date_of_birth','birth_date','birthdate','birth_day',
               'birthday','born','date_birth','birth','dob_clean'}
_OTHER_DATE = {'date','event_date','death_date','died','registration_date',
               'reg_date','entry_date','record_date','death','death_yr'}
_GENDER     = {'gender','sex','gender_code','sex_code','gender_mapped',
               'gender_assigned','sex_at_birth'}
_LOCATION   = {'city','town','county','district','borough','area',
               'birth_place','birthplace','place_of_birth','place',
               'location','region','state','country','nation'}
_POSTCODE   = {'postcode','post_code','zip','zip_code','postal_code',
               'pstcde','postcode_fake','postal'}
_EMAIL      = {'email','email_address','email_addr','e_mail','mail'}
_ID         = {'id','unique_id','record_id','person_id','uid','identifier',
               'patient_id','subject_id','row_id','index','ref','reference',
               'key','pid','rid','source_dataset','cluster'}


# =============================================================================
# ── STEP 1: CLEAN FIELD NAMES ─────────────────────────────────────────────────
# =============================================================================

def clean_field_names(df: pd.DataFrame) -> tuple:
    """Standardise all column names to lowercase_underscore_separated.

    Rules applied in order:
      1. Strip leading/trailing whitespace
      2. Lowercase everything
      3. Replace spaces and hyphens with underscores
      4. Remove characters that are not alphanumeric or underscores
      5. Remove trailing _1, _2 (lagging numeric suffixes from deduplication)
      6. Collapse multiple consecutive underscores to one
      7. Strip leading/trailing underscores

    Returns (cleaned_df, mapping_dict) where mapping_dict maps
    original_name → clean_name for display in the EDA report.
    """
    mapping = {}
    for col in df.columns:
        clean = col.strip()                           # Strip whitespace
        clean = clean.lower()                         # Lowercase
        clean = re.sub(r'[\s\-]+', '_', clean)       # Spaces/hyphens → _
        clean = re.sub(r'[^\w]', '', clean)           # Remove special chars
        clean = re.sub(r'_\d+$', '', clean)           # Remove trailing _1, _2 etc.
        clean = re.sub(r'_+', '_', clean)             # Collapse multiple __
        clean = clean.strip('_')                      # Strip leading/trailing _
        mapping[col] = clean if clean else col        # Never produce empty name
    df_clean = df.rename(columns=mapping)
    return df_clean, mapping


# =============================================================================
# ── STEP 2: DETECT FIELD TYPES ────────────────────────────────────────────────
# =============================================================================

def detect_field_types(df: pd.DataFrame) -> dict:
    """Infer the semantic type of each column from its name.

    Returns a dict mapping column_name → type_string where type_string is one of:
      'first_name'  – given name, forename
      'surname'     – family name, last name
      'full_name'   – combined first+last name
      'dob'         – date of birth
      'date'        – other date field
      'gender'      – gender or sex field
      'location'    – city, town, county etc.
      'postcode'    – postcode / zip code
      'email'       – email address
      'id'          – unique identifier (excluded from comparisons and blocking)
      'unknown'     – anything that doesn't match a known pattern
    """
    types = {}
    for col in df.columns:
        c = col.lower().strip()   # Normalise for matching (already clean but be safe)
        if c in _ID:
            types[col] = 'id'
        elif c in _FIRST_NAME:
            types[col] = 'first_name'
        elif c in _SURNAME:
            types[col] = 'surname'
        elif c in _FULL_NAME:
            types[col] = 'full_name'
        elif c in _DOB:
            types[col] = 'dob'
        elif c in _OTHER_DATE:
            types[col] = 'date'
        elif c in _GENDER:
            types[col] = 'gender'
        elif c in _LOCATION:
            types[col] = 'location'
        elif c in _POSTCODE:
            types[col] = 'postcode'
        elif c in _EMAIL:
            types[col] = 'email'
        else:
            types[col] = 'unknown'
    return types


# =============================================================================
# ── STEP 3: REMOVE FULLY-NULL COLUMNS ─────────────────────────────────────────
# =============================================================================

def remove_null_columns(df: pd.DataFrame) -> tuple:
    """Drop columns where 100% of values are null.

    Returns (cleaned_df, list_of_dropped_column_names).
    """
    fully_null = [c for c in df.columns if df[c].isna().all()]  # All values null
    return df.drop(columns=fully_null), fully_null


# =============================================================================
# ── STEPS 4-6: REMOVE RECORDS WITH EXCESS NULL VALUES ─────────────────────────
# =============================================================================

def remove_null_rows(df: pd.DataFrame) -> tuple:
    """Remove rows with 100%, n-1, and n-2 null values.

    Three passes applied in sequence:
      Pass 1: Remove rows where ALL n columns are null (100%)
      Pass 2: Remove rows with (n-1) nulls – only 1 column has a value
      Pass 3: Remove rows with (n-2) nulls – only 2 columns have values

    Returns (cleaned_df, counts_dict) where counts_dict maps pass → n_removed.
    """
    n = df.shape[1]                    # Total number of columns
    counts = {}

    # Pass 1: 100% null rows
    before = len(df)
    null_counts = df.isna().sum(axis=1)              # Null count per row
    df = df[null_counts < n].copy()                  # Keep rows with at least 1 non-null
    counts['100%_null'] = before - len(df)

    # Pass 2: n-1 null (only 1 value present)
    before = len(df)
    null_counts = df.isna().sum(axis=1)
    df = df[null_counts < n - 1].copy()              # Keep rows with at least 2 non-nulls
    counts['n-1_null'] = before - len(df)

    # Pass 3: n-2 null (only 2 values present)
    before = len(df)
    null_counts = df.isna().sum(axis=1)
    df = df[null_counts < n - 2].copy()              # Keep rows with at least 3 non-nulls
    counts['n-2_null'] = before - len(df)

    return df, counts


# =============================================================================
# ── STEP 7: CLEAN TEXT VALUES ─────────────────────────────────────────────────
# =============================================================================

def clean_text_values(df: pd.DataFrame, field_types: dict) -> pd.DataFrame:
    """Standardise text content of string columns.

    Rules:
      - All string columns: strip leading/trailing whitespace
      - Name fields (first_name, surname, full_name): Title Case
      - All other text fields: lowercase
    """
    df = df.copy()
    name_types = {'first_name', 'surname', 'full_name'}  # Fields to title-case

    for col in df.columns:
        if df[col].dtype == object:                       # Only process string columns
            ftype = field_types.get(col, 'unknown')
            df[col] = df[col].astype(str).str.strip()    # Strip whitespace from all
            df[col] = df[col].replace('nan', np.nan)     # Re-null any stringified NaNs
            if ftype in name_types:
                df[col] = df[col].str.title()            # Proper Case for names
            elif ftype not in ('id', 'email'):            # Don't alter IDs or emails
                df[col] = df[col].str.lower()            # lowercase for other text
    return df


# =============================================================================
# ── STEP 8: REMOVE DUPLICATE ROWS ─────────────────────────────────────────────
# =============================================================================

def remove_exact_duplicates(df: pd.DataFrame, id_col: Optional[str] = None) -> tuple:
    """Remove rows that are exact duplicates across all columns except the ID column.

    If id_col is provided, deduplication ignores the ID column (so two records
    with different IDs but identical data are still considered duplicates).

    Returns (cleaned_df, n_removed).
    """
    before = len(df)
    cols_to_check = [c for c in df.columns if c != id_col]   # Exclude ID from comparison
    df = df.drop_duplicates(subset=cols_to_check).copy()
    return df, before - len(df)


# =============================================================================
# ── STEP 9: STANDARDISE DATES ─────────────────────────────────────────────────
# =============================================================================

def standardise_dates(df: pd.DataFrame, date_cols: list) -> tuple:
    """Parse dates in various formats and standardise to YYYY-MM-DD.

    Tries a list of common formats; picks the one that successfully parses
    the most values. Falls back to pandas auto-inference.

    Returns (cleaned_df, dict mapping col → format_used).
    """
    FORMATS = [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y',
        '%Y%m%d',   '%d.%m.%Y', '%d %b %Y', '%d %B %Y',
        '%Y/%m/%d', '%Y-%m-%d %H:%M:%S',
    ]
    df       = df.copy()
    used_fmt = {}

    for col in date_cols:
        if col not in df.columns:
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue

        best_parsed = None
        best_count  = 0
        best_fmt    = 'auto'

        # Try each explicit format first
        for fmt in FORMATS:
            try:
                parsed = pd.to_datetime(df[col], format=fmt, errors='coerce')
                count  = parsed.notna().sum()
                if count > best_count:
                    best_count  = count
                    best_parsed = parsed
                    best_fmt    = fmt
            except Exception:
                continue

        # Fall back to pandas auto-inference if nothing worked well
        if best_count == 0 or best_count < 0.5 * non_null.notna().sum():
            try:
                auto_parsed = pd.to_datetime(df[col], infer_datetime_format=True,
                                             errors='coerce')
                if auto_parsed.notna().sum() > best_count:
                    best_parsed = auto_parsed
                    best_fmt    = 'auto'
            except Exception:
                pass

        if best_parsed is not None and best_parsed.notna().sum() > 0:
            df[col]      = best_parsed.dt.strftime('%Y-%m-%d')   # Standardise format
            used_fmt[col] = best_fmt

    return df, used_fmt


# =============================================================================
# ── STEP 10: CORRELATION CHECK ────────────────────────────────────────────────
# =============================================================================

def find_high_correlation_pairs(
    df:          pd.DataFrame,
    id_cols:     list,
    threshold:   float = 0.95,
) -> list:
    """Find pairs of non-ID columns with very high similarity.

    For text/categorical columns: measures the fraction of rows where
    both columns share the same value (value-level co-occurrence).
    For numeric columns: uses absolute Pearson correlation.

    ID columns (unique_id, cluster, source_dataset etc.) are excluded
    because they are identifiers and not semantic features.

    Returns a list of (col_a, col_b, score) tuples sorted by score descending.
    Requires at least 10 non-null observations in both columns to compare.
    """
    candidates = [c for c in df.columns if c not in id_cols]  # Exclude ID columns
    pairs      = []

    for i, col_a in enumerate(candidates):
        for col_b in candidates[i + 1:]:               # Avoid double-counting
            try:
                both_present = df[col_a].notna() & df[col_b].notna()
                if both_present.sum() < 10:             # Not enough data to compare
                    continue
                a = df.loc[both_present, col_a]
                b = df.loc[both_present, col_b]

                if (pd.api.types.is_numeric_dtype(a) and
                        pd.api.types.is_numeric_dtype(b)):
                    # Numeric: Pearson correlation
                    score = abs(float(a.corr(b)))
                else:
                    # Text/categorical: fraction of rows with matching value
                    score = float((a.astype(str) == b.astype(str)).mean())

                if score >= threshold:
                    pairs.append((col_a, col_b, round(score, 4)))
            except Exception:
                continue

    return sorted(pairs, key=lambda x: x[2], reverse=True)   # Highest correlation first


# =============================================================================
# ── COMPARISON TYPE SUGGESTIONS ───────────────────────────────────────────────
# =============================================================================

def suggest_comparison_types(field_types: dict) -> dict:
    """Map each non-ID field to a suggested Splink comparison type.

    Returns dict: column_name → comparison_type_string where type is one of:
      'NameComparison'       – for name fields (uses Jaro-Winkler fuzzy matching)
      'DateOfBirthComparison' – for DOB fields (handles transpositions, date ranges)
      'ExactMatch'           – for categorical fields (gender, city, postcode, email)
    """
    TYPE_MAP = {
        'first_name': 'NameComparison',
        'surname':    'NameComparison',
        'full_name':  'NameComparison',
        'dob':        'DateOfBirthComparison',
        'date':       'ExactMatch',        # Other dates: no fuzzy matching by default
        'gender':     'ExactMatch',
        'location':   'ExactMatch',
        'postcode':   'ExactMatch',
        'email':      'ExactMatch',
        'unknown':    'ExactMatch',        # Safest default for unrecognised fields
    }
    suggestions = {}
    for col, ftype in field_types.items():
        if ftype == 'id':
            continue                        # IDs excluded from comparisons
        suggestions[col] = TYPE_MAP.get(ftype, 'ExactMatch')
    return suggestions


def suggest_blocking_rules(field_types: dict) -> dict:
    """Suggest which fields to use as blocking rules based on their type.

    Name, DOB, postcode, and email fields are good blocking candidates.
    Gender is not useful alone (too few values; most pairs agree by chance).
    Location (city) can be useful but may be noisy.

    Returns dict: column_name → bool (True = suggested for blocking).
    """
    BLOCK_RECOMMENDED = {'first_name', 'surname', 'full_name', 'dob', 'postcode', 'email'}
    BLOCK_OPTIONAL    = {'location', 'date'}
    result = {}
    for col, ftype in field_types.items():
        if ftype == 'id':
            continue
        result[col] = ftype in BLOCK_RECOMMENDED    # Recommend high-signal fields
    return result


# =============================================================================
# ── ERROR INTRODUCTION (for derived test dataset) ─────────────────────────────
# =============================================================================

def introduce_errors_for_sample(
    df:          pd.DataFrame,
    field_types: dict,
    sample_frac: float = 0.3,
    seed:        int   = 42,
) -> pd.DataFrame:
    """Create an error-introducing sample from df for testing linkage.

    Applies the same error rates as data_builder.py's fakeb construction,
    but generalised to work with any dataset based on detected field types.

    Error rates applied:
      Name fields    : 14% random character substitution (typos)
      DOB fields     : 5% set to null (missing date)
      Email fields   : 15% variation (remove dots, append number, swap domain)
      Location fields: 11% abbreviate to first letters of each word
      Gender fields  : 7% flip (M→F or F→M)

    The sample_frac fraction of df is sampled, then errors are introduced.
    Returns the error-introducing sample with a new unique_id suffix '_B'.
    """
    import random
    rng = np.random.default_rng(seed)
    random.seed(seed)

    sample = df.sample(frac=sample_frac, random_state=seed).copy()

    for col, ftype in field_types.items():
        if col not in sample.columns:
            continue

        if ftype in ('first_name', 'surname', 'full_name'):
            # 14% random character typo
            mask = rng.random(len(sample)) < 0.14
            sample.loc[mask, col] = sample.loc[mask, col].apply(_typo)

        elif ftype == 'dob':
            # 5% missing DOB
            mask = rng.random(len(sample)) < 0.05
            sample.loc[mask, col] = np.nan

        elif ftype == 'email':
            # 15% email variation
            mask = rng.random(len(sample)) < 0.15
            sample.loc[mask, col] = sample.loc[mask, col].apply(_email_var)

        elif ftype == 'location':
            # 11% city abbreviation
            mask = rng.random(len(sample)) < 0.11
            sample.loc[mask, col] = sample.loc[mask, col].apply(_abbreviate)

        elif ftype == 'gender':
            # 7% gender flip
            mask = rng.random(len(sample)) < 0.07
            sample.loc[mask, col] = sample.loc[mask, col].apply(_flip_gender)

    # Suffix the unique_id column to create distinct IDs for dataset B
    id_cols = [c for c, t in field_types.items() if t == 'id' and c in sample.columns]
    for id_col in id_cols:
        sample[id_col] = sample[id_col].astype(str) + '_B'

    # Tag as dataset B
    sample['source_dataset'] = 'B'
    return sample


def _typo(val: str) -> str:
    """Replace one random character with a random lowercase letter."""
    import random, string
    if pd.isna(val) or len(str(val)) < 2:
        return val
    val = str(val)
    pos = random.randint(0, len(val) - 1)
    return val[:pos] + random.choice(string.ascii_lowercase) + val[pos + 1:]


def _email_var(email: str) -> str:
    """Return a plausible email variation."""
    import random
    if pd.isna(email) or '@' not in str(email):
        return email
    user, domain = str(email).split('@', 1)
    variants = [
        user.replace('.', '') + '@' + domain,
        user + str(random.randint(1, 99)) + '@' + domain,
        user + '@gmail.com',
        user + '@outlook.com',
    ]
    return random.choice(variants)


def _abbreviate(city: str) -> str:
    """Abbreviate a city name to initials (e.g. New York → NY)."""
    if pd.isna(city):
        return city
    words = str(city).split()
    return ''.join(w[0].upper() for w in words) if len(words) > 1 else str(city)


def _flip_gender(val: str) -> str:
    """Flip M↔F; other values stay unchanged."""
    if str(val).upper() in ('M', 'MALE'):
        return 'F'
    if str(val).upper() in ('F', 'FEMALE'):
        return 'M'
    return val


# =============================================================================
# ── FULL EDA PIPELINE ─────────────────────────────────────────────────────────
# =============================================================================

def run_full_eda(
    df:         pd.DataFrame,
    id_col:     Optional[str] = None,
) -> tuple:
    """Run all automated EDA cleaning steps on a user-uploaded DataFrame.

    Steps applied (in order):
      1. Clean field names
      2. Detect field types
      3. Remove 100%-null columns
      4. Remove rows with 100%, n-1, n-2 nulls
      5. Clean text values (strip, proper case for names, lower for rest)
      6. Remove exact duplicate rows
      7. Standardise date columns to YYYY-MM-DD

    Correlation check (step 10) is NOT included here because it requires
    user interaction to choose which correlated field to keep.
    Call find_high_correlation_pairs() separately after this function.

    Returns:
      df_clean       : pd.DataFrame – cleaned data
      field_types    : dict         – col → type string
      name_mapping   : dict         – original col → cleaned col name
      eda_log        : dict         – record of what each step did
    """
    original_rows = len(df)
    original_cols = df.shape[1]
    eda_log       = {}                        # Accumulate log entries

    # ── Step 1: Clean field names ──────────────────────────────────────────────
    df, name_mapping = clean_field_names(df)
    eda_log['field_names'] = {
        'changed': {k: v for k, v in name_mapping.items() if k != v}
    }

    # Update id_col to its cleaned name if applicable
    if id_col and id_col in name_mapping:
        id_col = name_mapping[id_col]

    # ── Step 2: Detect field types ─────────────────────────────────────────────
    field_types = detect_field_types(df)

    # Ensure the designated ID column is typed as 'id'
    if id_col and id_col in field_types:
        field_types[id_col] = 'id'

    # ── Step 3: Remove fully-null columns ─────────────────────────────────────
    df, dropped_cols = remove_null_columns(df)
    eda_log['null_columns_dropped'] = dropped_cols

    # Re-detect types after dropping columns
    field_types = {c: t for c, t in field_types.items() if c in df.columns}

    # ── Steps 4-6: Remove near-empty rows ─────────────────────────────────────
    df, row_removal_counts = remove_null_rows(df)
    eda_log['null_rows_removed'] = row_removal_counts

    # ── Step 7: Clean text values ──────────────────────────────────────────────
    df = clean_text_values(df, field_types)
    eda_log['text_cleaned'] = True

    # ── Step 8: Remove exact duplicates ───────────────────────────────────────
    df, n_dupes = remove_exact_duplicates(df, id_col=id_col)
    eda_log['duplicates_removed'] = n_dupes

    # ── Step 9: Standardise date columns ──────────────────────────────────────
    date_cols = [c for c, t in field_types.items() if t in ('dob', 'date')]
    df, date_formats_used = standardise_dates(df, date_cols)
    eda_log['dates_standardised'] = date_formats_used

    # Final counts
    eda_log['summary'] = {
        'original_rows':  original_rows,
        'final_rows':     len(df),
        'rows_removed':   original_rows - len(df),
        'original_cols':  original_cols,
        'final_cols':     df.shape[1],
        'cols_removed':   original_cols - df.shape[1],
    }

    return df, field_types, name_mapping, eda_log
