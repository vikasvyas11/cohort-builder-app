<<<<<<< Updated upstream
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
=======
# modules/eda_engine.py
# Automated EDA, field validation, profiling, and type-safe noise injection pipeline.
# Purpose: Manages the programmatic text cleaning, missingness parsing, correlation sweeps,
#          and generates dynamic error injections without triggering float type errors on missing entries.

import re  # Import standard regular expressions for pattern matching and string cleaning
import string  # Import standard string library to access ASCII characters for typographical errors
import numpy as np  # Import numpy for fast array processing and random mask evaluations
import pandas as pd  # Import pandas for data frame manipulation and metadata tracking
from typing import Optional  # Import Optional for clean type-hinting support


# =============================================================================
# AUTOMATED DATA CLEANING AND STANDARDIZATION FUNCTIONS
# =============================================================================

def clean_field_names(df: pd.DataFrame) -> tuple:
    """Standardises dataframe column titles to lowercase alphanumeric formats."""
    df_copy = df.copy()  # Make an isolated copy of the dataframe to protect the input data
    original_columns = list(df_copy.columns)  # Extract the list of original incoming column titles
    standardized_map = {}  # Initialize an empty dictionary to record the field renamings

    for col in original_columns:  # Iterate through each column name in the dataframe
        clean_name = str(col).strip().lower()  # Strip whitespace and cast the title to lowercase
        clean_name = re.sub(r'[^a-z0-9_]', '_', clean_name)  # Replace non-alphanumeric marks with underscores
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')  # Remove redundant consecutive underscores
        if clean_name != col:  # Check if the newly formatted title differs from the baseline
            standardized_map[col] = clean_name  # Log the column transformation mapping inside the dictionary

    df_copy = df_copy.rename(columns=standardized_map)  # Apply the renaming map to the dataframe columns
    return df_copy, standardized_map  # Return the updated dataframe along with its transformation log


def remove_null_elements(df: pd.DataFrame) -> tuple:
    """Drops completely unassigned columns and rows with excessive missing data.

    Removal criteria (three separate passes, matching the spec exactly):
      Pass 1 — 100% null columns: columns where every value is null.
      Pass 2 — 100% null rows:   rows where every value is null.
      Pass 3 — n-1 null rows:    rows with only 1 non-null value  (thresh=2).
      Pass 4 — n-2 null rows:    rows with only 2 non-null values (thresh=3).

    Note: thresh=k in pandas means 'keep rows with at least k non-null values'.
    Setting thresh=3 removes rows that have fewer than 3 populated fields,
    which is exactly n-2 removal (only rows with ≥3 values survive).
    """
    df_copy = df.copy()
    initial_rows = len(df_copy)

    # ── Pass 1: Remove 100%-null columns ─────────────────────────────────────
    null_cols = [c for c in df_copy.columns if df_copy[c].isna().all()]
    df_copy = df_copy.drop(columns=null_cols)

    # ── Pass 2: Remove 100%-null rows ─────────────────────────────────────────
    df_copy = df_copy.dropna(how='all')
    rows_after_all_null = len(df_copy)

    # ── Pass 3: Remove rows with n-1 nulls (only 1 field has a value) ─────────
    df_copy = df_copy.dropna(thresh=2)   # keep rows with ≥2 non-null values
    rows_after_n1 = len(df_copy)

    # ── Pass 4: Remove rows with n-2 nulls (only 2 fields have a value) ───────
    df_copy = df_copy.dropna(thresh=3)   # keep rows with ≥3 non-null values
    final_rows = len(df_copy)

    log_summary = {
        "null_columns_dropped": null_cols,
        "100%_null":            initial_rows - rows_after_all_null,
        "n-1_null":             rows_after_all_null - rows_after_n1,
        "n-2_null":             rows_after_n1 - final_rows,
        "partial_null_removed": rows_after_all_null - final_rows,   # kept for compat
    }
    return df_copy, log_summary


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Removes extra spaces and normalizes character layouts in text columns."""
    df_copy = df.copy()  # Make an isolated copy of the dataframe to protect the input data
    for col in df_copy.columns:  # Loop through every column present in the working dataframe
        try:  # Trap unexpected data conversion issues gracefully inside a try-catch block
            # Convert values to strings, strip leading/trailing spaces, and compress internal spaces
            df_copy[col] = df_copy[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
            # Replace literal text variants of missing data with standard numpy NaN objects
            df_copy[col] = df_copy[col].replace(['nan', 'NAN', 'None', '', 'null', 'NULL'], np.nan)
        except Exception:  # Catch data type anomalies across columns gracefully
            continue  # Bypass columns that cannot be processed as standard text strings
    return df_copy  # Return the sanitized text dataframe structure


def remove_duplicate_records(df: pd.DataFrame) -> tuple:
    """Identifies and removes exact duplicate rows from the dataset."""
    df_copy = df.copy()  # Make an isolated copy of the dataframe to protect the input data
    initial_rows = len(df_copy)  # Track the row volume before running duplicate lookups
    df_copy = df_copy.drop_duplicates()  # De-duplicate identical rows across the entire dataset matrix
    removed_count = initial_rows - len(df_copy)  # Compute the exact count of duplicate records removed
    return df_copy, removed_count  # Return the unique dataframe rows along with the removal count


def standardise_date_formats(df: pd.DataFrame) -> tuple:
    """Parses date fields and normalizes them into uniform YYYY-MM-DD character strings."""
    df_copy = df.copy()  # Make an isolated copy of the dataframe to protect the input data
    standardized_log = {}  # Initialize an empty dictionary to document successful date conversions

    # Target fields with names containing birth, date, or dob patterns
    date_candidates = [c for c in df_copy.columns if 'date' in c or 'dob' in c or 'birth' in c]
    for col in date_candidates:  # Loop through each potential date column candidate
        try:  # Wrap the conversion attempt in a try block to handle malformed strings
            # Parse the column rows into a standard timestamp object
            parsed_dates = pd.to_datetime(df_copy[col], errors='coerce')
            if parsed_dates.notna().sum() > 0:  # Verify that at least some dates were parsed successfully
                df_copy[col] = parsed_dates.dt.strftime('%Y-%m-%d')  # Reformat matching values to YYYY-MM-DD
                standardized_log[col] = "YYYY-MM-DD"  # Log the successful formatting rule
        except Exception:  # Bypass columns containing text that cannot be parsed as a date
            continue
    return df_copy, standardized_log  # Return the formatted date dataframe alongside its logging tag


# =============================================================================
# COHORT ANALYSIS AND FIELD TYPE PROFILE ENGINES
# =============================================================================

def run_full_eda(df: pd.DataFrame, id_col: Optional[str] = None) -> tuple:
    """Executes the complete automated data cleaning and profiling pipeline."""
    eda_manifest_log = {}  # Instantiate the primary repository metadata tracking dictionary
    original_rows = len(df)  # Record the initial row count before running cleaning rules

    # 1. Clean and standardize column titles
    df_working, shifted_names = clean_field_names(df)
    eda_manifest_log["field_names"] = {"changed": shifted_names}

    # 2. Remove rows and columns that are entirely null
    df_working, null_summary = remove_null_elements(df_working)
    eda_manifest_log["null_columns_dropped"] = null_summary["null_columns_dropped"]
    eda_manifest_log["null_rows_removed"] = {
        "100%_null": null_summary["100%_null"],
        "n-1_null":  null_summary["n-1_null"],
        "n-2_null":  null_summary["n-2_null"],
    }

    # 3. Normalize text fields and remove extra whitespaces
    df_working = clean_text_fields(df_working)

    # 4. Standardize date formats for target fields
    df_working, date_summary = standardise_date_formats(df_working)
    eda_manifest_log["dates_standardised"] = date_summary

    # 5. Remove exact duplicate records
    df_working, duplicate_count = remove_duplicate_records(df_working)
    eda_manifest_log["duplicates_removed"] = duplicate_count

    # 6. Build the final operational summary report
    eda_manifest_log["summary"] = {
        "original_rows": original_rows,
        "rows_removed": original_rows - len(df_working),
        "final_rows": len(df_working),
        "cols_removed": len(null_summary["null_columns_dropped"])
    }

    # 7. Infer field semantic data types based on column names and values
    inferred_types = {}
    mapped_id = shifted_names.get(id_col, id_col) if id_col else None

    for col in df_working.columns:  # Loop through every cleaned column to determine its type
        if col == "unique_id" or col == mapped_id:
            inferred_types[col] = "id"  # Lock unique tracking index keys to the id type parameter
        elif "first" in col or "given" in col:
            inferred_types[col] = "first_name"  # Assign first name tags
        elif "last" in col or "sur" in col or "family" in col:
            inferred_types[col] = "surname"  # Assign surname tags
        elif "name" in col:
            inferred_types[col] = "full_name"  # Assign general full name tags
        elif "date" in col or "dob" in col or "birth" in col:
            inferred_types[col] = "dob"  # Assign date tags to date fields
        elif "email" in col:
            inferred_types[col] = "email"  # Assign email tags to email fields
        elif "post" in col or "zip" in col:
            inferred_types[col] = "postcode"  # Assign geographic postcodes tags
        elif "gender" in col or "sex" in col:
            inferred_types[col] = "gender"  # Assign demographic gender tags
        elif "city" in col or "town" in col or "county" in col:
            inferred_types[col] = "location"  # Assign geographic location tags
        else:
            inferred_types[col] = "text"  # Default back to basic text categorization

    return df_working, inferred_types, eda_manifest_log["summary"], eda_manifest_log


def find_high_correlation_pairs(df: pd.DataFrame, id_cols: list) -> list:
    """Identifies highly correlated columns to flag redundant fields before matching."""
    correlated_pairs_list = []  # Initialize an empty list container to hold correlated column pairs
    feature_cols = [c for c in df.columns if c not in id_cols and c not in ("unique_id", "cluster", "source_dataset")]

    for i in range(len(feature_cols)):  # Run a nested double loop to compare columns side-by-side
        for j in range(i + 1, len(feature_cols)):
            col_a = feature_cols[i]  # Target column A
            col_b = feature_cols[j]  # Target column B
            try:  # Wrap inside a try block to handle non-numeric or empty categories safely
                # Calculate the percentage of rows where column A exactly matches column B
                exact_match_ratio = (df[col_a] == df[col_b]).mean()
                if exact_match_ratio > 0.85:  # Flag pairs that have an exact agreement rate above 85%
                    correlated_pairs_list.append((col_a, col_b, exact_match_ratio))  # Log the pair parameters
            except Exception:
                continue
    # Sort the list so the most highly correlated columns appear first
    return sorted(correlated_pairs_list, key=lambda x: x[2], reverse=True)


def suggest_comparison_types(field_types: dict) -> dict:
    """Recommends optimal Splink comparison library functions based on field types."""
    recommendation_map = {}  # Initialize an empty dictionary to hold the comparison recommendations
    for col, ftype in field_types.items():  # Loop through every column type entry
        if ftype == "id":
            continue  # Skip unique identifier columns
        elif ftype == "first_name":
            recommendation_map[col] = "JaroWinklerAtThresholds"  # Suggest Jaro-Winkler for names
        elif ftype == "surname":
            recommendation_map[col] = "JaroAtThresholds"  # Suggest Jaro for surnames
        elif ftype == "dob":
            recommendation_map[col] = "DateOfBirthComparison"  # Suggest date matching rules for DOB
        elif ftype == "email":
            recommendation_map[col] = "EmailComparison"  # Suggest email matching rules
        elif ftype == "postcode":
            recommendation_map[col] = "PostcodeComparison"  # Suggest postcode matching rules
        else:
            recommendation_map[col] = "ExactMatch"  # Fall back to exact matching for everything else
    return recommendation_map  # Return the mapping dictionary


def suggest_blocking_rules(field_types: dict) -> dict:
    """Flags high-cardinality fields as initial blocking rule candidates."""
    blocking_suggestions = {}  # Initialize an empty dictionary to hold the blocking recommendations
    for col, ftype in field_types.items():  # Loop through every column type entry
        # Recommend blocking on stable categorical fields like names, dates, and locations
        if ftype in ("first_name", "surname", "dob", "postcode", "location"):
            blocking_suggestions[col] = True  # Mark the field as an appropriate blocking rule candidate
        else:
            blocking_suggestions[col] = False  # Mark the field as disabled for default blocking
    return blocking_suggestions  # Return the blocking mapping configuration


# =============================================================================
# TYPE-SAFE CUSTOMIZABLE NOISE INJECTION MODULE (FIX FOR LEN() ERROR)
# Purpose: Handles customizable error rate sliders robustly across missing values.
# =============================================================================

def introduce_errors_for_sample(
        df: pd.DataFrame,
        field_types: dict,
        sample_frac: float = 0.5,
        seed: int = 42,
        error_rates: dict = None,
) -> pd.DataFrame:
    """Generates Dataset B by injecting customizable typographical and missingness errors into populated fields."""
    import random  # Import the standard random module for string character adjustments
    rng = np.random.default_rng(seed)  # Instantiate an isolated numpy random generator using a static seed
    random.seed(seed)  # Enforce a static seed for the standard random framework

    # Downsample Dataset A to isolate the branch evaluation cohort
    sample = df.sample(frac=sample_frac, random_state=seed).copy()
    if error_rates is None:  # Fall back to empty defaults if no error parameters are provided
        error_rates = {}  # Initialize as an empty dictionary to avoid iteration errors

    for col, ftype in field_types.items():  # Loop through each field type configuration
        if col not in sample.columns:
            continue  # Skip column entries that do not match the dataframe schema

        rate = error_rates.get(col, 0.0)  # Retrieve the custom error rate percentage set by the user
        if rate <= 0.0:
            continue  # Skip error injection for fields with an error rate of 0%

        # STABILITY SECURITY GUARD 1: Intersect random mask with .notna()
        # This prevents floating point NaN representations from entering string code blocks
        mask = (rng.random(len(sample)) < rate) & sample[col].notna()
        if not mask.any():
            continue  # If no rows qualify for mutation under this field, skip column processing

        # STABILITY SECURITY GUARD 2: Explicitly typecast and guard variables against non-string execution
        if ftype in ('first_name', 'surname', 'full_name'):
            def corrupt_string_value(val):
                s_val = str(val)  # Force string conversion to protect len() attributes
                if len(s_val) <= 1:
                    return s_val  # Return base value if text cannot sustain meaningful substitutions
                return "".join(
                    char if random.random() > 0.5 else random.choice(string.ascii_lowercase) for char in s_val)

            sample.loc[mask, col] = sample.loc[mask, col].apply(corrupt_string_value)

        elif ftype == 'dob':
            # Safe operation: Force assignment of uniform float missingness tags
            sample.loc[mask, col] = np.nan

        elif ftype == 'email':
            def corrupt_email_value(val):
                s_val = str(val)  # Protect string separation functions from float breakdowns
                if "@" in s_val:
                    parts = s_val.split('@', 1)
                    return parts[0] + str(random.randint(1, 9)) + "@" + parts[1]
                return s_val + str(random.randint(1, 9))

            sample.loc[mask, col] = sample.loc[mask, col].apply(corrupt_email_value)

        elif ftype in ('location', 'postcode'):
            def corrupt_location_value(val):
                s_val = str(val)  # Protect length scans and substring slicing tasks
                if len(s_val) > 3:
                    return s_val[:3].upper()  # Return sliced shorthand tag format
                return s_val.swapcase()  # Reverse typography casing parameters as alternative noise

            sample.loc[mask, col] = sample.loc[mask, col].apply(corrupt_location_value)

        elif ftype == 'gender':
            def corrupt_gender_value(val):
                s_val = str(val).upper()  # Uniformly clean standard character properties
                return "F" if "M" in s_val else "M"  # Swap gender class assignments

            sample.loc[mask, col] = sample.loc[mask, col].apply(corrupt_gender_value)

        else:
            def corrupt_generic_text(val):
                s_val = str(val)  # Standard generic string fallback task wrapper
                if len(s_val) > 1:
                    return s_val[:-1] + random.choice(string.ascii_lowercase)
                return s_val

            sample.loc[mask, col] = sample.loc[mask, col].apply(corrupt_generic_text)

    # Suffix ID columns with _B — guards against NaN, floats, and already-suffixed values
    # ── Assign ground-truth cluster BEFORE renaming unique_id ─────────────────
    # The cluster value must equal the Dataset A unique_id so the confusion
    # matrix can match A records to their derived B counterparts.
    # We use the pre-suffix unique_id value as the shared cluster identifier.
    if "unique_id" in sample.columns:
        # Capture the original A-side unique_id as the cluster ground truth
        sample["cluster"] = sample["unique_id"].apply(
            lambda x: str(x) if pd.notna(x) else "NA"
        )
    elif "cluster" not in sample.columns:
        # Fallback: use integer row position as cluster if no unique_id exists
        sample["cluster"] = range(1, len(sample) + 1)

    # ── Suffix ID columns with _B to separate them from Dataset A IDs ─────────
    id_cols = [c for c, t in field_types.items() if t == 'id' and c in sample.columns]
    for id_col in id_cols:
        def _safe_b_suffix(val):
            s = str(val) if pd.notna(val) else "NA"
            return s if s.endswith('_B') else s + '_B'
        sample[id_col] = sample[id_col].apply(_safe_b_suffix)

    sample['source_dataset'] = 'B'
    return sample
>>>>>>> Stashed changes
