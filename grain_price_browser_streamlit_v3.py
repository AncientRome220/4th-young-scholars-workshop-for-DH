# grain_price_browser_streamlit_v3.py
# Run:
#   python -m streamlit run grain_price_browser_streamlit_v3.py
# or:
#   python -m streamlit run grain_price_browser_streamlit_v3.py -- --csv wheat_mentions_v10b_strict.csv
#
# v3 additions:
# - "Open XML" button in the record inspector (when XML_RelPath is available)
# - Optional DDB XML root folder input to locate local XML files
# - Show XML content inside Streamlit + download XML

import argparse
from pathlib import Path

import pandas as pd
import streamlit as st


def normalize_search_blob(df: pd.DataFrame) -> pd.DataFrame:
    """Create a single search column for fast global substring filtering."""
    df = df.copy()
    df["_search_blob"] = df.astype(str).agg(" | ".join, axis=1).str.casefold()
    return df


@st.cache_data(show_spinner=False)
def load_csv_any(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["Date_When", "Date_NotBefore", "Date_NotAfter"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return normalize_search_blob(df)


@st.cache_data(show_spinner=False)
def load_csv_uploaded(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    for col in ["Date_When", "Date_NotBefore", "Date_NotAfter"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return normalize_search_blob(df)


def unique_nonempty(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return []
    vals = df[col].dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    return sorted(vals.unique().tolist())


def add_sidebar_multiselect(df: pd.DataFrame, col: str, label: str):
    options = unique_nonempty(df, col)
    if not options:
        return []
    return st.sidebar.multiselect(label, options, default=[])


def choose_display_key(df: pd.DataFrame):
    if "DDB_ID" in df.columns and "Mention_ID" in df.columns:
        return "mention"
    if "DDB_ID" in df.columns:
        return "doc"
    return "index"


@st.cache_data(show_spinner=False)
def read_text_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Path(path).read_text(encoding="utf-8", errors="replace")


def main():
    st.set_page_config(page_title="Grain Price CSV Browser", layout="wide")
    st.title("Roman Egypt Grain Price CSV Browser")
    st.caption("Browse, search, filter, inspect, and export your candidate/mention CSVs.")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", default="")
    args, _ = parser.parse_known_args()

    # Sidebar: data source
    st.sidebar.header("Data source")

    csv_files = sorted([p.name for p in Path(".").glob("*.csv")])
    default_guess = args.csv if args.csv else (csv_files[0] if csv_files else "")

    source_mode = st.sidebar.radio("Load mode", ["Pick a CSV file", "Type a path", "Upload CSV"], index=0)

    df = None
    loaded_name = ""

    if source_mode == "Pick a CSV file":
        if not csv_files:
            st.sidebar.info("No CSV files found in the current folder.")
        picked = st.sidebar.selectbox(
            "Choose CSV",
            options=[""] + csv_files,
            index=(1 if default_guess in csv_files else 0),
        )
        if picked:
            df = load_csv_any(picked)
            loaded_name = picked

    elif source_mode == "Type a path":
        path_str = st.sidebar.text_input("CSV path", value=default_guess)
        if path_str.strip():
            p = Path(path_str)
            if p.exists():
                df = load_csv_any(str(p))
                loaded_name = str(p)
            else:
                st.sidebar.error("CSV path not found.")

    else:  # Upload
        up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            df = load_csv_uploaded(up)
            loaded_name = up.name

    # Sidebar: XML options
    st.sidebar.header("XML viewer (optional)")
    enable_xml = st.sidebar.checkbox("Enable XML button", value=True)
    ddb_xml_root = st.sidebar.text_input(
        "DDB XML root folder",
        value="",
        help="Folder that contains the DDB EpiDoc XML tree. Used together with XML_RelPath.",
    )

    if df is None:
        st.info("Select or upload a CSV to begin.")
        return

    st.subheader("Loaded file")
    st.write(f"**{loaded_name}** — rows: **{len(df):,}**, columns: **{len(df.columns)-1:,}**")

    # Sidebar: global search
    st.sidebar.header("Search")
    q = st.sidebar.text_input(
        "Global substring search",
        value="",
        help="Search across ALL columns (case-insensitive).",
    )

    # Sidebar: filters
    st.sidebar.header("Filters (optional)")

    grain_col = "Grain_Hits" if "Grain_Hits" in df.columns else ("Grain_Hit" if "Grain_Hit" in df.columns else "")
    unit_col = "Unit_Hits" if "Unit_Hits" in df.columns else ("Unit_Hit" if "Unit_Hit" in df.columns else "")
    money_col = "Money_Hits" if "Money_Hits" in df.columns else ("Money_Hit" if "Money_Hit" in df.columns else "")

    grains = add_sidebar_multiselect(df, grain_col, "Grain")
    units = add_sidebar_multiselect(df, unit_col, "Unit")
    money = add_sidebar_multiselect(df, money_col, "Money")

    priceword = add_sidebar_multiselect(df, "Priceword_Hit", "Priceword_Hit")
    number_type = add_sidebar_multiselect(df, "Number_Type", "Number_Type")
    place = add_sidebar_multiselect(df, "Place", "Place")
    score = add_sidebar_multiselect(df, "Score", "Score")

    filtered = df.copy()

    def apply_in(colname, selected):
        nonlocal filtered
        if colname and selected and colname in filtered.columns:
            filtered = filtered[filtered[colname].astype(str).isin([str(x) for x in selected])]

    apply_in(grain_col, grains)
    apply_in(unit_col, units)
    apply_in(money_col, money)
    apply_in("Priceword_Hit", priceword)
    apply_in("Number_Type", number_type)
    apply_in("Place", place)
    apply_in("Score", score)

    # Date range filter
    if "Date_When" in filtered.columns and filtered["Date_When"].notna().any():
        st.sidebar.subheader("Date range (Date_When)")
        min_date = filtered["Date_When"].min()
        max_date = filtered["Date_When"].max()
        if pd.notna(min_date) and pd.notna(max_date):
            start, end = st.sidebar.date_input(
                "From / To",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            filtered = filtered[(filtered["Date_When"] >= start_ts) & (filtered["Date_When"] <= end_ts)]

    if q.strip():
        needle = q.casefold()
        filtered = filtered[filtered["_search_blob"].str.contains(needle, na=False)]

    # Display
    st.subheader("Results")
    st.write(f"Rows: **{len(filtered):,}** / {len(df):,}")

    available_cols = [c for c in df.columns if c != "_search_blob"]

    if "Mention_ID" in available_cols:
        default_cols = [
            "DDB_ID", "Mention_ID", "Block_Tag",
            "Title", "Place", "Date_Text",
            grain_col, unit_col, money_col,
            "Priceword_Hit",
            "Number_Hit", "Number_Type", "Num_Tag_Values",
            "Quantities", "Prices",
            "Raw_Block",
        ]
    else:
        default_cols = [
            "DDB_ID", "Title", "Place", "Date_Text",
            grain_col, unit_col, money_col,
            "Priceword_Hit",
            "Num_Tag_Values",
            "Score", "Block_Match_Count",
            "Snippet_1",
            "XML_RelPath",
        ]

    default_cols = [c for c in default_cols if c in available_cols]

    cols = st.multiselect("Columns to display", options=available_cols, default=default_cols)
    if not cols:
        st.warning("Select at least one column to display.")
        return

    show_df = filtered
    if "Date_When" in show_df.columns:
        show_df = show_df.sort_values(by="Date_When", ascending=False, na_position="last")

    st.dataframe(show_df[cols], use_container_width=True, hide_index=True)

    # Inspect one record
    st.subheader("Inspect one record")
    key_mode = choose_display_key(show_df)

    if len(show_df) == 0:
        st.info("No rows to inspect.")
    else:
        if key_mode == "mention":
            labels = show_df.apply(lambda r: f"{r.get('DDB_ID','')} / {r.get('Mention_ID','')}", axis=1)
            idx = st.selectbox("Choose mention", options=show_df.index.tolist(), format_func=lambda i: labels.loc[i])
        elif key_mode == "doc":
            labels = show_df["DDB_ID"].astype(str)
            idx = st.selectbox("Choose document (DDB_ID)", options=show_df.index.tolist(), format_func=lambda i: labels.loc[i])
        else:
            idx = st.selectbox("Choose row", options=show_df.index.tolist(), format_func=lambda i: str(i))

        row = show_df.loc[idx]

        meta_cols = [c for c in ["DDB_ID", "Mention_ID", "Title", "Place", "Date_Text", "Date_When", "Date_NotBefore", "Date_NotAfter"] if c in row.index]
        hit_cols = [c for c in [grain_col, unit_col, money_col, "Priceword_Hit", "Number_Hit", "Number_Type", "Num_Tag_Values", "Num_Tag_Texts", "Quantities", "Prices"] if c in row.index]

        st.markdown("**Metadata**")
        st.json({c: (None if pd.isna(row[c]) else str(row[c])) for c in meta_cols})

        st.markdown("**Hits / Extraction**")
        st.json({c: ("" if pd.isna(row[c]) else str(row[c])) for c in hit_cols})

        for long_col in ["Raw_Block", "Value_Block", "Snippet_1", "Snippet_2", "Snippet_3"]:
            if long_col in row.index and str(row[long_col]).strip():
                st.markdown(f"**{long_col}**")
                st.code(str(row[long_col]))

        # XML button section (doc-level CSV must include XML_RelPath)
        if enable_xml and "XML_RelPath" in row.index and str(row.get("XML_RelPath", "")).strip():
            st.markdown("### XML")

            if not ddb_xml_root.strip():
                st.info("To use the XML button: set the **DDB XML root folder** in the sidebar.")
            else:
                rel = str(row["XML_RelPath"]).replace("\\", "/").strip()
                xml_path = Path(ddb_xml_root) / rel
                st.write(f"XML path: `{xml_path}`")

                if not xml_path.exists():
                    st.error("XML file not found at that path. Check your 'DDB XML root folder'.")
                else:
                    colA, colB = st.columns([1, 1])

                    with colA:
                        open_now = st.button("Open XML in viewer")

                    with colB:
                        xml_text_for_dl = read_text_file(str(xml_path))
                        st.download_button(
                            "Download XML",
                            data=xml_text_for_dl.encode("utf-8"),
                            file_name=xml_path.name,
                            mime="application/xml",
                        )

                    if open_now:
                        xml_text = read_text_file(str(xml_path))
                        max_chars = 80_000
                        if len(xml_text) > max_chars:
                            st.warning(f"XML is large ({len(xml_text):,} chars). Showing first {max_chars:,} chars.")
                            xml_text = xml_text[:max_chars]
                        st.code(xml_text, language="xml")

    # Download filtered CSV
    export_df = show_df.drop(columns=["_search_blob"], errors="ignore")
    st.download_button(
        "Download filtered CSV",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
