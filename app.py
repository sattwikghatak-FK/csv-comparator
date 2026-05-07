import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import gc

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SLA Comparator", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 95%; }
    div[data-testid="metric-container"] {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 12px 16px;
    }
    .streamlit-expanderHeader { font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ── Status config ─────────────────────────────────────────────────────────────
STATUS_META = {
    "Degraded" : {"bg": "#fef2f2", "icon": "▲", "color": "#dc2626"},
    "Improved" : {"bg": "#f0fdf4", "icon": "▼", "color": "#16a34a"},
    "Same"     : {"bg": "#f8fafc", "icon": "●", "color": "#6b7280"},
    "New"      : {"bg": "#eff6ff", "icon": "✦", "color": "#2563eb"},
    "Removed"  : {"bg": "#fff7ed", "icon": "✖", "color": "#ea580c"},
}
STATUS_ORDER = ["Degraded", "Improved", "Same", "New", "Removed"]

# ── State Callbacks ───────────────────────────────────────────────────────────
def reset_computation():
    """Clear everything when core inputs change."""
    for key in ["results", "key_cols", "val_col", "grp_col", "higher_is", "csv_data"]:
        if key in st.session_state:
            del st.session_state[key]

def reset_exports():
    """Clear generated CSV when UI filters are changed."""
    if "csv_data" in st.session_state:
        del st.session_state["csv_data"]

# ── Data Processing Helpers ───────────────────────────────────────────────────
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    
    seen = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols

    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in str_cols:
        df[c] = df[c].str.strip()
        
    df.replace("", np.nan, inplace=True)
    return df

def normalise_col_name(name: str) -> str:
    return str(name).strip().lower()

def match_columns(cols_a: list, cols_b: list) -> dict:
    norm_b = {normalise_col_name(c): c for c in cols_b}
    mapping = {}
    for ca in cols_a:
        nb = norm_b.get(normalise_col_name(ca))
        if nb is not None:
            mapping[ca] = nb
    return mapping   

@st.cache_data(show_spinner=False)
def get_sheet_names(file_bytes: bytes) -> list:
    xls = pd.ExcelFile(BytesIO(file_bytes))
    return xls.sheet_names

@st.cache_data(show_spinner=False)
def get_preview_data(file_bytes: bytes, file_name: str, sheet_name: str = None) -> pd.DataFrame:
    if file_name.lower().endswith('.xlsx'):
        df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, dtype=str, keep_default_na=False, nrows=2000)
    else:
        df = pd.read_csv(BytesIO(file_bytes), dtype=str, keep_default_na=False, skipinitialspace=True, encoding_errors="replace", nrows=2000)
    return clean_df(df)

def load_data(file_bytes: bytes, file_name: str, sheet_name: str = None) -> pd.DataFrame:
    if file_name.lower().endswith('.xlsx'):
        df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(BytesIO(file_bytes), dtype=str, keep_default_na=False, skipinitialspace=True, encoding_errors="replace")
    return clean_df(df)

def sniff_numeric(df: pd.DataFrame, col: str, sample: int = 1000) -> bool:
    vals = df[col].dropna().head(sample)
    if vals.empty: return False
    return pd.to_numeric(vals, errors="coerce").notna().mean() > 0.6

def infer_direction(val_col: str) -> str:
    name = normalise_col_name(val_col)
    positive_keywords = ["score", "rating", "accuracy", "fill", "efficiency",
                         "utilisation", "utilization", "revenue", "profit",
                         "coverage", "success", "satisfaction", "nps"]
    for kw in positive_keywords:
        if kw in name: return "higher_is_better"
    return "higher_is_worse"

def make_key(df, cols):
    if len(cols) == 1: return df[cols[0]].astype(str)
    res = df[cols[0]].astype(str)
    for col in cols[1:]:
        res += " › " + df[col].astype(str)
    return res

# ── Unified Chunk Processing Engine ───────────────────────────────────────────
def process_comparison_chunked(df_a, df_b, comp_mode, granular_file, col_map, key_cols, val_col, grp_col, higher_is, status_text=None, progress_bar=None) -> pd.DataFrame:
    def update_ui(msg, pct):
        if status_text: status_text.markdown(f"**⏳ {msg}**")
        if progress_bar: progress_bar.progress(pct)

    df_a, df_b = df_a.copy(), df_b.copy()

    update_ui("Aligning columns and padding missing data...", 55)
    df_b.rename(columns={v: k for k, v in col_map.items()}, inplace=True)
    
    for k in key_cols:
        if k not in df_a.columns: df_a[k] = "MISSING_IN_A"
        if k not in df_b.columns: df_b[k] = "MISSING_IN_B"

    cols_a_all, cols_b_all = list(df_a.columns), list(df_b.columns)
    common_cols = [c for c in cols_a_all if c in cols_b_all]
    only_a = [c for c in cols_a_all if c not in cols_b_all]
    only_b = [c for c in cols_b_all if c not in cols_a_all]

    update_ui("Generating composite keys...", 60)
    df_a["__key__"] = make_key(df_a, key_cols)
    df_b["__key__"] = make_key(df_b, key_cols)

    update_ui("Preparing memory chunks...", 65)
    all_keys = pd.unique(pd.concat([df_a["__key__"], df_b["__key__"]]))
    
    CHUNK_SIZE = 50000 
    num_chunks = max(1, len(all_keys) // CHUNK_SIZE + (1 if len(all_keys) % CHUNK_SIZE != 0 else 0))
    
    processed_chunks = []
    is_strict = "1-to-1" in comp_mode
    dedup_a, dedup_b = True, True
    if not is_strict:
        if "File B" in granular_file: dedup_b = False
        else: dedup_a = False

    for i in range(num_chunks):
        chunk_pct = 65 + int(25 * (i / num_chunks)) 
        update_ui(f"Calculating chunk {i+1} of {num_chunks} ({(i*CHUNK_SIZE):,} to {min((i+1)*CHUNK_SIZE, len(all_keys)):,} identifiers)...", chunk_pct)
        
        chunk_keys = all_keys[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE]
        
        sub_a = df_a[df_a["__key__"].isin(chunk_keys)].copy()
        sub_b = df_b[df_b["__key__"].isin(chunk_keys)].copy()
        
        if dedup_a: sub_a = sub_a.drop_duplicates("__key__")
        if dedup_b: sub_b = sub_b.drop_duplicates("__key__")
        
        merged = pd.merge(
            sub_a.rename(columns={val_col: "_val_A"}),
            sub_b.rename(columns={val_col: "_val_B"}),
            on="__key__", how="outer", suffixes=("_A", "_B")
        )
        
        merged["_val_A"] = pd.to_numeric(merged["_val_A"], errors="coerce")
        merged["_val_B"] = pd.to_numeric(merged["_val_B"], errors="coerce")
        merged["Δ Change"] = merged["_val_B"] - merged["_val_A"]

        deg_cond = merged["_val_B"] > merged["_val_A"] if higher_is == "higher_is_worse" else merged["_val_B"] < merged["_val_A"]
        imp_cond = merged["_val_B"] < merged["_val_A"] if higher_is == "higher_is_worse" else merged["_val_B"] > merged["_val_A"]
        
        merged["Status"] = np.select(
            [merged["_val_A"].isna() & merged["_val_B"].notna(), merged["_val_A"].notna() & merged["_val_B"].isna(), deg_cond, imp_cond],
            ["New", "Removed", "Degraded", "Improved"], default="Same"
        )

        for c in [c for c in common_cols if c not in ["__key__", val_col]]:
            col_a, col_b = f"{c}_A", f"{c}_B"
            if col_a in merged.columns and col_b in merged.columns:
                if not is_strict and "File B" in granular_file:
                    merged[c] = merged[col_b].combine_first(merged[col_a])
                elif not is_strict:
                    merged[c] = merged[col_a].combine_first(merged[col_b])
                else:
                    merged[c] = merged[col_b].combine_first(merged[col_a])
                merged.drop(columns=[col_a, col_b], inplace=True)
            elif col_a in merged.columns: merged.rename(columns={col_a: c}, inplace=True)
            elif col_b in merged.columns: merged.rename(columns={col_b: c}, inplace=True)

        if not merged.empty:
            processed_chunks.append(merged)
        
        del sub_a, sub_b, merged
        gc.collect()

    update_ui("Assembling final dataset...", 95)
    
    if processed_chunks:
        final_merged = pd.concat(processed_chunks, ignore_index=True)
    else:
        final_merged = pd.DataFrame() 
    
    if grp_col and grp_col in final_merged.columns:
        final_merged["Group"] = final_merged[grp_col].fillna("Unknown")
    else:
        final_merged["Group"] = "All"
    
    key_display = " › ".join(key_cols)
    final_merged.rename(columns={"__key__": key_display, "_val_A": f"{val_col} (File A)", "_val_B": f"{val_col} (File B)"}, inplace=True)

    base_cols = [key_display, f"{val_col} (File A)", f"{val_col} (File B)", "Δ Change", "Status", "Group"]
    all_context = [c for c in (common_cols + only_a + only_b) if c not in base_cols and c not in ["__key__", val_col]]
    context_cols = list(dict.fromkeys([c for c in all_context if c in final_merged.columns]))
    
    if not is_strict:
        final_merged = final_merged.drop_duplicates()
        
    update_ui("Analysis Complete! 🚀", 100)
    return final_merged[[c for c in base_cols + context_cols if c in final_merged.columns]]

# ── Styler ────────────────────────────────────────────────────────────────────
def style_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_style(row):
        bg_color = STATUS_META.get(row['Status'], {}).get('bg', '#ffffff')
        text_color = STATUS_META.get(row['Status'], {}).get('color', '#1e293b')
        return [f"background-color: {bg_color}; color: {text_color};"] * len(row)

    df = df.copy()
    for c in df.columns:
        if c in ["Δ Change"] or c.endswith("(File A)") or c.endswith("(File B)"):
            try:
                converted = pd.to_numeric(df[c], errors="coerce")
                if converted.notna().sum() > 0: df[c] = converted
            except Exception: pass

    num_fmt = {c: "{:,.4g}" for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    return df.style.apply(row_style, axis=1).format(num_fmt, na_rep="—").set_properties(**{"font-size": "13px", "font-weight": "500"})

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📊 SLA Comparator")
st.caption("Compare massive CSV/Excel datasets seamlessly · Chunk Processing Engine Enabled")

# ── Step 1: Upload ─────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("#### 1. Upload Datasets")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📁 File A — Baseline / Previous**")
        up_a = st.file_uploader("File A", type=["csv", "xlsx"], key="fa", label_visibility="collapsed", on_change=reset_computation)
        sheet_a = st.selectbox("📝 Select Sheet (File A)", get_sheet_names(up_a.getvalue()), on_change=reset_computation) if up_a and up_a.name.lower().endswith(".xlsx") else None
    with c2:
        st.markdown("**📁 File B — Current / New**")
        up_b = st.file_uploader("File B", type=["csv", "xlsx"], key="fb", label_visibility="collapsed", on_change=reset_computation)
        sheet_b = st.selectbox("📝 Select Sheet (File B)", get_sheet_names(up_b.getvalue()), on_change=reset_computation) if up_b and up_b.name.lower().endswith(".xlsx") else None

if not (up_a and up_b):
    st.info("⬆ Upload both files to configure your comparison.", icon="ℹ️")
    st.stop()

with st.spinner("Extracting headers and detecting data types..."):
    df_a_preview = get_preview_data(up_a.getvalue(), up_a.name, sheet_a)
    df_b_preview = get_preview_data(up_b.getvalue(), up_b.name, sheet_b)

col_map = match_columns(list(df_a_preview.columns), list(df_b_preview.columns))
common = list(col_map.keys())

if not common:
    st.error("No matching columns found between the two files. Please verify headers.")
    st.stop()

# ── Step 2: Configure ──────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("#### 2. Configure Comparison")
    
    mode_c1, mode_c2 = st.columns(2)
    comp_mode = mode_c1.radio("⚙️ Match Architecture", ["Strict 1-to-1 (Deduplicate Both)", "1-to-Many (Broadcast granular rows)"], index=0, on_change=reset_computation)
    
    granular_file = None
    if "1-to-Many" in comp_mode:
        granular_file = mode_c2.selectbox("📌 Which file contains the granular data? (Keep its duplicates)", ["File B (Current/New)", "File A (Baseline/Previous)"], on_change=reset_computation)
    st.markdown("---")
    
    numeric_cols = [c for c in common if sniff_numeric(df_a_preview, c) or sniff_numeric(df_b_preview, col_map[c])]
    cfg1, cfg2, cfg3 = st.columns([2, 1, 1])
    with cfg1:
        key_cols = st.multiselect("🔑 Unique Identifier(s)", options=common, default=[common[0]] if common else [], on_change=reset_computation)
    with cfg2:
        val_col = st.selectbox("📐 Metric to Compare", options=[c for c in numeric_cols if c not in key_cols] or [c for c in common if c not in key_cols], on_change=reset_computation)
    with cfg3:
        grp_sel = st.selectbox("🗂 Group By (Optional)", options=["(none)"] + [c for c in common if c != val_col], on_change=reset_computation)
        grp_col = None if grp_sel == "(none)" else grp_sel

    higher_is = st.radio(
        "📈 Value direction meaning", options=["higher_is_worse", "higher_is_better"],
        index=0 if infer_direction(val_col) == "higher_is_worse" else 1,
        format_func=lambda x: "⬆ Higher = Worse (e.g., Latency)" if x == "higher_is_worse" else "⬆ Higher = Better (e.g., Score)",
        horizontal=True, on_change=reset_computation
    )

    run = st.button("🚀 Run Full Analysis", type="primary", width="stretch", disabled=not (key_cols and val_col))

if not run and "results" not in st.session_state:
    st.stop()

# ── Step 3: Compute ────────────────────────────────────────────────────────────
if run:
    st.markdown("---")
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    status_text.markdown(f"**⏳ Reading {up_a.name} into memory... (Excel files may take a minute)**")
    progress_bar.progress(10)
    df_a_full = load_data(up_a.getvalue(), up_a.name, sheet_a)
    
    status_text.markdown(f"**⏳ Reading {up_b.name} into memory... (Almost there)**")
    progress_bar.progress(35)
    df_b_full = load_data(up_b.getvalue(), up_b.name, sheet_b)

    results = process_comparison_chunked(df_a_full, df_b_full, comp_mode, granular_file, col_map, key_cols, val_col, grp_col, higher_is, status_text, progress_bar)
        
    st.session_state.update({"results": results, "key_cols": key_cols, "val_col": val_col, "grp_col": grp_col, "higher_is": higher_is})
    
    status_text.empty()
    progress_bar.empty()

results, key_cols, val_col, grp_col, higher_is = [st.session_state[k] for k in ["results", "key_cols", "val_col", "grp_col", "higher_is"]]

# ── Step 4: Display Results ────────────────────────────────────────────────────
sc = results["Status"].value_counts()
cols_m = st.columns(6)
cols_m[0].metric("Total Rows Evaluated", f"{len(results):,}")
for i, s in enumerate(STATUS_ORDER):
    cols_m[i+1].metric(f"{STATUS_META[s]['icon']} {s}", f"{sc.get(s, 0):,}")

st.markdown("")
tab_data, tab_export = st.tabs(["📋 Detailed Data Viewer", "💾 CSV Export"])

with tab_data:
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    status_filter = fc1.multiselect("Filter by Status", STATUS_ORDER, default=[], placeholder="All statuses", on_change=reset_exports)
    search = fc2.text_input("🔍 Search in Key", placeholder="Type to filter...", on_change=reset_exports)
    
    sort_opts = ["Δ Change", f"{val_col} (File A)", f"{val_col} (File B)", "Status", " › ".join(key_cols)]
    sort_by = fc3.selectbox("Sort Data By", sort_opts, on_change=reset_exports)

    view = results 
    if status_filter: view = view[view["Status"].isin(status_filter)]
    if search: view = view[view[" › ".join(key_cols)].astype(str).str.contains(search, case=False, na=False)]
    if sort_by in view.columns: view = view.sort_values(sort_by, ascending=(sort_by == "Status"), na_position="last")

    st.caption(f"Showing **{len(view):,}** of **{len(results):,}** rows across **{view['Group'].nunique()}** group(s)")

    for grp_name, grp_df in view.groupby("Group", sort=True, dropna=False):
        gc = grp_df["Status"].value_counts()
        badges = "  ".join(f"{STATUS_META[s]['icon']} {s}: {gc.get(s,0)}" for s in STATUS_ORDER if gc.get(s, 0) > 0)
        grp_label = "Overall Dataset" if grp_name == "All" else f"{grp_col}: {grp_name}"
        
        with st.expander(f"{grp_label} ({len(grp_df):,} rows)  |  {badges}", expanded=(grp_name == "All" or view["Group"].nunique() <= 2)):
            show = grp_df.drop(columns=["Group"], errors="ignore")
            st.dataframe(style_table(show.head(1500)), width="stretch", height=min(500, 45 + len(show) * 36))
            if len(show) > 1500:
                st.warning(f"⚠️ Showing first 1,500 rows for UI performance. Please export CSV to view all {len(show):,} rows.")

with tab_export:
    st.markdown("#### Download CSV Report")
    st.caption("Downloads the current filtered view as a fast, lightweight CSV file.")
    
    if "csv_data" not in st.session_state:
        if st.button("⚙️ Generate CSV File", width="stretch", type="primary"):
            with st.spinner("Preparing CSV data..."):
                st.session_state["csv_data"] = view.drop(columns=["Group"], errors="ignore").to_csv(index=False).encode('utf-8')
            st.rerun() 
    else:
        st.download_button(
            "⬇️ Download CSV File", 
            data=st.session_state["csv_data"], 
            file_name=f"SLA_Report_{val_col}.csv", 
            mime="text/csv", 
            width="stretch",
            type="primary"
        )
        if st.button("🗑️ Clear CSV from memory", width="stretch"):
            del st.session_state["csv_data"]
            st.rerun()
