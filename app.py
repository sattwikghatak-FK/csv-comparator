import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSV Comparator",
    page_icon="📊",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stDataFrame { font-size: 13px; }
    div[data-testid="metric-container"] {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 12px 16px;
    }
    .status-badge {
        display: inline-block; padding: 2px 10px;
        border-radius: 999px; font-size: 12px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
STATUS_COLORS = {
    "Degraded" : "#fef2f2",
    "Improved" : "#f0fdf4",
    "Same"     : "#f8fafc",
    "New"      : "#eff6ff",
    "Removed"  : "#fff7ed",
}
BADGE_COLORS = {
    "Degraded" : ("▲", "#dc2626"),
    "Improved" : ("▼", "#16a34a"),
    "Same"     : ("●", "#6b7280"),
    "New"      : ("✦", "#2563eb"),
    "Removed"  : ("✖", "#ea580c"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read CSV robustly, cache by content hash."""
    return pd.read_csv(
        BytesIO(file_bytes),
        dtype=str,           # keep everything as string initially
        keep_default_na=False,
        skipinitialspace=True,
    )

def sniff_numeric(df: pd.DataFrame, col: str, sample: int = 500) -> bool:
    """Check if a column looks numeric on a sample."""
    vals = df[col].dropna().head(sample)
    converted = pd.to_numeric(vals, errors="coerce")
    return converted.notna().mean() > 0.6   # >60% parseable → numeric

def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def compare(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    key_cols: list[str],
    val_col: str,
    grp_col: str | None,
) -> pd.DataFrame:
    """Core comparison — works on DataFrames of any shape."""
    # Build composite key
    def make_key(df):
        if len(key_cols) == 1:
            return df[key_cols[0]].astype(str)
        return df[key_cols].astype(str).agg(" › ".join, axis=1)

    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a["__key__"] = make_key(df_a)
    df_b["__key__"] = make_key(df_b)

    # Keep only what we need → memory efficient for large files
    cols_needed_a = ["__key__", val_col] + ([grp_col] if grp_col else [])
    cols_needed_b = ["__key__", val_col] + ([grp_col] if grp_col else [])
    df_a = df_a[[c for c in cols_needed_a if c in df_a.columns]].drop_duplicates("__key__")
    df_b = df_b[[c for c in cols_needed_b if c in df_b.columns]].drop_duplicates("__key__")

    # Outer join on key
    merged = pd.merge(
        df_a.rename(columns={val_col: "Value_A", **({"__key__": "__key__"} if True else {})}),
        df_b.rename(columns={val_col: "Value_B"}),
        on="__key__",
        how="outer",
        suffixes=("_A", "_B"),
    )

    merged["Value_A"] = to_numeric_safe(merged["Value_A"])
    merged["Value_B"] = to_numeric_safe(merged["Value_B"])
    merged["Δ Change"]= merged["Value_B"] - merged["Value_A"]

    # Status logic
    conditions = [
        merged["Value_A"].isna() & merged["Value_B"].notna(),
        merged["Value_A"].notna() & merged["Value_B"].isna(),
        merged["Value_B"] > merged["Value_A"],
        merged["Value_B"] < merged["Value_A"],
    ]
    choices = ["New", "Removed", "Degraded", "Improved"]
    merged["Status"] = np.select(conditions, choices, default="Same")

    # Group column — coalesce from A/B side
    if grp_col:
        gcol_a = f"{grp_col}_A" if f"{grp_col}_A" in merged.columns else grp_col
        gcol_b = f"{grp_col}_B" if f"{grp_col}_B" in merged.columns else grp_col
        if gcol_a in merged.columns and gcol_b in merged.columns:
            merged["Group"] = merged[gcol_a].combine_first(merged[gcol_b])
        elif gcol_a in merged.columns:
            merged["Group"] = merged[gcol_a]
        else:
            merged["Group"] = merged.get(gcol_b, "—")
        merged.drop(columns=[c for c in [gcol_a, gcol_b] if c in merged.columns and c != "Group"], inplace=True)
    else:
        merged["Group"] = "All"

    # Clean up
    merged = merged.rename(columns={
        "__key__" : " › ".join(key_cols),
        "Value_A" : f"{val_col} (A)",
        "Value_B" : f"{val_col} (B)",
    })

    final_cols = [" › ".join(key_cols), f"{val_col} (A)", f"{val_col} (B)", "Δ Change", "Status", "Group"]
    return merged[[c for c in final_cols if c in merged.columns]]

def style_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply row-level background colors by Status."""
    def row_color(row):
        color = STATUS_COLORS.get(row["Status"], "#ffffff")
        return [f"background-color: {color}"] * len(row)

    fmt = {}
    for c in df.columns:
        if df[c].dtype == float or c == "Δ Change":
            fmt[c] = "{:,.2f}"

    return (
        df.style
        .apply(row_color, axis=1)
        .format(fmt, na_rep="—")
        .set_properties(**{"font-size": "12px"})
    )

def build_excel(results_by_group: dict) -> bytes:
    """Export results to multi-sheet Excel (one sheet per group)."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for grp, df in results_by_group.items():
            sheet_name = str(grp)[:31]   # Excel sheet name limit
            df.drop(columns=["Group"], errors="ignore").to_excel(
                writer, sheet_name=sheet_name, index=False
            )
            ws = writer.sheets[sheet_name]
            ws.set_column(0, len(df.columns)-1, 20)
    return buf.getvalue()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📊 Universal CSV Comparator")
st.caption("Upload any two CSVs of similar format · Multi-column keys · Grouped analysis · Export to Excel")

# ── Step 1: Upload ─────────────────────────────────────────────────────────────
st.markdown("### 1 · Upload Files")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**📁 File A — Baseline / Previous**")
    up_a = st.file_uploader("Upload File A", type="csv", key="fa", label_visibility="collapsed")
with col2:
    st.markdown("**📁 File B — Current / New**")
    up_b = st.file_uploader("Upload File B", type="csv", key="fb", label_visibility="collapsed")

if not (up_a and up_b):
    st.info("⬆ Upload both CSV files to get started.", icon="ℹ️")
    st.stop()

# Load (cached by content)
with st.spinner("Parsing files…"):
    df_a = load_csv(up_a.read(), up_a.name)
    df_b = load_csv(up_b.read(), up_b.name)

c1, c2 = st.columns(2)
c1.success(f"**{up_a.name}** — {len(df_a):,} rows · {len(df_a.columns)} cols")
c2.success(f"**{up_b.name}** — {len(df_b):,} rows · {len(df_b.columns)} cols")

# Column overlap info
common  = [c for c in df_a.columns if c in df_b.columns]
only_a  = [c for c in df_a.columns if c not in df_b.columns]
only_b  = [c for c in df_b.columns if c not in df_a.columns]

with st.expander("📋 Column overview", expanded=False):
    cc1, cc2, cc3 = st.columns(3)
    cc1.markdown(f"**✅ Common ({len(common)})**\n\n" + "\n".join(f"- `{c}`" for c in common))
    cc2.markdown(f"**⚠ Only in A ({len(only_a)})**\n\n" + ("\n".join(f"- `{c}`" for c in only_a) or "None"))
    cc3.markdown(f"**✦ Only in B ({len(only_b)})**\n\n" + ("\n".join(f"- `{c}`" for c in only_b) or "None"))

if not common:
    st.error("No common columns found. Ensure both CSVs share at least one column header.")
    st.stop()

# ── Step 2: Configure ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 2 · Configure Comparison")

numeric_cols = [c for c in common if sniff_numeric(df_a, c) or sniff_numeric(df_b, c)]
non_numeric  = [c for c in common if c not in numeric_cols]

cfg1, cfg2, cfg3 = st.columns([2, 1, 1])

with cfg1:
    key_cols = st.multiselect(
        "🔑 Key Column(s) — row identifier(s), supports composite keys",
        options=common,
        default=[common[0]] if common else [],
        help="Select one or more columns that uniquely identify a row. e.g. Source + Destination for lane-level data.",
    )

with cfg2:
    val_options = [c for c in numeric_cols if c not in key_cols]
    val_col = st.selectbox(
        "📐 Value Column — numeric metric to compare",
        options=val_options if val_options else common,
        help="The column whose values will be compared between File A and File B.",
    )

with cfg3:
    grp_options = ["(none)"] + [c for c in common if c not in key_cols and c != val_col]
    grp_sel = st.selectbox(
        "🗂 Group By Column — optional segmentation",
        options=grp_options,
        help="Results will be grouped and shown in collapsible sections per unique value in this column.",
    )
    grp_col = None if grp_sel == "(none)" else grp_sel

run = st.button("▶ Run Comparison", type="primary", disabled=not (key_cols and val_col))

if not run and "results" not in st.session_state:
    st.stop()

# ── Step 3: Compute ────────────────────────────────────────────────────────────
if run:
    with st.spinner(f"Comparing {len(df_a)+len(df_b):,} rows…"):
        results = compare(df_a, df_b, key_cols, val_col, grp_col)
    st.session_state["results"]    = results
    st.session_state["key_cols"]   = key_cols
    st.session_state["val_col"]    = val_col
    st.session_state["grp_col"]    = grp_col

results = st.session_state["results"]
key_cols = st.session_state["key_cols"]
val_col  = st.session_state["val_col"]
grp_col  = st.session_state["grp_col"]

# ── Step 3: Display ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 3 · Results")

status_counts = results["Status"].value_counts()
total = len(results)

# Metrics row
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Total Rows",  f"{total:,}")
m2.metric("▲ Degraded",  f"{status_counts.get('Degraded',0):,}")
m3.metric("▼ Improved",  f"{status_counts.get('Improved',0):,}")
m4.metric("● Same",      f"{status_counts.get('Same',0):,}")
m5.metric("✦ New",       f"{status_counts.get('New',0):,}")
m6.metric("✖ Removed",   f"{status_counts.get('Removed',0):,}")

st.markdown("")

# Filter + search controls
fc1, fc2, fc3 = st.columns([2, 2, 1])
with fc1:
    status_filter = st.multiselect(
        "Filter by Status",
        options=["Degraded","Improved","Same","New","Removed"],
        default=[],
        placeholder="All statuses",
    )
with fc2:
    search = st.text_input("🔍 Search key…", placeholder="Type to filter rows")
with fc3:
    sort_by = st.selectbox("Sort by", ["Δ Change", f"{val_col} (A)", f"{val_col} (B)", "Status", " › ".join(key_cols)])

# Apply filters
view = results.copy()
if status_filter:
    view = view[view["Status"].isin(status_filter)]
if search:
    key_display_col = " › ".join(key_cols)
    view = view[view[key_display_col].str.contains(search, case=False, na=False)]
if sort_by in view.columns:
    asc = sort_by not in ["Δ Change"]
    view = view.sort_values(sort_by, ascending=asc, na_position="last")

st.caption(f"Showing {len(view):,} of {total:,} rows")

# Grouped display
groups = view.groupby("Group", sort=True)
results_by_group = {}

for grp_name, grp_df in groups:
    results_by_group[grp_name] = grp_df.copy()
    gc = grp_df["Status"].value_counts()
    badges = "  ".join(
        f'<span style="color:{BADGE_COLORS[s][1]};font-weight:600">{BADGE_COLORS[s][0]} {s}: {gc.get(s,0)}</span>'
        for s in ["Degraded","Improved","Same","New","Removed"] if gc.get(s,0) > 0
    )
    label = f"{'All Rows' if grp_name=='All' else f'{grp_col}: {grp_name}'} — {len(grp_df):,} rows   {badges}"
    with st.expander(label, expanded=(grp_name=="All" or len(groups)==1)):
        show_df = grp_df.drop(columns=["Group"], errors="ignore")
        st.dataframe(
            style_table(show_df),
            use_container_width=True,
            height=min(600, 40 + len(show_df)*36),
        )

# ── Export ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 4 · Export")
ec1, ec2 = st.columns(2)

with ec1:
    csv_bytes = view.drop(columns=["Group"], errors="ignore").to_csv(index=False).encode()
    st.download_button(
        "⬇ Download filtered results (CSV)",
        data=csv_bytes,
        file_name="comparison_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
with ec2:
    excel_bytes = build_excel(results_by_group)
    st.download_button(
        "⬇ Download all groups (Excel, multi-sheet)",
        data=excel_bytes,
        file_name="comparison_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
