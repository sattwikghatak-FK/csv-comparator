import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CSV Comparator", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="metric-container"] {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Status config — fully dynamic, no domain-specific keywords ────────────────
STATUS_META = {
    "Degraded" : {"bg": "#fef2f2", "icon": "▲", "color": "#dc2626"},
    "Improved" : {"bg": "#f0fdf4", "icon": "▼", "color": "#16a34a"},
    "Same"     : {"bg": "#f8fafc", "icon": "●", "color": "#6b7280"},
    "New"      : {"bg": "#eff6ff", "icon": "✦", "color": "#2563eb"},
    "Removed"  : {"bg": "#fff7ed", "icon": "✖", "color": "#ea580c"},
}
STATUS_ORDER = ["Degraded", "Improved", "Same", "New", "Removed"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a raw parsed DataFrame:
    - Strip whitespace from column names
    - Deduplicate column names (append _1, _2 …)
    - Drop fully empty rows and columns
    - Strip leading/trailing whitespace from all string cells
    - Normalise column names to Title Case internally for matching,
      but keep a mapping back to originals for display
    """
    # Strip column name whitespace
    df.columns = [str(c).strip() for c in df.columns]

    # Deduplicate columns
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

    # Drop fully empty rows/cols
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # Strip string cell whitespace
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # Replace empty strings with NaN so we can handle missingness uniformly
    df.replace("", np.nan, inplace=True)

    return df


def normalise_col_name(name: str) -> str:
    """Lowercase + strip for case-insensitive matching."""
    return str(name).strip().lower()


def match_columns(cols_a: list, cols_b: list) -> dict:
    """
    Return a mapping  original_col_a -> original_col_b
    for columns that match case-insensitively.
    Uses the File A column name as the canonical display name.
    """
    norm_b = {normalise_col_name(c): c for c in cols_b}
    mapping = {}
    for ca in cols_a:
        nb = norm_b.get(normalise_col_name(ca))
        if nb is not None:
            mapping[ca] = nb
    return mapping   # {col_in_A: col_in_B}


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(
        BytesIO(file_bytes),
        dtype=str,
        keep_default_na=False,
        skipinitialspace=True,
        encoding_errors="replace",
    )
    return clean_df(df)


def sniff_numeric(df: pd.DataFrame, col: str, sample: int = 500) -> bool:
    """Return True if >60 % of sampled non-null values parse as numbers."""
    vals = df[col].dropna().head(sample)
    if vals.empty:
        return False
    return pd.to_numeric(vals, errors="coerce").notna().mean() > 0.6


def infer_direction(val_col: str) -> str:
    """
    Try to infer whether a value increase is 'good' or 'bad'
    based on the column name. Defaults to: higher = worse (Degraded).
    Users can override via the UI toggle.
    """
    name = normalise_col_name(val_col)
    positive_keywords = ["score", "rating", "accuracy", "fill", "efficiency",
                         "utilisation", "utilization", "revenue", "profit",
                         "coverage", "success", "satisfaction", "nps"]
    for kw in positive_keywords:
        if kw in name:
            return "higher_is_better"
    return "higher_is_worse"


def compare(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    col_map: dict,          # {col_in_A: col_in_B}
    key_cols: list,         # column names as they appear in df_a
    val_col: str,           # column name as it appears in df_a
    grp_col: str | None,
    higher_is: str,         # "higher_is_worse" | "higher_is_better"
) -> pd.DataFrame:

    df_a = df_a.copy()
    df_b = df_b.copy()

    # Rename df_b columns to match df_a names (case-insensitive resolved)
    rename_b = {v: k for k, v in col_map.items()}
    df_b.rename(columns=rename_b, inplace=True)

    # Build composite key
    def make_key(df, cols):
        return df[cols].astype(str).agg(" › ".join, axis=1) if len(cols) > 1 else df[cols[0]].astype(str)

    df_a["__key__"] = make_key(df_a, key_cols)
    df_b["__key__"] = make_key(df_b, key_cols)

    # Retain only needed columns
    need = ["__key__", val_col] + ([grp_col] if grp_col else [])
    df_a = df_a[[c for c in need if c in df_a.columns]].drop_duplicates("__key__")
    df_b = df_b[[c for c in need if c in df_b.columns]].drop_duplicates("__key__")

    merged = pd.merge(
        df_a.rename(columns={val_col: "_val_A"}),
        df_b.rename(columns={val_col: "_val_B"}),
        on="__key__", how="outer", suffixes=("_A", "_B"),
    )

    merged["_val_A"] = pd.to_numeric(merged["_val_A"], errors="coerce")
    merged["_val_B"] = pd.to_numeric(merged["_val_B"], errors="coerce")
    merged["Δ Change"] = merged["_val_B"] - merged["_val_A"]

    # Status — direction-aware
    if higher_is == "higher_is_worse":
        deg_cond = merged["_val_B"] > merged["_val_A"]
        imp_cond = merged["_val_B"] < merged["_val_A"]
    else:
        deg_cond = merged["_val_B"] < merged["_val_A"]
        imp_cond = merged["_val_B"] > merged["_val_A"]

    conditions = [
        merged["_val_A"].isna() & merged["_val_B"].notna(),
        merged["_val_A"].notna() & merged["_val_B"].isna(),
        deg_cond,
        imp_cond,
    ]
    merged["Status"] = np.select(conditions, ["New", "Removed", "Degraded", "Improved"], default="Same")

    # Group column — coalesce A/B sides
    if grp_col:
        ga = f"{grp_col}_A" if f"{grp_col}_A" in merged.columns else grp_col
        gb = f"{grp_col}_B" if f"{grp_col}_B" in merged.columns else grp_col
        merged["Group"] = (
            merged.get(ga, pd.Series(dtype=str))
            .combine_first(merged.get(gb, pd.Series(dtype=str)))
        )
        drop = [c for c in [ga, gb] if c in merged.columns and c != "Group"]
        merged.drop(columns=drop, inplace=True)
    else:
        merged["Group"] = "All"

    # Friendly column names derived from actual column name (no hardcoding)
    key_display = " › ".join(key_cols)
    merged.rename(columns={
        "__key__": key_display,
        "_val_A": f"{val_col} (File A)",
        "_val_B": f"{val_col} (File B)",
    }, inplace=True)

    final_cols = [key_display, f"{val_col} (File A)", f"{val_col} (File B)", "Δ Change", "Status", "Group"]
    return merged[[c for c in final_cols if c in merged.columns]]


def style_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_bg(row):
        return [f"background-color:{STATUS_META.get(row['Status'],{}).get('bg','#fff')}"] * len(row)
    num_fmt = {c: "{:,.4g}" for c in df.columns
               if pd.api.types.is_float_dtype(df[c]) or c == "Δ Change"}
    return (
        df.style
        .apply(row_bg, axis=1)
        .format(num_fmt, na_rep="—")
        .set_properties(**{"font-size": "12px"})
    )


def build_excel(results: pd.DataFrame, val_col: str) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book

        # Formats
        fmt = {s: wb.add_format({"bg_color": STATUS_META[s]["bg"], "font_size": 11})
               for s in STATUS_META}
        hdr_fmt = wb.add_format({"bold": True, "bg_color": "#1e293b",
                                  "font_color": "#ffffff", "font_size": 11})

        def write_sheet(df, sheet_name):
            df = df.drop(columns=["Group"], errors="ignore")
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            ws = writer.sheets[sheet_name[:31]]
            for ci, col in enumerate(df.columns):
                ws.write(0, ci, col, hdr_fmt)
                ws.set_column(ci, ci, max(18, len(str(col)) + 4))
            status_ci = list(df.columns).index("Status") if "Status" in df.columns else None
            if status_ci is not None:
                for ri, status in enumerate(df["Status"], start=1):
                    cell_fmt = fmt.get(status)
                    if cell_fmt:
                        for ci in range(len(df.columns)):
                            val = df.iloc[ri-1, ci]
                            ws.write(ri, ci, "" if pd.isna(val) else val, cell_fmt)

        # Summary sheet
        summary_data = []
        for grp, gdf in results.groupby("Group", sort=True):
            sc = gdf["Status"].value_counts()
            avg_a = gdf[f"{val_col} (File A)"].mean()
            avg_b = gdf[f"{val_col} (File B)"].mean()
            summary_data.append({
                "Group": grp,
                "Total Rows": len(gdf),
                **{s: sc.get(s, 0) for s in STATUS_ORDER},
                f"Avg {val_col} (File A)": round(avg_a, 4) if not np.isnan(avg_a) else "",
                f"Avg {val_col} (File B)": round(avg_b, 4) if not np.isnan(avg_b) else "",
            })
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

        # All results
        write_sheet(results, "All Results")

        # Per-group sheets
        for grp, gdf in results.groupby("Group", sort=True):
            if grp != "All":
                write_sheet(gdf, str(grp))

    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📊 Universal CSV Comparator")
st.caption("Any two CSVs · Case-insensitive column matching · Composite keys · Grouped view · Direction-aware · Excel export")

# ── Step 1: Upload ─────────────────────────────────────────────────────────────
st.markdown("### 1 · Upload Files")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**📁 File A — Baseline / Previous**")
    up_a = st.file_uploader("File A", type="csv", key="fa", label_visibility="collapsed")
with c2:
    st.markdown("**📁 File B — Current / New**")
    up_b = st.file_uploader("File B", type="csv", key="fb", label_visibility="collapsed")

if not (up_a and up_b):
    st.info("⬆ Upload both CSV files to get started.", icon="ℹ️")
    st.stop()

with st.spinner("Parsing & cleaning files…"):
    df_a = load_csv(up_a.read())
    df_b = load_csv(up_b.read())

c1.success(f"**{up_a.name}** — {len(df_a):,} rows · {len(df_a.columns)} cols")
c2.success(f"**{up_b.name}** — {len(df_b):,} rows · {len(df_b.columns)} cols")

# Case-insensitive column matching
col_map = match_columns(list(df_a.columns), list(df_b.columns))
common    = list(col_map.keys())           # display names = File A names
only_a    = [c for c in df_a.columns if c not in col_map]
only_b    = [c for c in df_b.columns if c not in col_map.values()]

with st.expander("📋 Column overview (case-insensitive match)", expanded=True):
    oc1, oc2, oc3 = st.columns(3)
    oc1.markdown(f"**✅ Matched ({len(common)})**\n\n" +
                 "\n".join(f"- `{a}` ↔ `{col_map[a]}`" for a in common))
    oc2.markdown(f"**⚠ Only in File A ({len(only_a)})**\n\n" +
                 ("\n".join(f"- `{c}`" for c in only_a) or "_None_"))
    oc3.markdown(f"**✦ Only in File B ({len(only_b)})**\n\n" +
                 ("\n".join(f"- `{c}`" for c in only_b) or "_None_"))

if not common:
    st.error("No matching columns found between the two files (case-insensitive check). Verify the headers.")
    st.stop()

# ── Step 2: Configure ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 2 · Configure Comparison")

numeric_cols    = [c for c in common if sniff_numeric(df_a, c) or sniff_numeric(df_b, c)]
non_numeric_cols= [c for c in common if c not in numeric_cols]

cfg1, cfg2, cfg3 = st.columns([2, 1, 1])

with cfg1:
    key_cols = st.multiselect(
        "🔑 Key Column(s) — unique row identifier(s)",
        options=common,
        default=[common[0]] if common else [],
        help="Supports composite keys — select multiple columns. e.g. Source + Destination",
    )

with cfg2:
    val_options = [c for c in numeric_cols if c not in key_cols]
    if not val_options:
        val_options = [c for c in common if c not in key_cols]
    val_col = st.selectbox(
        "📐 Value Column — metric to compare",
        options=val_options,
        help="Only columns detected as numeric are listed first.",
    )

with cfg3:
    grp_options = ["(none)"] + [c for c in common if c not in key_cols and c != val_col]
    grp_sel = st.selectbox("🗂 Group By — optional", options=grp_options)
    grp_col = None if grp_sel == "(none)" else grp_sel

# Direction toggle — auto-detected but overridable
auto_dir   = infer_direction(val_col) if val_col else "higher_is_worse"
dir_label  = "Higher value = Worse (e.g. SLA days, Cost, TAT)" if auto_dir == "higher_is_worse" \
             else "Higher value = Better (e.g. Score, Revenue, Fill rate)"
higher_is  = st.radio(
    "📈 Value direction",
    options=["higher_is_worse", "higher_is_better"],
    index=0 if auto_dir == "higher_is_worse" else 1,
    format_func=lambda x: "⬆ Higher = Worse (Degraded if value rises)" if x == "higher_is_worse"
                          else "⬆ Higher = Better (Improved if value rises)",
    horizontal=True,
    help=f"Auto-detected from column name '{val_col}': {dir_label}",
)

run = st.button("▶ Run Comparison", type="primary", disabled=not (key_cols and val_col))

if not run and "results" not in st.session_state:
    st.stop()

# ── Step 3: Compute ────────────────────────────────────────────────────────────
if run:
    with st.spinner(f"Comparing {len(df_a) + len(df_b):,} rows…"):
        results = compare(df_a, df_b, col_map, key_cols, val_col, grp_col, higher_is)
    st.session_state.update({
        "results": results, "key_cols": key_cols,
        "val_col": val_col, "grp_col": grp_col, "higher_is": higher_is,
    })

results  = st.session_state["results"]
key_cols = st.session_state["key_cols"]
val_col  = st.session_state["val_col"]
grp_col  = st.session_state["grp_col"]
higher_is= st.session_state["higher_is"]

# ── Step 3: Display ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 3 · Results")
st.caption(
    f"Key: **{' + '.join(key_cols)}** · Value: **{val_col}** "
    f"{'· Group: **' + grp_col + '**' if grp_col else ''} · "
    f"Direction: **{'Higher = Worse' if higher_is == 'higher_is_worse' else 'Higher = Better'}**"
)

sc = results["Status"].value_counts()
cols_m = st.columns(6)
cols_m[0].metric("Total",      f"{len(results):,}")
for i, s in enumerate(STATUS_ORDER):
    cols_m[i+1].metric(f"{STATUS_META[s]['icon']} {s}", f"{sc.get(s, 0):,}")

st.markdown("")

# Filters
fc1, fc2, fc3 = st.columns([2, 2, 1])
with fc1:
    status_filter = st.multiselect("Filter by Status", STATUS_ORDER, default=[], placeholder="All statuses")
with fc2:
    search = st.text_input("🔍 Search in key…", placeholder="Type any value")
with fc3:
    sort_opts = ["Δ Change", f"{val_col} (File A)", f"{val_col} (File B)", "Status", " › ".join(key_cols)]
    sort_by = st.selectbox("Sort by", sort_opts)

view = results.copy()
if status_filter:
    view = view[view["Status"].isin(status_filter)]
if search:
    key_display = " › ".join(key_cols)
    view = view[view[key_display].astype(str).str.contains(search, case=False, na=False)]
if sort_by in view.columns:
    view = view.sort_values(sort_by, ascending=(sort_by == "Status"), na_position="last")

st.caption(f"Showing **{len(view):,}** of **{len(results):,}** rows across "
           f"**{view['Group'].nunique()}** group(s)")

# Grouped display
results_by_group = {}
for grp_name, grp_df in view.groupby("Group", sort=True):
    results_by_group[grp_name] = grp_df.copy()
    gc = grp_df["Status"].value_counts()
    badges = "  ".join(
        f'<span style="color:{STATUS_META[s]["color"]};font-weight:700">'
        f'{STATUS_META[s]["icon"]} {s}: {gc.get(s,0)}</span>'
        for s in STATUS_ORDER if gc.get(s, 0) > 0
    )
    grp_label = "All Rows" if grp_name == "All" else f"{grp_col}: {grp_name}"
    expander_label = f"{grp_label} — {len(grp_df):,} rows   {badges}"
    with st.expander(expander_label, expanded=(grp_name == "All" or view["Group"].nunique() == 1)):
        show = grp_df.drop(columns=["Group"], errors="ignore")
        st.dataframe(
            style_table(show),
            use_container_width=True,
            height=min(600, 45 + len(show) * 36),
        )

# ── Step 4: Export ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 4 · Export")
ec1, ec2 = st.columns(2)
with ec1:
    st.download_button(
        "⬇ Download filtered results (CSV)",
        data=view.drop(columns=["Group"], errors="ignore").to_csv(index=False).encode(),
        file_name=f"comparison_{val_col}_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
with ec2:
    st.download_button(
        "⬇ Download full report (Excel, multi-sheet)",
        data=build_excel(results, val_col),
        file_name=f"comparison_{val_col}_full.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
