import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SLA Comparator", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 95%; }
    div[data-testid="metric-container"] {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 12px 16px;
    }
    /* Smooth transitions for expanders */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
    }
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
    """Clear results when inputs change to prevent stale data views."""
    for key in ["results", "key_cols", "val_col", "grp_col", "higher_is"]:
        if key in st.session_state:
            del st.session_state[key]

# ── Helpers ───────────────────────────────────────────────────────────────────
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

    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip() if hasattr(s, 'str') else s)
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
def load_data(file_bytes: bytes, file_name: str, sheet_name: str = None) -> pd.DataFrame:
    if file_name.lower().endswith('.xlsx'):
        df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(BytesIO(file_bytes), dtype=str, keep_default_na=False, skipinitialspace=True, encoding_errors="replace")
    return clean_df(df)

def sniff_numeric(df: pd.DataFrame, col: str, sample: int = 500) -> bool:
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

def compare(df_a, df_b, col_map, key_cols, val_col, grp_col, higher_is) -> pd.DataFrame:
    df_a = df_a.copy()
    df_b = df_b.copy()

    rename_b = {v: k for k, v in col_map.items()}
    df_b.rename(columns=rename_b, inplace=True)

    def make_key(df, cols):
        return df[cols].astype(str).agg(" › ".join, axis=1) if len(cols) > 1 else df[cols[0]].astype(str)

    df_a["__key__"] = make_key(df_a, key_cols)
    df_b["__key__"] = make_key(df_b, key_cols)

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

    if higher_is == "higher_is_worse":
        deg_cond = merged["_val_B"] > merged["_val_A"]
        imp_cond = merged["_val_B"] < merged["_val_A"]
    else:
        deg_cond = merged["_val_B"] < merged["_val_A"]
        imp_cond = merged["_val_B"] > merged["_val_A"]

    conditions = [
        merged["_val_A"].isna() & merged["_val_B"].notna(),
        merged["_val_A"].notna() & merged["_val_B"].isna(),
        deg_cond, imp_cond,
    ]
    merged["Status"] = np.select(conditions, ["New", "Removed", "Degraded", "Improved"], default="Same")

    if grp_col:
        ga = f"{grp_col}_A" if f"{grp_col}_A" in merged.columns else grp_col
        gb = f"{grp_col}_B" if f"{grp_col}_B" in merged.columns else grp_col
        merged["Group"] = merged.get(ga, pd.Series(dtype=str)).combine_first(merged.get(gb, pd.Series(dtype=str)))
        drop = [c for c in [ga, gb] if c in merged.columns and c != "Group"]
        merged.drop(columns=drop, inplace=True)
    else:
        merged["Group"] = "All"

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

    df = df.copy()
    for c in df.columns:
        if c not in ("Status", "Group") and not c.endswith("(File A)") is False:
            try:
                converted = pd.to_numeric(df[c], errors="coerce")
                if converted.notna().sum() > 0:
                    df[c] = converted
            except Exception: pass

    num_fmt = {c: "{:,.4g}" for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    return df.style.apply(row_bg, axis=1).format(num_fmt, na_rep="—").set_properties(**{"font-size": "13px"})

def build_excel(results: pd.DataFrame, val_col: str) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book
        fmt = {s: wb.add_format({"bg_color": STATUS_META[s]["bg"], "font_size": 11}) for s in STATUS_META}
        hdr_fmt = wb.add_format({"bold": True, "bg_color": "#1e293b", "font_color": "#ffffff", "font_size": 11})

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

        summary_data = []
        for grp, gdf in results.groupby("Group", sort=True):
            sc = gdf["Status"].value_counts()
            avg_a, avg_b = gdf[f"{val_col} (File A)"].mean(), gdf[f"{val_col} (File B)"].mean()
            summary_data.append({
                "Group": grp, "Total Rows": len(gdf),
                **{s: sc.get(s, 0) for s in STATUS_ORDER},
                f"Avg {val_col} (A)": round(avg_a, 4) if not np.isnan(avg_a) else "",
                f"Avg {val_col} (B)": round(avg_b, 4) if not np.isnan(avg_b) else "",
            })
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
        write_sheet(results, "All Results")
        for grp, gdf in results.groupby("Group", sort=True):
            if grp != "All": write_sheet(gdf, str(grp))
    return buf.getvalue()

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📊 SLA Comparator")
st.caption("Compare CSV/Excel datasets seamlessly · Case-insensitive matching · Intelligent status inference")

# ── Step 1: Upload ─────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("#### 1. Upload Datasets")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**📁 File A — Baseline / Previous**")
        up_a = st.file_uploader("File A", type=["csv", "xlsx"], key="fa", label_visibility="collapsed", on_change=reset_computation)
        sheet_a = None
        if up_a and up_a.name.lower().endswith(".xlsx"):
            sheets_a = get_sheet_names(up_a.getvalue())
            sheet_a = st.selectbox("📝 Select Sheet (File A)", sheets_a, on_change=reset_computation)
            
    with c2:
        st.markdown("**📁 File B — Current / New**")
        up_b = st.file_uploader("File B", type=["csv", "xlsx"], key="fb", label_visibility="collapsed", on_change=reset_computation)
        sheet_b = None
        if up_b and up_b.name.lower().endswith(".xlsx"):
            sheets_b = get_sheet_names(up_b.getvalue())
            sheet_b = st.selectbox("📝 Select Sheet (File B)", sheets_b, on_change=reset_computation)

if not (up_a and up_b):
    st.info("⬆ Upload both files (CSV or Excel) to configure your comparison.", icon="ℹ️")
    st.stop()

with st.spinner("Parsing & mapping datasets…"):
    df_a = load_data(up_a.getvalue(), up_a.name, sheet_a)
    df_b = load_data(up_b.getvalue(), up_b.name, sheet_b)

# Case-insensitive column matching
col_map = match_columns(list(df_a.columns), list(df_b.columns))
common    = list(col_map.keys())
only_a    = [c for c in df_a.columns if c not in col_map]
only_b    = [c for c in df_b.columns if c not in col_map.values()]

if not common:
    st.error("No matching columns found between the two files. Please verify headers.")
    st.stop()

# ── Step 2: Configure ──────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("#### 2. Configure Comparison")
    
    numeric_cols = [c for c in common if sniff_numeric(df_a, c) or sniff_numeric(df_b, c)]
    
    cfg1, cfg2, cfg3 = st.columns([2, 1, 1])
    with cfg1:
        key_cols = st.multiselect("🔑 Unique Identifier(s)", options=common, default=[common[0]] if common else [],
                                  help="E.g., Ticket ID, Server Name. Supports multiple columns.", on_change=reset_computation)
    with cfg2:
        val_options = [c for c in numeric_cols if c not in key_cols] or [c for c in common if c not in key_cols]
        val_col = st.selectbox("📐 Metric to Compare", options=val_options, on_change=reset_computation)
    with cfg3:
        grp_options = ["(none)"] + [c for c in common if c not in key_cols and c != val_col]
        grp_sel = st.selectbox("🗂 Group By (Optional)", options=grp_options, on_change=reset_computation)
        grp_col = None if grp_sel == "(none)" else grp_sel

    auto_dir   = infer_direction(val_col) if val_col else "higher_is_worse"
    higher_is  = st.radio(
        "📈 Value direction meaning",
        options=["higher_is_worse", "higher_is_better"],
        index=0 if auto_dir == "higher_is_worse" else 1,
        format_func=lambda x: "⬆ Higher = Worse (e.g., Latency, Days)" if x == "higher_is_worse" else "⬆ Higher = Better (e.g., Score, Resolution %)",
        horizontal=True, on_change=reset_computation
    )

    run = st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=not (key_cols and val_col))

if not run and "results" not in st.session_state:
    with st.expander("🔍 View Column Mapping Details"):
        oc1, oc2, oc3 = st.columns(3)
        oc1.markdown(f"**✅ Matched ({len(common)})**\n\n" + "\n".join(f"- `{a}` ↔ `{col_map[a]}`" for a in common))
        oc2.markdown(f"**⚠ Only in {up_a.name} ({len(only_a)})**\n\n" + ("\n".join(f"- `{c}`" for c in only_a) or "_None_"))
        oc3.markdown(f"**✦ Only in {up_b.name} ({len(only_b)})**\n\n" + ("\n".join(f"- `{c}`" for c in only_b) or "_None_"))
    st.stop()

# ── Step 3: Compute ────────────────────────────────────────────────────────────
if run:
    with st.spinner(f"Analyzing {len(df_a) + len(df_b):,} total rows..."):
        results = compare(df_a, df_b, col_map, key_cols, val_col, grp_col, higher_is)
    st.session_state.update({"results": results, "key_cols": key_cols, "val_col": val_col, "grp_col": grp_col, "higher_is": higher_is})

# Pull from session state
results   = st.session_state["results"]
key_cols  = st.session_state["key_cols"]
val_col   = st.session_state["val_col"]
grp_col   = st.session_state["grp_col"]
higher_is = st.session_state["higher_is"]

# ── Step 4: Display Results ────────────────────────────────────────────────────
st.markdown("---")
sc = results["Status"].value_counts()
cols_m = st.columns(6)
cols_m[0].metric("Total Rows Evaluated", f"{len(results):,}")
for i, s in enumerate(STATUS_ORDER):
    cols_m[i+1].metric(f"{STATUS_META[s]['icon']} {s}", f"{sc.get(s, 0):,}")

st.markdown("")

# Using tabs for a cleaner, app-like view
tab_data, tab_export = st.tabs(["📋 Detailed Data Viewer", "💾 Export Reports"])

with tab_data:
    # Inline Filters
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    status_filter = fc1.multiselect("Filter by Status", STATUS_ORDER, default=[], placeholder="All statuses")
    search = fc2.text_input("🔍 Search in Key", placeholder="Type to filter...")
    sort_opts = ["Δ Change", f"{val_col} (File A)", f"{val_col} (File B)", "Status", " › ".join(key_cols)]
    sort_by = fc3.selectbox("Sort Data By", sort_opts)

    view = results.copy()
    if status_filter: view = view[view["Status"].isin(status_filter)]
    if search: 
        key_display = " › ".join(key_cols)
        view = view[view[key_display].astype(str).str.contains(search, case=False, na=False)]
    if sort_by in view.columns: 
        view = view.sort_values(sort_by, ascending=(sort_by == "Status"), na_position="last")

    st.caption(f"Showing **{len(view):,}** of **{len(results):,}** rows across **{view['Group'].nunique()}** group(s)")

    for grp_name, grp_df in view.groupby("Group", sort=True):
        gc = grp_df["Status"].value_counts()
        badges = "  ".join(f"{STATUS_META[s]['icon']} {s}: {gc.get(s,0)}" for s in STATUS_ORDER if gc.get(s, 0) > 0)
        grp_label = "Overall Dataset" if grp_name == "All" else f"{grp_col}: {grp_name}"
        
        with st.expander(f"{grp_label} ({len(grp_df):,} rows)  |  {badges}", expanded=(grp_name == "All" or view["Group"].nunique() <= 2)):
            show = grp_df.drop(columns=["Group"], errors="ignore")
            st.dataframe(style_table(show), use_container_width=True, height=min(500, 45 + len(show) * 36))

with tab_export:
    st.markdown("#### Download your insights")
    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button(
            "⬇️ Download Filtered View (CSV)",
            data=view.drop(columns=["Group"], errors="ignore").to_csv(index=False).encode(),
            file_name=f"SLA_filtered_{val_col}.csv",
            mime="text/csv", use_container_width=True
        )
    with ec2:
        st.download_button(
            "⬇️ Download Full Multi-Sheet Report (Excel)",
            data=build_excel(results, val_col),
            file_name=f"SLA_Full_Report_{val_col}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, type="primary"
        )
