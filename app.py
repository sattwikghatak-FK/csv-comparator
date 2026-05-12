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

STATUS_META = {
    "Degraded" : {"bg": "#fef2f2", "icon": "▲", "color": "#dc2626"},
    "Improved" : {"bg": "#f0fdf4", "icon": "▼", "color": "#16a34a"},
    "Same"     : {"bg": "#f8fafc", "icon": "●", "color": "#6b7280"},
    "New"      : {"bg": "#eff6ff", "icon": "✦", "color": "#2563eb"},
    "Removed"  : {"bg": "#fff7ed", "icon": "✖", "color": "#ea580c"},
}
STATUS_ORDER = ["Degraded", "Improved", "Same", "New", "Removed"]

def reset_computation():
    for key in ["results", "key_cols", "val_col", "grp_col", "higher_is"]:
        st.session_state.pop(key, None)

# ─────────────────────────────────────────────────────────────────────────────
# FILE BYTES CACHE — the core fix
#
# Problem: @st.cache_data hashes ALL arguments on every call to check for a
# cache hit.  Hashing a 50 MB bytes object takes ~100–200 ms.  With files
# uploaded, EVERY widget interaction (multiselect, selectbox, radio) triggers
# a full script rerun, and each rerun re-hashes the bytes → that IS the lag.
#
# Solution — two-layer approach:
#   Layer 1: session_state stores raw bytes keyed by (name, size).
#            get_file_bytes() reads from disk ONLY when the file actually
#            changes; all other reruns return the cached bytes in ~0 ms.
#
#   Layer 2: @st.cache_data functions receive bytes via a _-prefixed param.
#            Streamlit SKIPS hashing for parameters whose name starts with _.
#            The actual cache key is just (file_name, file_size, sheet_name)
#            — a tiny tuple that hashes in microseconds.
#
# Net result: cache lookup is instant on every rerun; file I/O only on upload.
# ─────────────────────────────────────────────────────────────────────────────
def get_file_bytes(uploaded_file, ss_key: str) -> bytes:
    file_id = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get(f"_fid_{ss_key}") != file_id:
        uploaded_file.seek(0)
        st.session_state[f"_fbytes_{ss_key}"] = uploaded_file.read()
        st.session_state[f"_fid_{ss_key}"]    = file_id
    return st.session_state[f"_fbytes_{ss_key}"]


@st.cache_data(show_spinner=False)
def _cached_sheet_names(file_name: str, file_size: int, _file_bytes: bytes) -> list:
    return pd.ExcelFile(BytesIO(_file_bytes)).sheet_names


@st.cache_data(show_spinner=False)
def _cached_preview(file_name: str, file_size: int, sheet_name,
                    _file_bytes: bytes) -> pd.DataFrame:
    buf = BytesIO(_file_bytes)
    df = (pd.read_excel(buf, sheet_name=sheet_name, dtype=str,
                        keep_default_na=False, nrows=2000)
          if file_name.lower().endswith(".xlsx")
          else pd.read_csv(buf, dtype=str, keep_default_na=False,
                           skipinitialspace=True, encoding_errors="replace", nrows=2000))
    return _clean_df(df)


@st.cache_data(show_spinner=False)
def _cached_sniff(file_name: str, file_size: int, sheet_name,
                  _file_bytes: bytes) -> dict:
    df = _cached_preview(file_name, file_size, sheet_name, _file_bytes)
    out = {}
    for col in df.columns:
        vals = df[col].dropna().head(1000)
        out[col] = (not vals.empty and
                    pd.to_numeric(vals, errors="coerce").notna().mean() > 0.6)
    return out


# ── Public wrappers (pass bytes as unhashed _param) ───────────────────────────
def get_sheet_names(f) -> list:
    if not f: return []
    return _cached_sheet_names(f.name, f.size, get_file_bytes(f, f.name))

def get_preview_data(f, sheet=None) -> pd.DataFrame:
    return _cached_preview(f.name, f.size, sheet, get_file_bytes(f, f.name))

def get_sniff_map(f, sheet=None) -> dict:
    return _cached_sniff(f.name, f.size, sheet, get_file_bytes(f, f.name))

def load_full_data(f, sheet=None) -> pd.DataFrame:
    buf = BytesIO(get_file_bytes(f, f.name))
    df = (pd.read_excel(buf, sheet_name=sheet, dtype=str, keep_default_na=False)
          if f.name.lower().endswith(".xlsx")
          else pd.read_csv(buf, dtype=str, keep_default_na=False,
                           skipinitialspace=True, encoding_errors="replace"))
    return _clean_df(df)


# ── Data helpers ──────────────────────────────────────────────────────────────
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    seen, new_cols = {}, []
    for c in df.columns:
        if c in seen:
            seen[c] += 1; new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0; new_cols.append(c)
    df.columns = new_cols
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    for c in df.select_dtypes(include=["object","string"]).columns:
        df[c] = df[c].str.strip()
    df.replace("", np.nan, inplace=True)
    return df

def match_columns(cols_a, cols_b) -> dict:
    norm_b = {str(c).strip().lower(): c for c in cols_b}
    return {ca: norm_b[str(ca).strip().lower()]
            for ca in cols_a if str(ca).strip().lower() in norm_b}

def infer_direction(val_col: str) -> str:
    name = val_col.strip().lower()
    for kw in ["score","rating","accuracy","fill","efficiency","utilisation",
               "utilization","revenue","profit","coverage","success","satisfaction","nps"]:
        if kw in name: return "higher_is_better"
    return "higher_is_worse"

def make_key(df, cols):
    res = df[cols[0]].astype(str)
    for c in cols[1:]: res = res + " › " + df[c].astype(str)
    return res

# ── Chunk processing engine ───────────────────────────────────────────────────
def process_comparison_chunked(df_a, df_b, comp_mode, granular_file, col_map,
                                key_cols, val_col, grp_col, higher_is,
                                status_text=None, progress_bar=None) -> pd.DataFrame:
    def ui(msg, pct):
        if status_text:  status_text.markdown(f"**⏳ {msg}**")
        if progress_bar: progress_bar.progress(pct)

    df_a, df_b = df_a.copy(), df_b.copy()
    ui("Aligning columns...", 55)
    df_b.rename(columns={v: k for k, v in col_map.items()}, inplace=True)
    for k in key_cols:
        if k not in df_a.columns: df_a[k] = "MISSING_IN_A"
        if k not in df_b.columns: df_b[k] = "MISSING_IN_B"

    cols_a_all = list(df_a.columns)
    cols_b_all = list(df_b.columns)
    common_cols = [c for c in cols_a_all if c in cols_b_all]
    only_a = [c for c in cols_a_all if c not in cols_b_all]
    only_b = [c for c in cols_b_all if c not in cols_a_all]

    ui("Generating composite keys...", 60)
    df_a["__key__"] = make_key(df_a, key_cols)
    df_b["__key__"] = make_key(df_b, key_cols)

    all_keys = pd.unique(pd.concat([df_a["__key__"], df_b["__key__"]]))
    CHUNK = 50_000
    num_chunks = max(1, -(-len(all_keys) // CHUNK))

    is_strict = "1-to-1" in comp_mode
    dedup_a = dedup_b = True
    if not is_strict:
        if granular_file and "File B" in granular_file: dedup_b = False
        else: dedup_a = False

    chunks = []
    for i in range(num_chunks):
        ui(f"Chunk {i+1}/{num_chunks} ({i*CHUNK:,}–{min((i+1)*CHUNK, len(all_keys)):,} keys)...",
           65 + int(25 * i / num_chunks))
        ck = all_keys[i*CHUNK:(i+1)*CHUNK]
        sa = df_a[df_a["__key__"].isin(ck)].copy()
        sb = df_b[df_b["__key__"].isin(ck)].copy()
        if dedup_a: sa = sa.drop_duplicates("__key__")
        if dedup_b: sb = sb.drop_duplicates("__key__")

        m = pd.merge(sa.rename(columns={val_col: "_val_A"}),
                     sb.rename(columns={val_col: "_val_B"}),
                     on="__key__", how="outer", suffixes=("_A","_B"))
        m["_val_A"] = pd.to_numeric(m["_val_A"], errors="coerce")
        m["_val_B"] = pd.to_numeric(m["_val_B"], errors="coerce")
        m["Δ Change"] = m["_val_B"] - m["_val_A"]

        deg = m["_val_B"] > m["_val_A"] if higher_is == "higher_is_worse" else m["_val_B"] < m["_val_A"]
        imp = m["_val_B"] < m["_val_A"] if higher_is == "higher_is_worse" else m["_val_B"] > m["_val_A"]
        m["Status"] = np.select(
            [m["_val_A"].isna() & m["_val_B"].notna(),
             m["_val_A"].notna() & m["_val_B"].isna(), deg, imp],
            ["New","Removed","Degraded","Improved"], default="Same")

        for c in [c for c in common_cols if c not in ["__key__", val_col]]:
            ca, cb = f"{c}_A", f"{c}_B"
            if ca in m.columns and cb in m.columns:
                src = cb if (not is_strict and granular_file and "File B" in granular_file) else ca
                alt = ca if src == cb else cb
                m[c] = m[src].combine_first(m[alt])
                m.drop(columns=[ca, cb], inplace=True)
            elif ca in m.columns: m.rename(columns={ca: c}, inplace=True)
            elif cb in m.columns: m.rename(columns={cb: c}, inplace=True)

        if not m.empty: chunks.append(m)
        del sa, sb, m; gc.collect()

    ui("Assembling final dataset...", 95)
    final = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    final["Group"] = (final[grp_col].fillna("Unknown")
                      if grp_col and grp_col in final.columns else "All")

    kd = " › ".join(key_cols)
    final.rename(columns={"__key__": kd,
                           "_val_A": f"{val_col} (File A)",
                           "_val_B": f"{val_col} (File B)"}, inplace=True)
    base = [kd, f"{val_col} (File A)", f"{val_col} (File B)", "Δ Change", "Status", "Group"]
    ctx  = list(dict.fromkeys([c for c in (common_cols + only_a + only_b)
                                if c not in base and c not in ["__key__", val_col]
                                and c in final.columns]))
    if not is_strict: final = final.drop_duplicates()
    ui("Analysis Complete! 🚀", 100)
    return final[[c for c in base + ctx if c in final.columns]]

# ── Styler ────────────────────────────────────────────────────────────────────
def style_table(df: pd.DataFrame):
    def row_style(row):
        bg = STATUS_META.get(row["Status"], {}).get("bg", "#fff")
        fg = STATUS_META.get(row["Status"], {}).get("color", "#1e293b")
        return [f"background-color:{bg};color:{fg}"] * len(row)
    df = df.copy()
    for c in df.columns:
        if "File A" in c or "File B" in c or c == "Δ Change":
            try:
                conv = pd.to_numeric(df[c], errors="coerce")
                if conv.notna().sum() > 0: df[c] = conv
            except Exception: pass
    num_fmt = {c: "{:,.4g}" for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    return (df.style.apply(row_style, axis=1)
              .format(num_fmt, na_rep="—")
              .set_properties(**{"font-size":"13px","font-weight":"500"}))

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📊 SLA Comparator")
st.caption("Compare massive CSV/Excel datasets seamlessly · Chunk Processing Engine Enabled")

with st.container(border=True):
    st.markdown("#### 1. Upload Datasets")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📁 File A — Baseline / Previous**")
        up_a = st.file_uploader("File A", type=["csv","xlsx"], key="fa",
                                 label_visibility="collapsed", on_change=reset_computation)
        sheet_a = (st.selectbox("📝 Sheet (File A)", get_sheet_names(up_a),
                                 on_change=reset_computation)
                   if up_a and up_a.name.lower().endswith(".xlsx") else None)
    with c2:
        st.markdown("**📁 File B — Current / New**")
        up_b = st.file_uploader("File B", type=["csv","xlsx"], key="fb",
                                 label_visibility="collapsed", on_change=reset_computation)
        sheet_b = (st.selectbox("📝 Sheet (File B)", get_sheet_names(up_b),
                                 on_change=reset_computation)
                   if up_b and up_b.name.lower().endswith(".xlsx") else None)

if not (up_a and up_b):
    st.info("⬆ Upload both files to configure your comparison.", icon="ℹ️")
    st.stop()

# Instant after first upload — cache key is (name, size, sheet), not bytes
with st.spinner("Extracting headers and detecting column types..."):
    df_a_prev = get_preview_data(up_a, sheet_a)
    df_b_prev = get_preview_data(up_b, sheet_b)
    sniff_a   = get_sniff_map(up_a, sheet_a)
    sniff_b   = get_sniff_map(up_b, sheet_b)

col_map = match_columns(list(df_a_prev.columns), list(df_b_prev.columns))
common  = list(col_map.keys())

if not common:
    st.error("No matching columns found between the two files. Please verify headers.")
    st.stop()

numeric_cols = [c for c in common if sniff_a.get(c) or sniff_b.get(col_map.get(c, ""))]

with st.container(border=True):
    st.markdown("#### 2. Configure Comparison")
    mode_c1, mode_c2 = st.columns(2)
    comp_mode = mode_c1.radio("⚙️ Match Architecture",
                               ["Strict 1-to-1 (Deduplicate Both)",
                                "1-to-Many (Broadcast granular rows)"],
                               index=0, on_change=reset_computation)
    granular_file = None
    if "1-to-Many" in comp_mode:
        granular_file = mode_c2.selectbox(
            "📌 Which file is granular? (keep its duplicates)",
            ["File B (Current/New)", "File A (Baseline/Previous)"],
            on_change=reset_computation)
    st.markdown("---")

    cfg1, cfg2, cfg3 = st.columns([2, 1, 1])
    with cfg1:
        key_cols = st.multiselect("🔑 Unique Identifier(s)", options=common,
                                   default=[common[0]] if common else [],
                                   on_change=reset_computation)
    with cfg2:
        val_opts = ([c for c in numeric_cols if c not in key_cols]
                    or [c for c in common if c not in key_cols])
        val_col  = st.selectbox("📐 Metric to Compare", options=val_opts,
                                 on_change=reset_computation)
    with cfg3:
        grp_sel = st.selectbox("🗂 Group By (Optional)",
                                ["(none)"] + [c for c in common if c != val_col],
                                on_change=reset_computation)
        grp_col = None if grp_sel == "(none)" else grp_sel

    higher_is = st.radio(
        "📈 Value direction meaning",
        ["higher_is_worse", "higher_is_better"],
        index=0 if infer_direction(val_col) == "higher_is_worse" else 1,
        format_func=lambda x: "⬆ Higher = Worse (e.g., Latency)"
                               if x == "higher_is_worse" else "⬆ Higher = Better (e.g., Score)",
        horizontal=True, on_change=reset_computation)

    run = st.button("🚀 Run Full Analysis", type="primary",
                    disabled=not (key_cols and val_col))

if not run and "results" not in st.session_state:
    st.stop()

if run:
    st.markdown("---")
    status_text  = st.empty()
    progress_bar = st.progress(0)

    status_text.markdown(f"**⏳ Reading {up_a.name} into memory...**"); progress_bar.progress(10)
    df_a_full = load_full_data(up_a, sheet_a)

    status_text.markdown(f"**⏳ Reading {up_b.name} into memory...**"); progress_bar.progress(35)
    df_b_full = load_full_data(up_b, sheet_b)

    results = process_comparison_chunked(
        df_a_full, df_b_full, comp_mode, granular_file, col_map,
        key_cols, val_col, grp_col, higher_is, status_text, progress_bar)

    st.session_state.update({"results": results, "key_cols": key_cols,
                              "val_col": val_col, "grp_col": grp_col, "higher_is": higher_is})
    status_text.empty(); progress_bar.empty()

results   = st.session_state["results"]
key_cols  = st.session_state["key_cols"]
val_col   = st.session_state["val_col"]
grp_col   = st.session_state["grp_col"]
higher_is = st.session_state["higher_is"]

sc = results["Status"].value_counts()
cols_m = st.columns(6)
cols_m[0].metric("Total Rows Evaluated", f"{len(results):,}")
for i, s in enumerate(STATUS_ORDER):
    cols_m[i+1].metric(f"{STATUS_META[s]['icon']} {s}", f"{sc.get(s,0):,}")

st.markdown("")

@st.fragment
def results_viewer():
    tab_data, tab_export = st.tabs(["📋 Detailed Data Viewer", "💾 CSV Export"])
    with tab_data:
        fc1, fc2, fc3 = st.columns([2, 2, 1])
        status_filter = fc1.multiselect("Filter by Status", STATUS_ORDER,
                                         default=[], placeholder="All statuses")
        search  = fc2.text_input("🔍 Search in Key", placeholder="Type to filter...")
        kd      = " › ".join(key_cols)
        sort_by = fc3.selectbox("Sort By",
                                 ["Δ Change", f"{val_col} (File A)",
                                  f"{val_col} (File B)", "Status", kd])

        view = results
        if status_filter: view = view[view["Status"].isin(status_filter)]
        if search:        view = view[view[kd].astype(str).str.contains(search, case=False, na=False)]
        if sort_by in view.columns:
            view = view.sort_values(sort_by, ascending=(sort_by == "Status"), na_position="last")

        st.caption(f"Showing **{len(view):,}** of **{len(results):,}** rows · "
                   f"**{view['Group'].nunique()}** group(s)")

        for grp_name, grp_df in view.groupby("Group", sort=True, dropna=False):
            gc_c   = grp_df["Status"].value_counts()
            badges = "  ".join(f"{STATUS_META[s]['icon']} {s}: {gc_c.get(s,0)}"
                                for s in STATUS_ORDER if gc_c.get(s, 0) > 0)
            label  = "Overall Dataset" if grp_name == "All" else f"{grp_col}: {grp_name}"
            with st.expander(f"{label} ({len(grp_df):,} rows)  |  {badges}",
                             expanded=(grp_name == "All" or view["Group"].nunique() <= 2)):
                show = grp_df.drop(columns=["Group"], errors="ignore")
                st.dataframe(style_table(show.head(1500)), use_container_width=True,
                             height=min(500, 45 + len(show) * 36))
                if len(show) > 1500:
                    st.warning(f"⚠️ First 1,500 rows shown. Export CSV for all {len(show):,}.")

    with tab_export:
        st.markdown("#### Download CSV Report")
        csv_bytes = view.drop(columns=["Group"], errors="ignore").to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_bytes,
                           file_name=f"SLA_Report_{val_col}.csv",
                           mime="text/csv", type="primary")

results_viewer()
