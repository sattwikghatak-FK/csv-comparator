import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import gc

st.set_page_config(page_title="Air SLA Comparator", page_icon="✈️", layout="wide")

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

# ── File bytes cache (no re-hash on reruns) ───────────────────────────────────
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

# ── KEY FIX: normalise values to UPPERCASE + strip before joining ─────────────
# Root cause of 0 matches: June had "MUMBAI", May had "Mumbai".
# Raw string concat → "MUMBAI-110001" ≠ "Mumbai-110001" → zero overlap.
def make_key(df, cols):
    res = df[cols[0]].astype(str).str.strip().str.upper()
    for c in cols[1:]:
        res = res + "-" + df[c].astype(str).str.strip().str.upper()
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

        # Keep ALL columns from both files side-by-side for reference.
        # Common cols → "{col} (File A)" and "{col} (File B)" (both retained).
        # Only-A cols → "{col} (File A)".
        # Only-B cols → "{col} (File B)".
        rename_map = {}
        for c in [c for c in common_cols if c not in ["__key__", val_col]]:
            ca, cb = f"{c}_A", f"{c}_B"
            if ca in m.columns: rename_map[ca] = f"{c} (File A)"
            if cb in m.columns: rename_map[cb] = f"{c} (File B)"
        for c in only_a:
            if c in m.columns: rename_map[c] = f"{c} (File A)"
        for c in only_b:
            if c in m.columns: rename_map[c] = f"{c} (File B)"
        m.rename(columns=rename_map, inplace=True)

        if not m.empty: chunks.append(m)
        del sa, sb, m; gc.collect()

    ui("Assembling final dataset...", 95)
    final = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    # Normalise string columns to title-case so BANGALORE/Bangalore unify.
    skip_norm = {"__key__", "_val_A", "_val_B", "Status"}
    for c in final.select_dtypes(include=["object", "string"]).columns:
        if c not in skip_norm:
            final[c] = final[c].astype(str).str.strip().str.title()
            final[c] = final[c].replace("Nan", np.nan)

    # Resolve the group column — common columns get renamed to "(File A)"/"(File B)"
    # by the rename block above, so the plain name no longer exists.
    # Coalesce A then B so New rows (only in B) and Removed rows (only in A) are covered.
    if grp_col:
        grp_a = f"{grp_col} (File A)"
        grp_b = f"{grp_col} (File B)"
        if grp_col in final.columns:
            grp_series = final[grp_col]
        elif grp_a in final.columns:
            grp_series = final[grp_a].combine_first(
                final[grp_b] if grp_b in final.columns else pd.Series(dtype=str))
        elif grp_b in final.columns:
            grp_series = final[grp_b]
        else:
            grp_series = None

        final["Group"] = (grp_series.astype(str).str.strip().str.title().fillna("Unknown")
                          if grp_series is not None else "All")
    else:
        final["Group"] = "All"

    kd = "-".join(key_cols)
    final.rename(columns={"__key__": kd,
                           "_val_A": f"{val_col} (File A)",
                           "_val_B": f"{val_col} (File B)"}, inplace=True)

    # Build ordered column list:
    # [key | val A | val B | Δ | Status | Group]
    # then common cols interleaved as (File A)/(File B) pairs
    # then only-A cols, then only-B cols
    base = [kd, f"{val_col} (File A)", f"{val_col} (File B)", "Δ Change", "Status", "Group"]
    paired = []
    for c in common_cols:
        if c in ["__key__", val_col]: continue
        fa, fb = f"{c} (File A)", f"{c} (File B)"
        if fa in final.columns: paired.append(fa)
        if fb in final.columns: paired.append(fb)
    solo_a = [f"{c} (File A)" for c in only_a if f"{c} (File A)" in final.columns]
    solo_b = [f"{c} (File B)" for c in only_b if f"{c} (File B)" in final.columns]
    ctx = list(dict.fromkeys(paired + solo_a + solo_b))

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

# ── Output column selector ────────────────────────────────────────────────────
def slim_output(df: pd.DataFrame, val_col: str, key_cols: list) -> pd.DataFrame:
    """
    Keep only the reference columns the user cares about.
    All pattern matching is case-insensitive so Pincode/pincode/PINCODE all resolve.

    Columns retained (in order):
        Source City | Pincode | SLA A | SLA B | Δ | Status
        | Total SLA Hrs A | Total SLA Hrs B
        | MH Name A | MH Name B
        | S2H (coalesced) | PH Name (coalesced) | DH Name (coalesced)
        | Dest City (best available)
        | Group
    """
    kd   = "-".join(key_cols)
    SKIP = {kd, "Status", "Δ Change", "Group",
            f"{val_col} (File A)", f"{val_col} (File B)"}

    # ── helpers — all comparisons are .lower() so column case never matters ──
    def _find_a(patterns):
        for pat in patterns:
            c = next((c for c in df.columns
                      if pat.lower() in c.lower() and "(file a)" in c.lower()), None)
            if c: return c
        return None

    def _find_b(patterns):
        for pat in patterns:
            c = next((c for c in df.columns
                      if pat.lower() in c.lower() and "(file b)" in c.lower()), None)
            if c: return c
        return None

    def _find_any(patterns):
        return _find_a(patterns) or _find_b(patterns) or next(
            (c for c in df.columns
             if any(p.lower() in c.lower() for p in patterns)
             and c not in SKIP), None)

    def _coalesce(patterns):
        """A where available, B as fallback — handles New/Removed rows gracefully."""
        ca, cb = _find_a(patterns), _find_b(patterns)
        if ca and cb:  return df[ca].combine_first(df[cb])
        if ca:         return df[ca]
        if cb:         return df[cb]
        plain = _find_any(patterns)
        return df[plain] if plain else None

    # ── assemble ──────────────────────────────────────────────────────────────
    out = {}

    # ── Composite key (kept for reference) ────────────────────────────────────
    out[kd] = df[kd]

    # ── Source City — extracted as a standalone column for grouping & filtering
    # Matches "source city", "Source City", "SOURCE CITY" etc.
    s = _coalesce(["source city", "source_city"])
    if s is not None:
        out["Source City"] = s

    # ── Pincode — standalone
    # Exact match on the part before "(File A/B)" to avoid
    # "Pincode_Formatted", "pincode check", etc.
    def _exact_a(name):
        n = name.lower()
        return next((c for c in df.columns
                     if c.lower().split("(file")[0].strip().rstrip("_") == n
                     and "(file a)" in c.lower()), None)

    def _exact_b(name):
        n = name.lower()
        return next((c for c in df.columns
                     if c.lower().split("(file")[0].strip().rstrip("_") == n
                     and "(file b)" in c.lower()), None)

    pc_a, pc_b = _exact_a("pincode"), _exact_b("pincode")
    pincode_series = (df[pc_a].combine_first(df[pc_b]) if pc_a and pc_b
                      else df[pc_a] if pc_a
                      else df[pc_b] if pc_b else None)
    if pincode_series is not None:
        out["Pincode"] = pincode_series

    # ── SLA days — both files (primary comparison metric) ─────────────────────
    for lbl in [f"{val_col} (File A)", f"{val_col} (File B)", "Δ Change", "Status"]:
        if lbl in df.columns: out[lbl] = df[lbl]

    # ── Total SLA Hours — both files ──────────────────────────────────────────
    ca = _find_a(["total_sla_hrs", "total_sla"])
    cb = _find_b(["total_sla_hrs", "total_sla"])
    if ca: out["Total SLA Hrs (File A)"] = df[ca]
    if cb: out["Total SLA Hrs (File B)"] = df[cb]

    # ── MH Name — both files ──────────────────────────────────────────────────
    ca = _find_a(["ekart_mh_name", "mh_name"])
    cb = _find_b(["ekart_mh_name", "mh_name"])
    if ca: out["MH Name (File A)"] = df[ca]
    if cb: out["MH Name (File B)"] = df[cb]

    # ── Single-value reference cols (coalesced A → B) ─────────────────────────
    s = _coalesce(["s2h_in_hr", "s2h"])
    if s is not None: out["S2H (hrs)"] = s

    s = _coalesce(["ph_name"])
    if s is not None: out["PH Name"] = s

    s = _coalesce(["dh_name"])
    if s is not None: out["DH Name"] = s

    # ── Dest City — best available ─────────────────────────────────────────────
    dest = _find_any(["mapped_dest_city", "city_from_mapped_dmh",
                      "city_from_dmh", "dest_city", "dest city"])
    if dest: out["Dest City"] = df[dest]

    if "Group" in df.columns: out["Group"] = df["Group"]

    return pd.DataFrame(out)


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.title("✈️ Air SLA Comparator")
st.caption("Compare massive CSV/Excel datasets seamlessly · Optimized Chunk Processing Engine")

with st.container(border=True):
    st.markdown("#### 1. Upload Datasets")
    st.warning("""
    **🚨 Mandatory Columns:** `Source City`, `pincode` (keys) · `total_sla_hrs` + `f2f_buffer_sla` (formula)

    **💡 Optional/Contextual:** `ph_name`, `dh_name`, `f2p_sla`, `s2h_in_hr`, `lpht`
    """, icon="⚠️")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📁 File A — Baseline / Previous (e.g. May SLA)**")
        up_a = st.file_uploader("File A", type=["csv","xlsx"], key="fa",
                                 label_visibility="collapsed", on_change=reset_computation)
        sheet_a = (st.selectbox("📝 Sheet (File A)", get_sheet_names(up_a), on_change=reset_computation)
                   if up_a and up_a.name.lower().endswith(".xlsx") else None)
    with c2:
        st.markdown("**📁 File B — Current / New (e.g. June SLA)**")
        up_b = st.file_uploader("File B", type=["csv","xlsx"], key="fb",
                                 label_visibility="collapsed", on_change=reset_computation)
        sheet_b = (st.selectbox("📝 Sheet (File B)", get_sheet_names(up_b), on_change=reset_computation)
                   if up_b and up_b.name.lower().endswith(".xlsx") else None)

if not (up_a and up_b):
    st.info("⬆ Upload both files to configure your comparison.", icon="ℹ️")
    st.stop()

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

    default_strat = 1 if (any("total_sla" in c.lower() for c in common) and
                          any("f2f" in c.lower() for c in common)) else 0
    metric_strat = st.radio(
        "⚙️ Evaluation Strategy",
        ["Compare an existing column",
         "Compute Derived SLA (Days): CEIL((Total SLA Hrs − F2F Buffer Hrs) / 24)"],
        index=default_strat, on_change=reset_computation)

    if "Compute" in metric_strat:
        mc1, mc2, mc3 = st.columns(3)
        cols_for_formula = numeric_cols or common

        # ── FIX: default to total_sla_hrs for SLA col ─────────────────────
        def_sla_idx = next(
            (i for i, c in enumerate(cols_for_formula) if "total_sla" in c.lower()), 0)

        # ── FIX: default to f2f_BUFFER_sla, NOT f2f_del_sla ──────────────
        # Formula: ceil((total_sla_hrs - f2f_buffer_sla) / 24)
        # f2f_buffer_sla is the correct deduction column (f2f_del_sla is ~50 hrs
        # and would wrongly collapse all SLA values to ~0).
        def_f2f_idx = next(
            (i for i, c in enumerate(cols_for_formula) if "f2f_buffer" in c.lower()),
            next(
                (i for i, c in enumerate(cols_for_formula) if "f2f" in c.lower()),
                min(1, len(cols_for_formula) - 1)))

        sla_hrs_col = mc1.selectbox("⏱️ Total SLA Hours Col",
                                     options=cols_for_formula, index=def_sla_idx,
                                     on_change=reset_computation)
        f2f_hrs_col = mc2.selectbox("🛑 F2F Buffer Col",
                                     options=cols_for_formula, index=def_f2f_idx,
                                     on_change=reset_computation,
                                     help="Use f2f_buffer_sla (not f2f_del_sla). "
                                          "Formula: CEIL((total_sla_hrs − f2f_buffer_sla) / 24)")
        val_col = mc3.text_input("✏️ Computed Column Name",
                                  value="Computed_SLA_Days", on_change=reset_computation)

        # Live formula preview using File A sample
        try:
            sla_sample = pd.to_numeric(df_a_prev[sla_hrs_col].head(3), errors='coerce')
            f2f_sample = pd.to_numeric(df_a_prev[f2f_hrs_col].head(3), errors='coerce')
            computed   = np.ceil((sla_sample - f2f_sample) / 24)
            st.info(f"📐 **Formula preview (File A sample):** "
                    f"`CEIL(({sla_sample.iloc[0]:.1f} − {f2f_sample.iloc[0]:.1f}) / 24)` "
                    f"= **{computed.iloc[0]:.0f} days** "
                    f"| next rows → {', '.join(f'{v:.0f}' for v in computed.iloc[1:])} days",
                    icon="🔢")
        except Exception:
            pass
    else:
        # key_cols may not be defined yet at this point — use empty list as fallback
        _existing_key_cols = st.session_state.get("key_cols", [])
        val_opts = ([c for c in numeric_cols if c not in _existing_key_cols]
                    or [c for c in common if c not in _existing_key_cols])
        val_col  = st.selectbox("📐 Metric to Compare", options=val_opts,
                                 on_change=reset_computation)

    st.markdown("---")
    mode_c1, mode_c2 = st.columns(2)
    comp_mode = mode_c1.radio("⚙️ Match Architecture",
                               ["Strict 1-to-1 (Deduplicate Both)",
                                "1-to-Many (Broadcast granular rows)"],
                               index=1, on_change=reset_computation)
    granular_file = None
    if "1-to-Many" in comp_mode:
        granular_file = mode_c2.selectbox(
            "📌 Which file is granular? (keep its duplicates)",
            ["File B (Current/New)", "File A (Baseline/Previous)"],
            on_change=reset_computation)
    st.markdown("---")

    cfg1, cfg2 = st.columns(2)
    with cfg1:
        suggested_keys = [c for c in common
                          if c.lower().replace(" ","") in
                          ["sourcecity","pincode","source_city"]]
        default_keys = suggested_keys if suggested_keys else ([common[0]] if common else [])
        key_cols = st.multiselect("🔑 Unique Identifier(s)", options=common,
                                   default=default_keys, on_change=reset_computation)
    with cfg2:
        grp_sel = st.selectbox("🗂 Group By (Optional)",
                                ["(none)"] + [c for c in common if c != val_col],
                                on_change=reset_computation)
        grp_col = None if grp_sel == "(none)" else grp_sel

    higher_is = st.radio(
        "📈 Value direction meaning",
        ["higher_is_worse", "higher_is_better"],
        index=0 if infer_direction(val_col) == "higher_is_worse" else 1,
        format_func=lambda x: "⬆ Higher = Worse (e.g., Days)"
                               if x == "higher_is_worse" else "⬆ Higher = Better (e.g., Score)",
        horizontal=True, on_change=reset_computation)

    run_disabled = not key_cols or not val_col
    if "Compute" in metric_strat:
        run_disabled = run_disabled or not sla_hrs_col or not f2f_hrs_col
    run = st.button("🚀 Run Full Analysis", type="primary", disabled=run_disabled)

if not run and "results" not in st.session_state:
    st.stop()

# ── Compute ───────────────────────────────────────────────────────────────────
if run:
    st.markdown("---")
    status_text  = st.empty()
    progress_bar = st.progress(0)

    essential_cols_a = set(key_cols)
    if grp_col: essential_cols_a.add(grp_col)
    if "Compute" in metric_strat:
        essential_cols_a.update([sla_hrs_col, f2f_hrs_col])
    else:
        essential_cols_a.add(val_col)
    essential_cols_b = {col_map.get(c, c) for c in essential_cols_a}

    status_text.markdown(f"**⏳ Reading {up_a.name}...**"); progress_bar.progress(10)
    df_a_full = load_full_data(up_a, sheet_a)
    contextual_a = [c for c in df_a_full.columns if c in common]
    df_a_full = df_a_full[list(set(list(essential_cols_a) + contextual_a)
                               .intersection(df_a_full.columns))]

    status_text.markdown(f"**⏳ Reading {up_b.name}...**"); progress_bar.progress(35)
    df_b_full = load_full_data(up_b, sheet_b)
    mapped_b_keep = [col_map.get(c, c) for c in common]
    df_b_full = df_b_full[list(set(list(essential_cols_b) + mapped_b_keep)
                               .intersection(df_b_full.columns))]

    if "Compute" in metric_strat:
        status_text.markdown(f"**⏳ Computing {val_col}...**"); progress_bar.progress(45)
        for df, is_a in [(df_a_full, True), (df_b_full, False)]:
            mapped_sla = sla_hrs_col if is_a else col_map.get(sla_hrs_col, sla_hrs_col)
            mapped_f2f = f2f_hrs_col if is_a else col_map.get(f2f_hrs_col, f2f_hrs_col)
            sla = pd.to_numeric(df.get(mapped_sla, pd.Series(0, index=df.index)),
                                errors='coerce').fillna(0)
            f2f = pd.to_numeric(df.get(mapped_f2f, pd.Series(0, index=df.index)),
                                errors='coerce').fillna(0)
            df[val_col] = np.ceil((sla - f2f) / 24)

    results_full = process_comparison_chunked(
        df_a_full, df_b_full, comp_mode, granular_file, col_map,
        key_cols, val_col, grp_col, higher_is, status_text, progress_bar)

    # Slim to only the reference columns the user needs
    results = slim_output(results_full, val_col, key_cols)
    del results_full; gc.collect()

    st.session_state.update({"results": results, "key_cols": key_cols,
                              "val_col": val_col, "grp_col": grp_col, "higher_is": higher_is})
    del df_a_full, df_b_full; gc.collect()
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
        kd      = "-".join(key_cols)
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

        # Group by Source City column if it exists, otherwise fall back to Group
        grp_field = "Source City" if "Source City" in view.columns else "Group"
        for grp_name, grp_df in view.groupby(grp_field, sort=True, dropna=False):
            gc_c   = grp_df["Status"].value_counts()
            badges = "  ".join(f"{STATUS_META[s]['icon']} {s}: {gc_c.get(s,0)}"
                                for s in STATUS_ORDER if gc_c.get(s, 0) > 0)
            label  = ("Overall Dataset" if grp_name in ("All", "Unknown")
                      else f"Source City: {grp_name}")
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
                           file_name=f"Air_SLA_Report_{val_col}.csv",
                           mime="text/csv", type="primary")

results_viewer()
