# pathway_enrichment.py
# Pathway enrichment analysis for PSME1 KD DEPs
# Run from project root: python pathway_enrichment.py
#
# Requirements: pip install gseapy matplotlib pandas numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gseapy as gp
import re
import textwrap
import warnings
warnings.filterwarnings("ignore")

# Config
INPUT_CSV      = "data/raw/DEA_results_annotated.csv"
OUT_PDF        = "results/psme1_kd_pathway_enrichment.pdf"
OUT_PNG        = "results/psme1_kd_pathway_enrichment.png"
TOP_N_TERMS    = 12     # max terms per panel

LIBRARIES = {
    "GO Biological\nProcess": "GO_Biological_Process_2025",
    "Reactome":                "Reactome_2022",
    "Hallmark":                "MSigDB_Hallmark_2020",
}

# Step 1: Load DEPs
print("Loading DEA results...")
df = pd.read_csv(INPUT_CSV)
print(f"  Read {len(df)} rows from {INPUT_CSV}")

sig = df[df["significant"] == True].copy()
up_genes   = sig[sig["direction"] == "Up in KO"]["ProteinID"].tolist()
down_genes = sig[sig["direction"] == "Down in KO"]["ProteinID"].tolist()
all_genes  = sig["ProteinID"].tolist()

print(f"  {len(all_genes)} significant DEPs — {len(up_genes)} upregulated, {len(down_genes)} downregulated")
print(f"  Gene list: {all_genes}\n")

# Step 2: Run Enrichr
def run_enrichr(gene_list, db_label, db_lib):
    try:
        res = gp.enrichr(
            gene_list=gene_list,
            gene_sets=db_lib,
            organism="human",
            outdir=None,
            verbose=False,
        )
        r = res.results.copy()
        r["Database"] = db_label
        n_sig = (r["Adjusted P-value"] < 0.05).sum()
        print(f"  {db_label.replace(chr(10), ' ')}: {len(r)} terms returned, {n_sig} pass FDR < 0.05")
        return r
    except Exception as e:
        print(f"  Query failed for {db_label}: {e}")
        return None

print("Querying Enrichr databases...")
frames = []
for label, lib in LIBRARIES.items():
    print(f"  Sending {len(all_genes)} genes to {lib}...")
    r = run_enrichr(all_genes, label, lib)
    if r is not None:
        frames.append(r)

if not frames:
    raise RuntimeError("All queries failed — check internet connection.")

print(f"  Done. {len(frames)} databases returned results.\n")

# Step 3: Tidy results
print("Tidying combined results...")
combined = pd.concat(frames, ignore_index=True)
print(f"  {len(combined)} total terms across all databases")

combined[["hits", "bg"]] = (
    combined["Overlap"].str.split("/", expand=True).astype(int)
)
combined["gene_ratio"]    = combined["hits"] / combined["bg"]
combined["neg_log10_fdr"] = -np.log10(combined["Adjusted P-value"].clip(lower=1e-10))
print("  Calculated gene ratio and -log10(FDR) per term")

def clean_term(raw, wrap_at=32):
    s = re.sub(r"\s*\(GO:\d+\)\s*$", "", str(raw)).strip()
    s = s[0].upper() + s[1:] if s else s
    return "\n".join(textwrap.wrap(s, width=wrap_at))

combined["term_label"] = combined["Term"].apply(clean_term)
print("  Cleaned term labels")

sig_terms = combined[combined["Adjusted P-value"] < 0.05].copy()
print(f"  {len(sig_terms)} terms pass FDR < 0.05")

# Per-database fallback: if a database has no FDR-significant terms,
# show its top N by raw p-value so it always gets a panel
top_frames = []
for db_label in LIBRARIES.keys():
    db_data = combined[combined["Database"] == db_label].copy()
    if db_data.empty:
        print(f"  {db_label.replace(chr(10), ' ')}: no results — skipping panel")
        continue
    db_sig = db_data[db_data["Adjusted P-value"] < 0.05]
    if db_sig.empty:
        print(f"  {db_label.replace(chr(10), ' ')}: no FDR-significant terms — showing top {TOP_N_TERMS} by p-value")
        db_top = db_data.sort_values("P-value").head(TOP_N_TERMS)
    else:
        db_top = db_sig.sort_values("Adjusted P-value").head(TOP_N_TERMS)
    top_frames.append(db_top)

top_terms = pd.concat(top_frames, ignore_index=True)
print(f"  Keeping top {TOP_N_TERMS} terms per database for plotting\n")

# Step 4: Build figure
DB_ORDER  = list(LIBRARIES.keys())
DB_COLORS = {
    "GO Biological\nProcess": "#3A7EBF",
    "Reactome":                "#C1552A",
    "Hallmark":                "#4DAF6F",
}
CMAP      = plt.get_cmap("YlOrRd")
DOT_SCALE = 38
FONT      = "DejaVu Sans"

db_groups = {db: grp for db, grp in top_terms.groupby("Database", sort=False)}
print(f"  Panels to render: {list(db_groups.keys())}")

# Always create one panel per library in DB_ORDER, skip missing ones
active_dbs = [db for db in DB_ORDER if db in db_groups]
n_panels   = len(active_dbs)

if n_panels == 0:
    raise RuntimeError("No databases returned any results.")

# Dynamic figure height based on max terms across panels
max_rows   = max(len(db_groups[db]) for db in active_dbs)
fig_height = max(6, max_rows * 0.75 + 2.5)
fig_width  = 8.5 * n_panels

fig, axes = plt.subplots(
    1, n_panels,
    figsize=(fig_width, fig_height),
    facecolor="white",
)
if n_panels == 1:
    axes = [axes]

vmin = 1.0
vmax = max(top_terms["neg_log10_fdr"].max(), 2.0)

for ax, db in zip(axes, active_dbs):

    grp   = db_groups[db].sort_values("Adjusted P-value", ascending=True).reset_index(drop=True)
    y_pos = list(range(len(grp)))

    sc = ax.scatter(
        grp["gene_ratio"],
        y_pos,
        s=grp["hits"] * DOT_SCALE,
        c=grp["neg_log10_fdr"],
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        edgecolors="#444444",
        linewidths=0.45,
        zorder=3,
    )

    # Subtle row banding
    for i in y_pos:
        ax.axhspan(i - 0.5, i + 0.5, color="#f7f7f7" if i % 2 == 0 else "white",
                   zorder=0, linewidth=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(grp["term_label"], fontsize=9, fontfamily=FONT)
    ax.set_xlabel("Gene Ratio", fontsize=10, fontfamily=FONT, labelpad=7)
    ax.set_ylim(-0.7, len(grp) - 0.3)
    ax.invert_yaxis()

    ax.set_title(
        db.replace("\n", " "),
        fontsize=15, fontweight="bold",
        color=DB_COLORS[db], pad=11,
        fontfamily=FONT,
    )

    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.4, zorder=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", labelsize=9)

    # Colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm   = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.30, pad=0.025, aspect=14)
    cbar.set_label("−log₁₀(FDR)", fontsize=8.5, fontfamily=FONT)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)

# Gene count size legend on first panel
legend_ax = axes[0]
for n_genes in [1, 3, 5]:
    legend_ax.scatter(
        [], [], s=n_genes * DOT_SCALE,
        c="#aaaaaa", edgecolors="#444444", linewidths=0.45,
        label=str(n_genes),
    )
legend_ax.legend(
    title="Proteins\nhit", title_fontsize=8,
    fontsize=8, loc="lower right",
    framealpha=0.9, edgecolor="#cccccc",
    labelspacing=0.6,
)

# Suptitle
up_n   = len(up_genes)
down_n = len(down_genes)
fig.suptitle(
    f"Pathway Enrichment  ·  PSME1 KD vs Scramble  ·  NCI-H441\n"
    f"{len(all_genes)} DEPs  ({up_n} ↑ up  ·  {down_n} ↓ down)  |  FDR < 0.05  |  |log₂FC| > 1",
    fontsize=16, fontweight="bold", y=1.02,
    fontfamily=FONT, color="#1a1a1a",
)

print("Rendering figure...")
plt.tight_layout(w_pad=3.5)
fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
print(f"  Saved {OUT_PDF}")
print(f"  Saved {OUT_PNG}")

# Step 5: Write genes per displayed term to markdown
OUT_MD = "results/psme1_kd_pathway_genes.md"
print(f"\nWriting gene associations to {OUT_MD}...")

with open(OUT_MD, "w") as f:
    f.write("# Pathway Enrichment — PSME1 KD vs Scramble — NCI-H441\n\n")
    f.write(f"**{len(all_genes)} DEPs** ({len(up_genes)} up, {len(down_genes)} down) &nbsp;|&nbsp; FDR < 0.05 &nbsp;|&nbsp; |log₂FC| > 1\n\n")

    for db_label in active_dbs:
        grp = db_groups[db_label].sort_values("Adjusted P-value").reset_index(drop=True)
        f.write(f"## {db_label.replace(chr(10), ' ')}\n\n")
        f.write("| Term | FDR | Genes |\n")
        f.write("|------|-----|-------|\n")
        for _, row in grp.iterrows():
            genes_hit = row.get("Genes", "")
            gene_list_str = " · ".join(sorted(str(genes_hit).split(";"))) if genes_hit else "—"
            fdr = row["Adjusted P-value"]
            term = row["Term"].strip()
            term = re.sub(r"\s*\(GO:\d+\)\s*$", "", term)
            f.write(f"| {term} | {fdr:.4f} | {gene_list_str} |\n")
        f.write("\n")

print(f"  Saved {OUT_MD}")

print("\nDone.")
plt.show()