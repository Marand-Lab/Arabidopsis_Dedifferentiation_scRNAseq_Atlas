#!/usr/bin/env python3
"""Build signature_metadata.tsv from At_MotifClusters.txt + at_motif_family_assignments.tsv.

Produces a metadata table with columns:
  signature_id, display_name, representative_name, primary_family, primary_class,
  n_members, member_ids, member_names, all_families, is_multi_family

Run after v3_motif_clustering.R completes.

Usage:
  python build_signature_metadata.py
"""

import re
import pandas as pd
from pathlib import Path


def main():
    # Data files live in data/motif_signatures/ (relative to project root)
    project_root = Path(__file__).parent.parent.parent
    base = project_root / "data" / "motif_signatures"

    # Load family assignments
    fam_df = pd.read_csv(base / "at_motif_family_assignments.tsv", sep="\t")
    # Build lookup: motif_key (dots→underscores of at_name) → row
    fam_df["motif_key"] = fam_df["at_name"].str.replace(".", "_", regex=False)
    fam_lookup = fam_df.set_index("motif_key")
    # Secondary lookup by motif_version key (e.g. MA1253.1 → MA1253_1)
    # Used when MEME file TF names differ from TSV at_name (e.g. ERF036 vs AT3G16280)
    fam_df["version_key"] = fam_df["motif_version"].str.replace(".", "_", regex=False)
    version_lookup = fam_df.set_index("version_key")

    # Load cluster table
    clusters = {}
    with open(base / "At_MotifClusters.txt") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sig_label = parts[0]  # MOTIFN
                members_str = parts[1]
                clusters[sig_label] = members_str

    rows = []
    sig_counter = 0
    for sig_label in sorted(clusters.keys(), key=lambda x: int(x.replace("MOTIF", ""))):
        sig_counter += 1
        sig_id = f"sig_{sig_counter:03d}"
        members_str = clusters[sig_label]

        # Parse members (semicolon-separated)
        members = [m.strip() for m in members_str.split(";") if m.strip()]

        # Look up families for each member
        member_families = []
        member_names = []
        member_ids = []
        for m in members:
            # m is like MA0587_1_TCP16 (underscores, from R gsub)
            if m in fam_lookup.index:
                row = fam_lookup.loc[m]
                member_families.append(row["tf_family"])
                member_names.append(row["tf_name"])
                member_ids.append(row["base_id"])
            else:
                # Primary lookup failed: TF name in MEME file differs from TSV.
                # Fall back to version-key lookup (MAxxxx_version prefix only).
                parts = m.split("_")
                version_key = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else m
                tf_name = "_".join(parts[2:]) if len(parts) >= 3 else m
                if version_key in version_lookup.index:
                    row = version_lookup.loc[version_key]
                    member_families.append(row["tf_family"])
                    member_names.append(row["tf_name"])
                    member_ids.append(row["base_id"])
                else:
                    member_names.append(tf_name or m)
                    member_ids.append(m)
                    member_families.append("Unknown")

        # Determine primary family (most common)
        from collections import Counter
        fam_counts = Counter(member_families)
        primary_family = fam_counts.most_common(1)[0][0]
        all_families_set = sorted(set(member_families))
        is_multi_family = len(all_families_set) > 1

        # Representative: first member's TF name
        representative = member_names[0] if member_names else "Unknown"

        # Primary class from first member
        first_key = members[0] if members[0] in fam_lookup.index else None
        primary_class = ""
        if first_key and first_key in fam_lookup.index:
            pc = fam_lookup.loc[first_key, "tf_class"]
            if pd.notna(pc):
                primary_class = pc

        # Display name: {family}_{representative}
        display_name = re.sub(r"[/\\: ]+", "_", f"{primary_family}_{representative}")

        rows.append({
            "signature_id": sig_id,
            "signature_label": sig_label,
            "display_name": display_name,
            "representative_name": representative,
            "primary_family": primary_family,
            "primary_class": primary_class,
            "n_members": len(members),
            "member_ids": ";".join(member_ids),
            "member_names": ";".join(member_names),
            "all_families": ";".join(all_families_set),
            "is_multi_family": is_multi_family,
        })

    df = pd.DataFrame(rows)
    out_path = base / "signature_metadata.tsv"
    df.to_csv(out_path, sep="\t", index=False)

    print(f"[INFO] Wrote {len(df)} signatures to {out_path}")
    print(f"[INFO] Families: {df['primary_family'].nunique()}")
    print(f"[INFO] Multi-family: {df['is_multi_family'].sum()}")
    print(f"\nFamily distribution:")
    print(df["primary_family"].value_counts().to_string())


if __name__ == "__main__":
    main()
