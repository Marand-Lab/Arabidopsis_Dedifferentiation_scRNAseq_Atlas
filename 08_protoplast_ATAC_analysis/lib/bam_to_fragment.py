#!/usr/bin/env python3
import sys
import gzip
import argparse
from pathlib import Path
import pysam
from collections import Counter


def main():
    ap = argparse.ArgumentParser(
        description=(
            "NAME-SORTED BAM -> fragments.tsv.gz.\n"
            "sc mode: chr start end barcode count\n"
            "bulk mode: chr start end group sample count"
        )
    )
    ap.add_argument("--bam", required=True, help="NAME-SORTED BAM (samtools sort -n)")
    ap.add_argument("--out", required=True, help="Output fragments.tsv.gz")
    ap.add_argument("--min-mapq", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--allow-crosschrom",
        action="store_true",
        help="Allow mates on different chromosomes (default: off).",
    )

    # --- mode control ---
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--sc",
        action="store_true",
        help="Single-cell mode (default): write chr start end barcode count using BC tag.",
    )
    mode.add_argument(
        "--bulk",
        action="store_true",
        help="Bulk mode: write chr start end group sample count (no BC tag required).",
    )

    # --- SC-specific options ---
    ap.add_argument(
        "--keep-unknown",
        action="store_true",
        help="(sc) Keep pairs with no BC tag as barcode=UNKNOWN (default: drop).",
    )

    # --- Bulk-specific options ---
    ap.add_argument("--group", help="(bulk) Group label to write (e.g., rep1).")
    ap.add_argument("--sample", help="(bulk) Sample label to write (e.g., Athaliana_leaf).")

    # optional: allow choosing a different tag than BC if needed later
    ap.add_argument(
        "--barcode-tag",
        default="BC",
        help="(sc) BAM tag to use as barcode (default: BC).",
    )

    args = ap.parse_args()

    is_bulk = bool(args.bulk)
    is_sc = not is_bulk  # default to sc unless --bulk provided
    if args.sc:
        is_sc = True
        is_bulk = False

    if is_bulk:
        if args.group is None or args.sample is None:
            ap.error("--bulk requires --group and --sample.")
    else:
        # sc mode: group/sample must not be required; ignore if provided
        pass

    bam = pysam.AlignmentFile(args.bam, "rb", check_sq=False, threads=args.threads)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    drops = Counter()
    n_pairs = n_kept = 0
    cur_name = None
    first = None

    def ok(r):
        return (
            (not r.is_unmapped)
            and (not r.is_secondary)
            and (not r.is_supplementary)
            and (r.mapping_quality >= args.min_mapq)
        )

    with gzip.open(outp, "wt") as out:
        for rec in bam:
            if rec.query_name != cur_name:
                cur_name = rec.query_name
                first = rec
                continue

            second = rec
            n_pairs += 1

            if not ok(first) or not ok(second):
                drops["mapq_or_unmapped_or_secondary"] += 1
                cur_name = None
                first = None
                continue

            if not (first.is_paired and second.is_paired):
                drops["not_paired"] += 1
                cur_name = None
                first = None
                continue

            if (first.reference_id != second.reference_id) and (not args.allow_crosschrom):
                drops["different_chrom"] += 1
                cur_name = None
                first = None
                continue

            # Coordinates (reference_start/reference_end exclude soft-clips)
            rname = bam.get_reference_name(first.reference_id)
            s1, e1 = first.reference_start, first.reference_end
            s2, e2 = second.reference_start, second.reference_end
            if None in (s1, e1, s2, e2):
                drops["no_coords"] += 1
                cur_name = None
                first = None
                continue

            start = min(s1, s2)
            end = max(e1, e2)
            if end <= start:
                drops["invalid_len"] += 1
                cur_name = None
                first = None
                continue

            if is_bulk:
                # bulk output: chr start end group sample count
                out.write(f"{rname}\t{start}\t{end}\t{args.group}\t{args.sample}\t1\n")
                n_kept += 1
            else:
                # sc output: chr start end barcode count
                tag = args.barcode_tag
                try:
                    bc = first.get_tag(tag)
                except KeyError:
                    try:
                        bc = second.get_tag(tag)
                    except KeyError:
                        if args.keep_unknown:
                            bc = "UNKNOWN"
                        else:
                            drops[f"no_{tag}_tag"] += 1
                            cur_name = None
                            first = None
                            continue
                out.write(f"{rname}\t{start}\t{end}\t{bc}\t1\n")
                n_kept += 1

            cur_name = None
            first = None

    bam.close()

    sys.stderr.write(
        f"[STATS] mode={'bulk' if is_bulk else 'sc'} pairs={n_pairs} kept={n_kept} dropped={sum(drops.values())}\n"
    )
    for k, v in drops.most_common():
        sys.stderr.write(f"[DROP] {k}: {v}\n")
    sys.stderr.write(f"[OUT] {outp}\n")


if __name__ == "__main__":
    main()
