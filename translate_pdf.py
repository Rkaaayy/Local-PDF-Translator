#!/usr/bin/env python3
"""Translate PDFs into English on your own PC.

Uses an NVIDIA GPU automatically when one is available, and falls back to the CPU
otherwise.

The original layout is kept: the English text is written into the same places
as the original text. Scanned pages are read with OCR; text that is not clearly
legible is left in its original language.

Examples:
  python translate_pdf.py report.pdf                      # -> report_en.pdf
  python translate_pdf.py report.pdf -o out/english.pdf
  python translate_pdf.py ./pdfs -o ./english              # every PDF in a folder
  python translate_pdf.py scan.pdf --ocr-langs de,en       # scanned German pages
  python translate_pdf.py --check-hardware
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from hardware import (default_batch_size, format_report, inspect_hardware,
                      resolve_device)


def collect_inputs(items: list[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for item in items:
        if item.is_dir():
            pdfs.extend(sorted(p for p in item.glob("*.pdf") if p.is_file()))
        elif item.is_file() and item.suffix.lower() == ".pdf":
            pdfs.append(item)
        else:
            print(f"Skipping '{item}': not a PDF file or folder.", file=sys.stderr)
    return pdfs


def output_path_for(src: Path, output: Path | None, n_inputs: int) -> Path:
    if output is None:
        dst = src.with_name(f"{src.stem}_en.pdf")
    elif output.suffix.lower() == ".pdf" and n_inputs == 1:
        dst = output
    else:
        dst = output / f"{src.stem}_en.pdf"
    dst.parent.mkdir(parents=True, exist_ok=True)
    return dst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate PDF documents into English locally.")
    parser.add_argument("inputs", nargs="*", type=Path,
                        help="PDF file(s) or folder(s) containing PDFs")
    parser.add_argument("-o", "--output", type=Path,
                        help="output PDF path (single file) or output folder")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="auto = GPU if available, else CPU (default)")
    parser.add_argument("--source-lang", metavar="CODE",
                        help="source language code such as de, fr, es (skips auto-detection)")
    parser.add_argument("--ocr-langs", default="en",
                        help="comma-separated OCR languages for scanned pages, e.g. en,de (default: en)")
    parser.add_argument("--min-ocr-confidence", type=float, default=0.2,
                        help="OCR lines below this confidence (0-1) are kept in their original "
                             "language (default: 0.2)")
    parser.add_argument("--batch-size", type=int, help="translation batch size (auto if omitted)")
    parser.add_argument("--beams", type=int, default=4,
                        help="beam search width: higher is slightly better and slower (default: 4)")
    parser.add_argument("--fp32", action="store_true",
                        help="use full precision on the GPU (default: fp16, which is faster)")
    parser.add_argument("--check-hardware", action="store_true",
                        help="only print the hardware check and exit")
    args = parser.parse_args(argv)

    report = inspect_hardware()

    if args.check_hardware:
        print(format_report(report, resolve_device("auto", report)))
        return 0

    try:
        device = resolve_device(args.device, report)
    except RuntimeError as err:
        print(format_report(report))
        print(f"\nError: {err}", file=sys.stderr)
        return 2

    gpu_vram = report.gpus[0].vram_mb if device == "cuda" and report.gpus else 0
    batch_size = args.batch_size or default_batch_size(device, gpu_vram)
    print(format_report(report, device))
    print(f"  Batch size         : {batch_size}"
          + (" (fp16)" if device == "cuda" and not args.fp32 else ""))

    pdfs = collect_inputs(args.inputs)
    if not pdfs:
        parser.print_usage(sys.stderr)
        print("Error: no PDF files to process. Pass a PDF or a folder.", file=sys.stderr)
        return 2

    # Import heavy libraries only once the device is known.
    from pipeline import PDFTranslator

    ocr_langs = [code.strip() for code in args.ocr_langs.split(",") if code.strip()]
    translator = PDFTranslator(
        device=device,
        batch_size=batch_size,
        beams=args.beams,
        use_fp16=not args.fp32,
        source_lang=args.source_lang,
        ocr_langs=ocr_langs,
        min_ocr_conf=args.min_ocr_confidence,
    )

    failures = 0
    total_start = time.perf_counter()
    for src in pdfs:
        dst = output_path_for(src, args.output, len(pdfs))
        print(f"\n==> {src.name}  ->  {dst}")
        try:
            stats = translator.translate_file(src, dst)
        except Exception as err:  # keep going with the next file
            failures += 1
            print(f"  FAILED: {err}", file=sys.stderr)
            continue
        print(f"  done in {stats['seconds']} s: {stats['translated']} text areas translated, "
              f"{stats['scanned_pages']} scanned page(s) OCR'd, {stats['kept']} kept original")
        print(f"  details: {stats['report_path']}")

    print(f"\nFinished {len(pdfs) - failures}/{len(pdfs)} file(s) in "
          f"{time.perf_counter() - total_start:.1f} s.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
