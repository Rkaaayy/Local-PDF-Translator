"""
Models: Helsinki-NLP Opus-MT (CC-BY-4.0) for translation, EasyOCR (Apache-2.0) for
OCR, and PyMuPDF (AGPL, fine for personal use) for PDF editing. Translation and OCR
run on the GPU when the device is 'cuda'.
"""

from __future__ import annotations

import gc
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf
from tqdm import tqdm

TEXT_SAMPLE_SCALE = 2.0       # render scale for sampling background colours (~144 dpi)
OCR_SCALE = 1.5               # render scale for OCR (~108 dpi); higher uses a lot of RAM
MIN_TEXT_CHARS = 25           # a page with less text than this is treated as scanned
BIG_IMAGE_FRACTION = 0.7      # a page covered by one image this much is treated as scanned
MIN_TEXT_CONTRAST = 1.5       # text-layer text below this contrast ratio is invisible
MIN_OCR_CONTRAST = 2.0        # OCR lines below this contrast are not clearly visible
MIN_FONT_SCALE = 0.6          # English text may shrink to 60% of the original size
LINE_SPACING = 1.2            # line height as a multiple of font size
MAX_SEGMENT_CHARS = 350       # translate in chunks of about this many characters

OPUS_MODEL = "Helsinki-NLP/opus-mt-{src}-en"
FALLBACK_MODEL = "Helsinki-NLP/opus-mt-mul-en"   # multilingual -> English

LANGUAGE_NAMES = {
    "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
    "nl": "Dutch", "ru": "Russian", "uk": "Ukrainian", "pl": "Polish", "cs": "Czech",
    "sv": "Swedish", "da": "Danish", "fi": "Finnish", "no": "Norwegian", "el": "Greek",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "he": "Hebrew",
    "tr": "Turkish", "hi": "Hindi", "vi": "Vietnamese", "id": "Indonesian", "en": "English",
    "mul": "mixed languages",
}

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",                                    # Windows
    "/System/Library/Fonts/Supplemental/Arial.ttf",                  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",               # Linux
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

@dataclass
class Unit:
    """One piece of text to translate, with the area it occupies on its page."""
    page: int                          # 0-based page index
    rect: pymupdf.Rect                 # area in PDF points
    text: str                          # source text
    size: float                        # original font size (points)
    bg: tuple[float, float, float]     # background colour, 0..1 per channel
    ink: tuple[float, float, float]    # colour for the English text, 0..1
    source: str                        # "text" (text layer) or "ocr"
    lang: str | None = None
    english: str | None = None
    skip_reason: str | None = None
    layout: tuple | None = None        # (lines, font size, line height, ascender)


@dataclass
class OcrLine:
    rect: pymupdf.Rect
    text: str
    conf: float
    bg: np.ndarray
    contrast: float


# --------------------------------------------------------------------------- #
# Colour helpers
# --------------------------------------------------------------------------- #

def _rel_lum(rgb: tuple[float, float, float]) -> float:
    """WCAG relative luminance of an RGB colour with channels in 0..1."""
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: float, b: float) -> float:
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _luminance_array(pixels: np.ndarray) -> np.ndarray:
    c = pixels.astype(np.float64) / 255.0
    lin = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722])


def _pick_ink(bg: tuple[float, float, float],
              preferred: tuple[float, float, float] | None = None) -> tuple[float, float, float]:
    """Use the original text colour if it stays readable, else black or white."""
    bg_lum = _rel_lum(bg)
    if preferred is not None and _contrast(_rel_lum(preferred), bg_lum) >= 3.0:
        return preferred
    return max([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
               key=lambda c: _contrast(_rel_lum(c), bg_lum))


def _int_to_rgb01(color: int) -> tuple[float, float, float]:
    return (((color >> 16) & 255) / 255, ((color >> 8) & 255) / 255, (color & 255) / 255)


# --------------------------------------------------------------------------- #
# Page rendering and sampling
# --------------------------------------------------------------------------- #

def render_page(page: pymupdf.Page, scale: float) -> np.ndarray:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]


def _region(arr: np.ndarray, rect: pymupdf.Rect, scale: float) -> np.ndarray | None:
    h, w = arr.shape[:2]
    x0, y0 = max(0, int(rect.x0 * scale)), max(0, int(rect.y0 * scale))
    x1, y1 = min(w, int(np.ceil(rect.x1 * scale))), min(h, int(np.ceil(rect.y1 * scale)))
    if x1 <= x0 or y1 <= y0:
        return None
    return arr[y0:y1, x0:x1].reshape(-1, 3)


def _median_bg(pixels: np.ndarray) -> tuple[float, float, float]:
    med = np.median(pixels, axis=0) / 255.0
    return (float(med[0]), float(med[1]), float(med[2]))


def _has_letters(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def _join_lines(texts: list[str]) -> str:
    """Join wrapped lines, re-joining words that were hyphenated at a line end."""
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if not out:
            out = t
        elif out.endswith("-") and t[:1].islower():
            out = out[:-1] + t
        else:
            out = f"{out} {t}"
    return " ".join(out.split())


# --------------------------------------------------------------------------- #
# Extraction: text layer
# --------------------------------------------------------------------------- #

def text_layer_units(page: pymupdf.Page, page_no: int, arr: np.ndarray,
                     skipped: Counter) -> list[Unit]:
    image_rects = [pymupdf.Rect(info["bbox"]) for info in page.get_image_info()]
    units: list[Unit] = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:                      # 0 = text, 1 = image
            continue
        rect = pymupdf.Rect(block["bbox"])
        lines = [l for l in block["lines"] if any(s["text"].strip() for s in l["spans"])]
        if not lines:
            continue
        text = _join_lines(["".join(s["text"] for s in l["spans"]) for l in lines])
        if not _has_letters(text):
            continue
        if any(abs(l["dir"][1]) > 0.1 for l in lines):
            skipped["rotated text (kept original)"] += 1
            continue
        if any(rect.intersects(img) and rect.intersect(img).get_area() > 0.2 * rect.get_area()
               for img in image_rects):
            skipped["text over an image (kept original)"] += 1
            continue

        spans = [s for l in lines for s in l["spans"] if s["text"].strip()]
        sizes = sorted(s["size"] for s in spans)
        size = sizes[len(sizes) // 2]
        ink_weights: Counter = Counter()
        for s in spans:
            ink_weights[s["color"]] += len(s["text"])
        ink_text = _int_to_rgb01(ink_weights.most_common(1)[0][0])

        pixels = _region(arr, rect, TEXT_SAMPLE_SCALE)
        if pixels is None or len(pixels) == 0:
            continue
        bg = _median_bg(pixels)
        if _contrast(_rel_lum(ink_text), _rel_lum(bg)) < MIN_TEXT_CONTRAST:
            skipped["text not clearly visible (low contrast, kept original)"] += 1
            continue

        units.append(Unit(page=page_no, rect=rect, text=text, size=size, bg=bg,
                          ink=_pick_ink(bg, ink_text), source="text"))
    return units


# --------------------------------------------------------------------------- #
# Extraction: OCR for scanned pages
# --------------------------------------------------------------------------- #

class OCREngine:
    def __init__(self, languages: list[str], device: str):
        import easyocr
        self.reader = easyocr.Reader(languages, gpu=(device == "cuda"), verbose=False)

    def read_lines(self, arr: np.ndarray) -> list[tuple[list, str, float]]:
        return self.reader.readtext(arr, detail=1, paragraph=False)


def _merge_same_row(lines: list[OcrLine]) -> list[OcrLine]:
    """Join OCR boxes that sit on the same text row (EasyOCR often splits one line).

    Boxes are merged only when they overlap vertically and are close horizontally,
    so the two columns of a page are never joined.
    """
    if len(lines) < 2:
        return lines
    med_h = float(np.median([l.rect.height for l in lines]))
    changed = True
    while changed:
        changed = False
        out: list[OcrLine] = []
        for line in sorted(lines, key=lambda l: (l.rect.y0, l.rect.x0)):
            merged = False
            for other in out:
                overlap = min(line.rect.y1, other.rect.y1) - max(line.rect.y0, other.rect.y0)
                min_h = min(line.rect.height, other.rect.height)
                hgap = max(line.rect.x0 - other.rect.x1, other.rect.x0 - line.rect.x1)
                if min_h > 0 and overlap >= 0.5 * min_h and hgap <= 1.5 * med_h:
                    left, right = sorted([line, other], key=lambda l: l.rect.x0)
                    out.remove(other)
                    out.append(OcrLine(
                        rect=pymupdf.Rect(left.rect) | right.rect,
                        text=f"{left.text} {right.text}",
                        conf=min(left.conf, right.conf),
                        bg=(left.bg + right.bg) / 2,
                        contrast=min(left.contrast, right.contrast),
                    ))
                    merged = changed = True
                    break
            if not merged:
                out.append(line)
        lines = out
    return lines


def _group_ocr_lines(lines: list[OcrLine]) -> list[list[OcrLine]]:
    """Group OCR lines into paragraph-like blocks (works with multi-column pages)."""
    if not lines:
        return []
    line_h = float(np.median([l.rect.height for l in lines]))
    groups: list[list[OcrLine]] = []
    boxes: list[pymupdf.Rect] = []
    for line in sorted(lines, key=lambda l: (l.rect.y0, l.rect.x0)):
        best = None
        for i, box in enumerate(boxes):
            gap = line.rect.y0 - box.y1
            overlap = min(line.rect.x1, box.x1) - max(line.rect.x0, box.x0)
            narrower = min(line.rect.width, box.width)
            if -0.5 * line_h <= gap <= 0.9 * line_h and overlap > 0.3 * narrower:
                if best is None or box.y1 > boxes[best].y1:
                    best = i
        if best is None:
            groups.append([line])
            boxes.append(pymupdf.Rect(line.rect))
        else:
            groups[best].append(line)
            boxes[best] |= line.rect
    return groups


def ocr_units(page: pymupdf.Page, page_no: int, engine: OCREngine,
              min_conf: float, skipped: Counter) -> list[Unit]:
    arr = render_page(page, OCR_SCALE)
    raw = engine.read_lines(arr)

    lines: list[OcrLine] = []
    for bbox, text, conf in raw:
        text = text.strip()
        if not text or not _has_letters(text):
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        rect = pymupdf.Rect(min(xs) / OCR_SCALE, min(ys) / OCR_SCALE,
                            max(xs) / OCR_SCALE, max(ys) / OCR_SCALE)
        pixels = _region(arr, rect, OCR_SCALE)
        if pixels is None or len(pixels) == 0:
            continue
        lum = _luminance_array(pixels)
        bg_lum = float(np.median(lum))
        ink_lum = float(np.percentile(lum, 2 if bg_lum > 0.5 else 98))
        lines.append(OcrLine(rect=rect, text=text, conf=float(conf),
                             bg=np.median(pixels, axis=0) / 255.0,
                             contrast=_contrast(bg_lum, ink_lum)))

    lines = _merge_same_row(lines)
    units: list[Unit] = []
    for group in _group_ocr_lines(lines):
        run: list[OcrLine] = []

        def flush() -> None:
            if not run:
                return
            rect = pymupdf.Rect(run[0].rect)
            for l in run[1:]:
                rect |= l.rect
            bg = _median_bg(np.array([l.bg * 255 for l in run]).reshape(-1, 3))
            median_h = float(np.median([l.rect.height for l in run]))
            units.append(Unit(page=page_no, rect=rect,
                              text=_join_lines([l.text for l in run]),
                              size=max(4.0, median_h * 0.8), bg=bg,
                              ink=_pick_ink(bg), source="ocr"))
            run.clear()

        for line in group:
            if line.conf < min_conf:
                flush()
                skipped[f"OCR confidence below {min_conf:.2f} (kept original)"] += 1
            elif line.contrast < MIN_OCR_CONTRAST:
                flush()
                skipped["OCR text not clearly visible (kept original)"] += 1
            else:
                run.append(line)
        flush()
    return units


# --------------------------------------------------------------------------- #
# Translation
# --------------------------------------------------------------------------- #

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


def split_segments(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split text into sentence-based chunks the model handles well."""
    segments: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            if current:
                segments.append(current)
                current = ""
            segments.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) > max_chars:
            segments.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        segments.append(current)
    return segments


class Translator:
    def __init__(self, device: str, batch_size: int, beams: int = 4, use_fp16: bool = True):
        import torch
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.beams = beams
        self.use_fp16 = use_fp16 and device == "cuda"
        self._models: dict[str, tuple] = {}

    def _load(self, lang: str):
        if lang in self._models:
            return self._models[lang]
        from transformers import MarianMTModel, MarianTokenizer

        tok = model = None
        used = None
        for name in (OPUS_MODEL.format(src=lang), FALLBACK_MODEL):
            try:
                tok = MarianTokenizer.from_pretrained(name)
                model = MarianMTModel.from_pretrained(name)
                used = name
                break
            except Exception:
                continue
        if model is None:
            raise RuntimeError(f"No translation model could be loaded for language '{lang}'.")
        model.eval()
        model.generation_config.max_length = None   # use max_new_tokens only
        if self.use_fp16:
            model.half()
        model.to(self.device)
        self._models[lang] = (tok, model, used)
        tqdm.write(f"  loaded {used} on {self.device}" + (" (fp16)" if self.use_fp16 else ""))
        return self._models[lang]

    def _generate(self, tok, model, texts: list[str]) -> list[str]:
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(self.device)
        max_new = min(512, int(enc["input_ids"].shape[1] * 2) + 16)
        with self.torch.inference_mode():
            out = model.generate(**enc, num_beams=self.beams, max_new_tokens=max_new)
        return tok.batch_decode(out, skip_special_tokens=True)

    def translate_segments(self, segments: list[tuple[str, str]]) -> list[str]:
        """segments: (source language code, text). Returns English texts in the same order."""
        results: list[str | None] = [None] * len(segments)
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, (lang, _) in enumerate(segments):
            groups[lang].append(idx)

        for lang, indices in groups.items():
            tok, model, _ = self._load(lang)
            order = sorted(indices, key=lambda i: len(segments[i][1]), reverse=True)
            bar = tqdm(total=len(order), desc=f"Translating [{lang}]", unit="seg")
            pos, bs = 0, self.batch_size
            while pos < len(order):
                batch = order[pos:pos + bs]
                try:
                    outputs = self._generate(tok, model, [segments[i][1] for i in batch])
                except self.torch.cuda.OutOfMemoryError:
                    self.torch.cuda.empty_cache()
                    if bs == 1:
                        raise
                    bs = max(1, bs // 2)
                    tqdm.write(f"  GPU memory full - retrying with batch size {bs}")
                    continue
                for i, text in zip(batch, outputs):
                    results[i] = text
                pos += len(batch)
                bar.update(len(batch))
            bar.close()
        return [r or "" for r in results]


def build_detector():
    from lingua import LanguageDetectorBuilder
    return LanguageDetectorBuilder.from_all_languages().build()


def _detect(detector, text: str) -> str | None:
    if len(text.split()) < 4:
        return None
    lang = detector.detect_language_of(text)
    return lang.iso_code_639_1.name.lower() if lang else None


def translate_units(units: list[Unit], translator: Translator, detector,
                    source_override: str | None) -> str:
    """Fill in unit.english for every unit that needs translating. Returns the dominant language."""
    for u in units:
        if source_override:
            u.lang = source_override
        elif detector is not None:
            u.lang = _detect(detector, u.text)

    counts = Counter(u.lang for u in units if u.lang)
    dominant = source_override or (counts.most_common(1)[0][0] if counts else "mul")
    for u in units:
        if u.lang is None:
            u.lang = dominant

    keys = list(dict.fromkeys((u.lang, u.text) for u in units
                              if u.lang != "en" and _has_letters(u.text)))
    segments: list[tuple[str, str]] = []
    owners: list[int] = []
    for ki, (lang, text) in enumerate(keys):
        for seg in split_segments(text):
            segments.append((lang, seg))
            owners.append(ki)

    translated = translator.translate_segments(segments) if segments else []
    pieces: dict[int, list[str]] = defaultdict(list)
    for ki, text in zip(owners, translated):
        pieces[ki].append(text)
    table = {keys[ki]: " ".join(" ".join(parts).split()) for ki, parts in pieces.items()}

    for u in units:
        if u.lang == "en":
            u.skip_reason = "already English"
        else:
            u.english = table.get((u.lang, u.text)) or None
    return dominant


# --------------------------------------------------------------------------- #
# Fitting English text into the original area
# --------------------------------------------------------------------------- #

def _find_font() -> str | None:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def _wrap(text: str, font: pymupdf.Font, size: float, width: float) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        candidate = word if not cur else f"{cur} {word}"
        if font.text_length(candidate, fontsize=size) <= width:
            cur = candidate
            continue
        if cur:
            lines.append(cur)
        while len(word) > 1 and font.text_length(word, fontsize=size) > width:
            k = len(word)
            while k > 1 and font.text_length(word[:k], fontsize=size) > width:
                k -= 1
            lines.append(word[:k])
            word = word[k:]
        cur = word
    if cur:
        lines.append(cur)
    return lines


def _fit(u: Unit, font: pymupdf.Font) -> bool:
    """Choose the largest font size (≤ original) at which the English text fits the area."""
    width = u.rect.width - 2
    if width < 10:
        return False
    asc, desc = font.ascender, font.descender
    size = u.size
    min_size = max(4.0, u.size * MIN_FONT_SCALE)
    while size >= min_size - 1e-6:
        lines = _wrap(u.english, font, size, width)
        line_h = size * LINE_SPACING
        need = asc * size + (len(lines) - 1) * line_h + abs(desc) * size
        if need <= u.rect.height + 2:
            u.layout = (lines, size, line_h, asc)
            return True
        size -= 0.5
    return False


# --------------------------------------------------------------------------- #
# Document editing
# --------------------------------------------------------------------------- #

def _is_scanned(page: pymupdf.Page) -> bool:
    if len(page.get_text("text").strip()) < MIN_TEXT_CHARS:
        return True
    page_area = page.rect.get_area()
    return any(pymupdf.Rect(info["bbox"]).get_area() >= BIG_IMAGE_FRACTION * page_area
               for info in page.get_image_info())


def translate_document(src: Path, dst: Path, *, translator: Translator, detector,
                       ocr_factory, source_override: str | None,
                       min_ocr_conf: float, after_reading=None) -> dict:
    start = time.perf_counter()
    doc = pymupdf.open(src)
    skipped: Counter = Counter()
    units: list[Unit] = []
    ocr_pages: list[int] = []

    # 1) Find what to translate, page by page
    for page_no in tqdm(range(len(doc)), desc="Reading pages", unit="page"):
        page = doc[page_no]
        if _is_scanned(page):
            ocr_pages.append(page_no)
            units += ocr_units(page, page_no, ocr_factory(), min_ocr_conf, skipped)
        else:
            arr = render_page(page, TEXT_SAMPLE_SCALE)
            units += text_layer_units(page, page_no, arr, skipped)

    # Free the OCR models before translating, so both never need to fit in memory at once
    if after_reading is not None:
        after_reading()

    # 2) Translate all units together
    dominant = "mul"
    if units:
        dominant = translate_units(units, translator, detector, source_override)

    # 3) Edit each page: cover the original, write the English in place
    fontfile = _find_font()
    font = pymupdf.Font(fontfile=fontfile) if fontfile else pymupdf.Font("helv")
    alias = "EN" if fontfile else "helv"

    by_page: dict[int, list[Unit]] = defaultdict(list)
    for u in units:
        by_page[u.page].append(u)

    translated = 0
    for page_no in sorted(set(by_page) | set(ocr_pages)):
        page = doc[page_no]
        page_units = by_page.get(page_no, [])
        accepted = []
        for u in page_units:
            if u.english is None:
                if u.skip_reason is None:
                    u.skip_reason = "no translation produced"
                continue
            if not _fit(u, font):
                u.skip_reason = "English text does not fit the original area (kept original)"
                continue
            accepted.append(u)

        if page_no in ocr_pages:
            # Remove the old (invisible or OCR) text layer; the scan image itself stays.
            page.add_redact_annot(page.rect)
        for u in accepted:
            page.add_redact_annot(pymupdf.Rect(u.rect) + (-0.3, -0.3, 0.3, 0.3), fill=u.bg)
        if accepted or page_no in ocr_pages:
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                                  graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
        # Redaction fills do not reliably paint over images (scanned pages), so paint
        # the background colour explicitly over every area we are replacing.
        for u in accepted:
            page.draw_rect(pymupdf.Rect(u.rect) + (-0.5, -0.5, 0.5, 0.5),
                           color=None, fill=u.bg, overlay=True, width=0)

        if accepted:
            if fontfile:
                page.insert_font(fontname=alias, fontfile=fontfile)
            for u in accepted:
                lines, size, line_h, asc = u.layout
                y = u.rect.y0 + asc * size
                for ln in lines:
                    page.insert_text(pymupdf.Point(u.rect.x0 + 1, y), ln,
                                     fontname=alias, fontsize=size, color=u.ink)
                    y += line_h
                translated += 1

    for u in units:
        if u.skip_reason and u.skip_reason != "already English":
            skipped[u.skip_reason] += 1

    try:
        doc.subset_fonts()
    except Exception:
        pass
    meta = {k: (v or "") for k, v in doc.metadata.items()}
    meta["subject"] = ("English translation (layout preserved), translated locally "
                       "with Helsinki-NLP Opus-MT (CC-BY-4.0)")
    doc.set_metadata(meta)
    doc.save(dst, garbage=3, deflate=True)
    doc.close()

    edited_pages = len(set(by_page) | set(ocr_pages))
    report_lines = [
        f"Source: {src.name}",
        f"Output: {dst.name}",
        f"Source language (detected): {LANGUAGE_NAMES.get(dominant, dominant)}",
        f"Pages edited: {edited_pages} ({len(ocr_pages)} scanned page(s) read with OCR)",
        f"Text areas translated: {translated}",
        "",
        "Kept in the original language:",
    ]
    report_lines += [f"  {count} x {reason}" for reason, count in skipped.most_common()] or ["  none"]
    if ocr_pages:
        report_lines.append("  (Scanned pages: anything the OCR could not read at all is left unchanged.)")
    report_lines += ["", "Text that was not translated:"]
    for u in units:
        if u.skip_reason and u.skip_reason != "already English":
            report_lines.append(f"  page {u.page + 1}: {u.skip_reason}: {u.text[:90]}")
    return {
        "translated": translated,
        "scanned_pages": len(ocr_pages),
        "kept": sum(skipped.values()),
        "language": dominant,
        "seconds": round(time.perf_counter() - start, 1),
        "report": "\n".join(report_lines) + "\n",
    }


class PDFTranslator:
    """Runs the pipeline for one or more PDFs, reusing loaded models."""

    def __init__(self, device: str, batch_size: int, beams: int = 4, use_fp16: bool = True,
                 source_lang: str | None = None, ocr_langs: list[str] | None = None,
                 min_ocr_conf: float = 0.6):
        self.device = device
        self.source_lang = source_lang
        self.ocr_langs = ocr_langs or ["en"]
        self.min_ocr_conf = min_ocr_conf
        self.translator = Translator(device, batch_size, beams, use_fp16)
        self._detector = None
        self._ocr: OCREngine | None = None

    def _get_ocr(self) -> OCREngine:
        if self._ocr is None:
            tqdm.write(f"  loading OCR models ({', '.join(self.ocr_langs)}) on {self.device}")
            self._ocr = OCREngine(self.ocr_langs, self.device)
        return self._ocr

    def _release_ocr(self) -> None:
        self._ocr = None
        gc.collect()

    def translate_file(self, src: Path, dst: Path) -> dict:
        if self.source_lang is None and self._detector is None:
            self._detector = build_detector()
        stats = translate_document(
            src, dst,
            translator=self.translator,
            detector=self._detector if self.source_lang is None else None,
            ocr_factory=self._get_ocr,
            source_override=self.source_lang,
            min_ocr_conf=self.min_ocr_conf,
            after_reading=self._release_ocr,
        )
        report_path = dst.with_name(f"{dst.stem}_report.txt")
        report_path.write_text(stats.pop("report"), encoding="utf-8")
        stats["report_path"] = report_path
        # Free the translation models so the next file starts with the least memory use
        self.translator._models.clear()
        gc.collect()
        return stats
