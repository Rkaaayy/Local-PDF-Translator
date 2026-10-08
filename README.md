# PDF Translator (local, free, GPU-accelerated, layout-preserving)

Translates PDF documents into English while keeping the original layout. The English
text is written into the same places as the original text, so images, tables,
colours, headers and footers all stay where they were.

Everything runs on your PC: no accounts, no API keys, no per-page costs, and your
documents never leave your machine.

- **Layout preserved:** the original PDF is edited in place. Each text block is covered with its background colour and the English is written in the same area, shrinking the font only when needed.
- **GPU support:** detects an NVIDIA GPU and uses it automatically (CUDA). Falls back to the CPU if none is found.
- **Scanned pages:** read with OCR (EasyOCR, GPU-accelerated). Only text that is clearly legible is translated.
- **Unclear text stays in its original language:** text the OCR is not confident about, low-contrast or invisible text, and text that does not fit its area is left untouched.
- **Report:** every run writes a `_report.txt` listing what was translated and what was kept, with reasons.

## 1. Requirements

- Python 3.10 to 3.13 (64-bit)
- For GPU use: an NVIDIA graphics card with a recent driver (check with `nvidia-smi` in a terminal)
- About 5 GB of free disk space for PyTorch and the models

## 2. Setup

### Windows (PowerShell)

```powershell
cd path\to\pdf-translator
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 1) PyTorch with CUDA. The default PyPI build is CPU-only on Windows,
#    so install this first.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2) Everything else
pip install -r requirements.txt
```

### Linux

```bash
cd path/to/pdf-translator
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

On Linux, the PyTorch CUDA wheels include the CUDA runtime, so you don't need to install the CUDA Toolkit separately. You only need a working NVIDIA driver.

> **Why torch first?** `torch` and `torchvision` must come from the same index and
> version. Installing them from the wrong index is the most common cause of
> "CPU only" or import errors.

## 3. Check your hardware

```bash
python translate_pdf.py --check-hardware
```

Example output with a GPU:

```
Hardware check
  PyTorch            : 2.14.1+cu126 (CUDA: 12.6)
  NVIDIA driver      : found
  CUDA usable        : yes
  GPU 0              : NVIDIA GeForce RTX 4070, 12282 MB VRAM, driver 566.36
  => Processing on   : GPU (CUDA)
```

If the GPU is not used, the report explains why. For example, it tells you to reinstall PyTorch with CUDA support if you have a GPU but a CPU-only PyTorch build.

## 4. Use it

```bash
# One file -> report_en.pdf (next to the original), plus report_en_report.txt
python translate_pdf.py report.pdf

# Choose the output file
python translate_pdf.py report.pdf -o english/report_en.pdf

# Translate every PDF in a folder
python translate_pdf.py ./documents -o ./english

# Scanned pages in German (OCR reads these languages)
python translate_pdf.py scan.pdf --ocr-langs de,en

# Stricter OCR: keep more lines in the original language (default 0.6)
python translate_pdf.py scan.pdf --ocr-langs de,en --min-ocr-confidence 0.8

# Force a device (useful for comparing speed)
python translate_pdf.py report.pdf --device cpu
python translate_pdf.py report.pdf --device cuda

# Skip auto-detection if you know the source language
python translate_pdf.py report.pdf --source-lang fr
```

Models are downloaded on first use and cached in your user folder: roughly 300 to 600 MB per translation language pair, plus about 100 MB for the OCR models. After that the tool works offline.

## 5. What happens to each part of the page

| Content | What the tool does |
|---|---|
| Normal text (with a text layer) | Covered with the local background colour and replaced by the English in the same place. Font size is kept, or reduced slightly (down to 60%) if the English is longer. |
| Text that is hard to see (for example white on white) | Left unchanged, and listed in the report. |
| Text over a photo or image | Left unchanged, and listed in the report. |
| Rotated text | Left unchanged, and listed in the report. |
| Images, lines, table borders, colours | Not changed. |
| Scanned pages | Read with OCR. Each line is checked for OCR confidence and contrast. Clear lines are translated; unclear lines keep the original pixels. |
| English text | Not changed. |
| English text that does not fit the original area even at 60% size | Left unchanged, and listed in the report. |

The `_report.txt` file lists the counts for each reason and the text of every area that was kept.

## 6. How GPU selection works

1. `nvidia-smi` checks that an NVIDIA driver is installed and lists the GPUs.
2. PyTorch checks that it can actually use CUDA.
3. If both pass, all translation and OCR runs on the GPU. Otherwise it runs on the CPU and prints the reason.
4. The translation batch size is chosen from your VRAM (for example, 32 segments on an 8 GB card). If the GPU runs out of memory mid-job, the batch size is halved automatically and the job continues.
5. Models run in fp16 on the GPU for speed. Use `--fp32` if you see odd output.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| "CUDA usable: no" but `nvidia-smi` works | You have the CPU-only PyTorch build. Run `pip uninstall torch torchvision`, then the `--index-url .../cu126` command again. |
| "could not initialise CUDA" | Update the NVIDIA driver to the latest version from nvidia.com, then restart the terminal. |
| `RuntimeError: operator torchvision::nms does not exist` | `torch` and `torchvision` versions don't match. Reinstall both from the same index. |
| Out-of-memory errors on the GPU | Use a smaller `--batch-size` (for example, 8), or `--beams 1`. |
| A scanned page is mostly left in German | Pass the right OCR languages with `--ocr-langs`, e.g. `--ocr-langs de,en`. Lower `--min-ocr-confidence` to translate more lines. |
| Translated text looks cramped | The English was too long for the original area, so it was shrunk. Those areas are listed in the report if they could not fit. |
| Translation looks wrong for a language | Opus-MT has a dedicated model for many languages. Others fall back to a multilingual model, which is weaker. |

## 8. Limitations

- Original fonts are not reused. The English uses Arial (Windows), Liberation Sans or DejaVu Sans (Linux) so that all characters display correctly.
- Very dense pages (for example, a long paragraph in a tiny font inside a small box) may keep some original text, which is listed in the report.
- Scanned pages keep their original image, so the English text is only as sharp as the scan itself.
- A page with both a full-page image and a text layer is treated as scanned: its text layer is replaced with OCR results.
- Machine translation is good for understanding a document, but it is not a certified translation. Check important text before relying on it.

## 9. Licenses of the components

- Opus-MT translation models: CC-BY-4.0 (attribution is included in the output PDF's subject field)
- EasyOCR: Apache-2.0
- PyMuPDF: AGPL-3.0 (fine for personal, non-commercial use; a commercial license is needed to distribute it in a closed product)
- PyTorch, Transformers: BSD / Apache-2.0
- lingua-language-detector: Apache-2.0

This project is for personal use, so no commercial licensing is needed. If you ever distribute it commercially, check each component's license again.
