# PDF Translator (local, free, GPU-accelerated, layout-preserving)

Translates PDF documents into English while keeping the original layout. The English
text is written into the same places as the original text, so images, tables,
colours, headers and footers all stay where they were.

## Why I built this

I wanted to translate contracts and other important documents without uploading them to an online service. Most translation tools send your files to a remote server, and once a document leaves your computer you no longer control where it goes, who can access it, or how long it is kept.

So this tool is designed to keep everything local:

- Your documents are processed on your own PC, from start to finish.
- Translation, OCR, and PDF editing run offline. The only internet access is a one-time download of the language and OCR models.
- No accounts, no API keys, and no third-party services are involved.

Your document content is never sent anywhere.

## What it does

- **Keeps the layout:** the original PDF is edited in place. Each text block is covered with its background colour and the English is written in the same area, shrinking the font only when needed.
- **Works on scanned pages:** scanned pages are read with OCR. Only text that is clearly legible is translated.
- **Leaves unclear text alone:** text the OCR is not confident about, low-contrast or invisible text, and text that does not fit its area stays in the original language.
- **Uses your NVIDIA GPU when it can:** it detects a usable GPU and uses it automatically. Without one, it runs on the CPU.
- **Writes a report:** every translation produces a `_report.txt` listing what was translated and what was kept, with reasons.
- **Has a window:** pick files, choose the document language, and set the OCR confidence threshold, all without typing commands.

## 1. Requirements

- **Windows 10 or 11, or Linux**
- **Python 3.10 to 3.13 (64-bit).** On Windows, get it from python.org and tick **"Add python.exe to PATH"** during installation. The window needs tkinter, which the python.org installer includes by default.
- **RAM:** 8 GB recommended. Scanned pages use the most memory.
- **Disk:** about 5 GB free for PyTorch and the models, plus about 0.5 GB per translation language.
- **For GPU use:** an NVIDIA graphics card with a recent driver. Check with `nvidia-smi` in a terminal.

## 2. Quick start on Windows

1. Download or copy the `pdf-translator` folder.
2. Double-click **`start_translator.bat`**.
3. The first run takes a while (see section 3). When it finishes, the window opens.
4. Add your PDFs, pick the language, and click **Translate**.

Later runs open the window straight away.

## 3. The `start_translator.bat` file

`start_translator.bat` is the Windows launcher. You don't need to type any commands.

**What it does on the first run:**

1. Creates a private Python environment in a `.venv` folder inside the project. Nothing is installed system-wide.
2. Upgrades pip, the Python package installer.
3. Installs PyTorch with CUDA support from the official PyTorch server. This is several GB. The CUDA build also works on PCs without an NVIDIA GPU, where it uses the CPU.
4. Installs the remaining libraries from `requirements.txt`.
5. Starts the window with `pythonw.exe`. This runs Python without a console window, so only the PDF Translator window appears.

**What it does on later runs:** once setup has finished, it writes a `.venv\setup_complete` marker file. On later runs it sees the marker, skips the setup, and opens the window immediately.

**Things to know:**

- **The first run needs a good internet connection** and can take a long time. The translation and OCR models download the first time you use each language, and that also needs internet access. After that, the tool works offline.
- **If setup fails,** the window stays open with the error message. Fix the cause (for example, no internet connection, or Python not on the PATH) and run the file again. Because the marker file is only written after a complete setup, the batch file runs the remaining steps again; packages that are already installed are skipped. To start from scratch, delete the `.venv` folder.
- **To update the libraries,** delete the `.venv` folder and run the batch file again. It will reinstall everything.
- **Technical messages** (for example, errors from the translation engine) go to `logs/translator.log` in the project folder, because the window has no console. The window's own log shows the important messages.
- **To make a desktop shortcut:** right-click `start_translator.bat`, choose **Send to > Desktop (create shortcut)**.

## 4. Using the window

**1. PDF files**
- **Add files...** opens a file browser. You can select several PDFs at once.
- **Remove selected** and **Clear list** change the list.

**2. Options**
- **Language of the document:** choose the language of your PDFs, or leave it on **Auto-detect**. Choosing the language is faster and more reliable, especially for scanned pages.
- **OCR confidence threshold:** the slider runs from 0.00 to 1.00 in steps of 0.05, and defaults to 0.60. OCR gives each line of a scanned page a confidence score. Lines scoring below the threshold stay in their original language.
  - **Lower** (for example, 0.40): more text is translated, with a higher risk of errors in hard-to-read scans.
  - **Higher** (for example, 0.80): only very clear text is translated, and more stays in the original language.
  - The slider has no effect on text PDFs, which have a text layer. It only applies to scanned pages.
- **Save translated files to:** leave this empty to save each `_en.pdf` next to its original. Or click **Browse...** to choose one folder for all results.

**3. Run**
- Click **Translate**. The progress bar and status line show the current file.
- The log shows the device in use, the result for each file, and the report file name.
- A translated file is named after the original with `_en` added, for example `contract_en.pdf`. The report is `contract_en_report.txt`.

**Language choices:** German, French, Spanish, Italian, Portuguese, Dutch, Polish, Czech, Swedish, Danish, Finnish, Norwegian, Turkish, Russian, Ukrainian, Indonesian, Vietnamese, Arabic, Hindi, Japanese, Korean, Chinese (Simplified), and English. Choosing English skips translation, so you can use it only for OCR.

> **Note on scanned pages with Auto-detect:** the tool reads the common Latin-alphabet languages (English, German, French, Spanish, Italian, Portuguese, Dutch, Polish). For scanned Russian, Ukrainian, Chinese, Japanese, Korean, Arabic, or Hindi documents, pick the language explicitly.

## 5. Command line (optional)

The same tool can be run from a terminal, after setup in the project folder:

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

Setup for Linux and the command line:

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

## 6. Check your hardware

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

## 7. What happens to each part of the page

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

## 8. How GPU selection works

1. `nvidia-smi` checks that an NVIDIA driver is installed and lists the GPUs.
2. PyTorch checks that it can actually use CUDA.
3. If both pass, all translation and OCR runs on the GPU. Otherwise it runs on the CPU and says why.
4. The translation batch size is chosen from your VRAM (for example, 32 segments on an 8 GB card). If the GPU runs out of memory mid-job, the batch size is halved automatically and the job continues.
5. Models run in fp16 on the GPU for speed. The command line's `--fp32` option turns this off if you see odd output.

The window shows which device is in use at the bottom of the screen.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| The batch file says Python was not found | Install Python from python.org and tick "Add python.exe to PATH". Then run the batch file again. |
| Setup fails partway through | Check your internet connection and run the batch file again. If it still fails, delete the `.venv` folder and start over. |
| The window doesn't open, and nothing appears | Check `logs/translator.log` for the error. |
| "CUDA usable: no" but `nvidia-smi` works | You have the CPU-only PyTorch build. Delete the `.venv` folder and run the batch file again (on Linux: `pip uninstall torch torchvision`, then reinstall with the `cu126` index). |
| "could not initialise CUDA" | Update the NVIDIA driver to the latest version from nvidia.com, then restart. |
| `RuntimeError: operator torchvision::nms does not exist` | `torch` and `torchvision` versions don't match. Reinstall both from the same index. |
| Out-of-memory errors on the GPU | Use a smaller `--batch-size` (for example, 8) or `--beams 1` on the command line. |
| The computer runs out of memory on scanned pages | Close other programs, and translate scanned files one at a time. |
| A scanned page is mostly left in its original language | Pick the document's language in the window, or lower the OCR confidence threshold to translate more lines. |
| Translated text looks cramped | The English was too long for the original area, so it was shrunk. Areas that could not fit at all are listed in the report. |
| Translation looks wrong for a language | Opus-MT has a dedicated model for many languages. Others fall back to a multilingual model, which is weaker. |

## 10. Limitations

- Original fonts are not reused. The English uses Arial (Windows), Liberation Sans, or DejaVu Sans (Linux) so that all characters display correctly.
- Very dense pages (for example, a long paragraph in a tiny font inside a small box) may keep some original text, which is listed in the report.
- Scanned pages keep their original image, so the English text is only as sharp as the scan.
- A page with both a full-page image and a text layer is treated as scanned: its text layer is replaced with OCR results.
- Machine translation is good for understanding a document, but it is not a certified translation. Check important text before relying on it.

## 11. Project files

| File | Purpose |
|---|---|
| `start_translator.bat` | Windows launcher: first-time setup, then opens the window |
| `gui.py` | The window: file list, language, OCR threshold, output folder |
| `translate_pdf.py` | Command-line version of the same tool |
| `pipeline.py` | Reading, OCR, translation, and editing of the PDF |
| `hardware.py` | GPU detection and device selection |
| `requirements.txt` | Python libraries (PyTorch is installed separately, see section 3) |
| `samples/` | Test PDFs and their translated outputs |
| `logs/` | Technical log written when the window runs without a console |

## 12. Licenses of the components

- Opus-MT translation models: CC-BY-4.0 (attribution is included in the output PDF's subject field)
- EasyOCR: Apache-2.0
- PyMuPDF: AGPL-3.0 (fine for personal, non-commercial use; a commercial license is needed to distribute it in a closed product)
- PyTorch, Transformers: BSD / Apache-2.0
- lingua-language-detector: Apache-2.0

This project is for personal use, so no commercial licensing is needed. If you ever distribute it commercially, check each component's license again.
