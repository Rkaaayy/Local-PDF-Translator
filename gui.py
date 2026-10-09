#!/usr/bin/env python3
"""Graphical interface for the local PDF translator.

Pick PDF files, choose the document language, set the OCR confidence threshold,
and translate. Everything runs on this PC; no document content leaves it.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

# When started with pythonw.exe (no console window), stdout/stderr are None.
# Send the technical output to a log file instead so nothing crashes.
if sys.stdout is None or sys.stderr is None:
    log_dir = APP_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    _log = open(log_dir / "translator.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _log

import tkinter as tk  # noqa: E402
import tkinter.font as tkfont  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402

from hardware import (default_batch_size, format_report, inspect_hardware,  # noqa: E402
                      resolve_device)

AUTO_LABEL = "Auto-detect"
DEFAULT_CONF = 0.6

# (code used for translation, name shown in the list, EasyOCR language code)
LANGUAGES = [
    ("de", "German", "de"),
    ("fr", "French", "fr"),
    ("es", "Spanish", "es"),
    ("it", "Italian", "it"),
    ("pt", "Portuguese", "pt"),
    ("nl", "Dutch", "nl"),
    ("pl", "Polish", "pl"),
    ("cs", "Czech", "cs"),
    ("sv", "Swedish", "sv"),
    ("da", "Danish", "da"),
    ("fi", "Finnish", "fi"),
    ("no", "Norwegian", "no"),
    ("tr", "Turkish", "tr"),
    ("ru", "Russian", "ru"),
    ("uk", "Ukrainian", "uk"),
    ("id", "Indonesian", "id"),
    ("vi", "Vietnamese", "vi"),
    ("ar", "Arabic", "ar"),
    ("hi", "Hindi", "hi"),
    ("bn", "Bengali", "bn"),
    ("ta", "Tamil", "ta"),
    ("te", "Telugu", "te"),
    ("mr", "Marathi", "mr"),
    ("gu", "Gujarati", None),      # EasyOCR cannot read this script
    ("kn", "Kannada", "kn"),
    ("ml", "Malayalam", None),     # EasyOCR cannot read this script
    ("pa", "Punjabi", None),       # EasyOCR cannot read this script
    ("ur", "Urdu", "ur"),
    ("or", "Odia", None),          # EasyOCR cannot read this script
    ("ja", "Japanese", "ja"),
    ("ko", "Korean", "ko"),
    ("zh", "Chinese (Simplified)", "ch_sim"),
    ("en", "English (no translation needed)", "en"),
]
LABEL_TO_CODE = {f"{name} ({code})": code for code, name, _ in LANGUAGES}
OCR_CODE = {code: ocr for code, _, ocr in LANGUAGES}
# With auto-detection the language of a scanned page is unknown, so read the
# common Latin-alphabet languages.
AUTO_OCR_LANGS = ["en", "de", "fr", "es", "it", "pt", "nl", "pl"]


def ocr_languages(lang: str | None) -> list[str]:
    """OCR languages for scanned pages. An empty list means OCR is not available."""
    if lang is None:
        return AUTO_OCR_LANGS
    if OCR_CODE[lang] is None:
        return []
    return list(dict.fromkeys([OCR_CODE[lang], "en"]))


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PDF Translator - runs locally on this PC")
        self.geometry("780x860")
        self.minsize(700, 780)

        self.files: list[Path] = []
        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.hardware = None
        self._logo_icon = None
        self._logo_header = None
        self._load_logo()

        self._build()
        threading.Thread(target=self._load_hardware, daemon=True).start()
        self.after(100, self._poll)

    # ------------------------------------------------------------------ UI
    def _load_logo(self) -> None:
        """Load the logo for the window icon (title bar / taskbar) and the header."""
        icon_path = APP_DIR / "assets" / "logo_icon.png"
        header_path = APP_DIR / "assets" / "logo_header.png"
        try:
            if icon_path.exists():
                self._logo_icon = tk.PhotoImage(file=str(icon_path))
                self.iconphoto(True, self._logo_icon)
            if header_path.exists():
                self._logo_header = tk.PhotoImage(file=str(header_path))
        except tk.TclError:
            self._logo_icon = None
            self._logo_header = None

    def _build(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        # Header with the logo and the app name
        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 6))
        if self._logo_header is not None:
            ttk.Label(header, image=self._logo_header).pack(side="left", padx=(0, 12))
        title_box = ttk.Frame(header)
        title_box.pack(side="left", fill="x")
        title_font = tkfont.Font(family="TkDefaultFont", size=18, weight="bold")
        ttk.Label(title_box, text="PDF Translator", font=title_font).pack(anchor="w")
        ttk.Label(title_box, text="Translate PDFs into English. Your documents never leave this PC.",
                  foreground="#555555").pack(anchor="w")

        # 1. Files
        files_box = ttk.LabelFrame(root, text="1. PDF files")
        files_box.pack(fill="x")
        self.listbox = tk.Listbox(files_box, height=7, selectmode="extended")
        self.listbox.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scroll = ttk.Scrollbar(files_box, command=self.listbox.yview)
        scroll.pack(side="left", fill="y", pady=8)
        self.listbox.config(yscrollcommand=scroll.set)
        buttons = ttk.Frame(files_box)
        buttons.pack(side="left", fill="y", padx=8, pady=8)
        ttk.Button(buttons, text="Add files...", command=self.add_files).pack(fill="x", pady=2)
        ttk.Button(buttons, text="Remove selected", command=self.remove_selected).pack(fill="x", pady=2)
        ttk.Button(buttons, text="Clear list", command=self.clear_files).pack(fill="x", pady=2)

        # 2. Options
        opts = ttk.LabelFrame(root, text="2. Options")
        opts.pack(fill="x", pady=(10, 0))
        opts.columnconfigure(1, weight=1)

        ttk.Label(opts, text="Language of the document:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.lang_var = tk.StringVar(value=AUTO_LABEL)
        self.lang_combo = ttk.Combobox(
            opts, textvariable=self.lang_var, state="readonly", width=34,
            values=[AUTO_LABEL] + list(LABEL_TO_CODE))
        self.lang_combo.grid(row=0, column=1, sticky="w", padx=8, pady=6)

        ttk.Label(opts, text="OCR confidence threshold:").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        slider_row = ttk.Frame(opts)
        slider_row.grid(row=1, column=1, sticky="w", padx=8, pady=6)
        self.conf_var = tk.DoubleVar(value=DEFAULT_CONF)
        ttk.Scale(slider_row, from_=0.0, to=1.0, variable=self.conf_var,
                  command=self._on_conf, length=300).pack(side="left")
        self.conf_label = ttk.Label(slider_row, text=f"{DEFAULT_CONF:.2f}", width=5)
        self.conf_label.pack(side="left", padx=8)
        ttk.Label(
            opts,
            text=("Scanned text the OCR is less sure about than this stays in its original "
                  "language. Lower = translate more text. Higher = keep more original text."),
            wraplength=560, foreground="#555555",
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 6))

        ttk.Label(opts, text="Save translated files to:").grid(row=3, column=0, sticky="w", padx=8, pady=6)
        out_row = ttk.Frame(opts)
        out_row.grid(row=3, column=1, sticky="ew", padx=8, pady=6)
        self.out_var = tk.StringVar(value="")
        ttk.Entry(out_row, textvariable=self.out_var).pack(side="left", fill="x", expand=True)
        ttk.Button(out_row, text="Browse...", command=self.pick_output).pack(side="left", padx=(6, 0))
        ttk.Label(opts, text="Leave empty to save each file next to the original.",
                  foreground="#555555").grid(row=4, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 6))

        # 3. Run
        action = ttk.Frame(root)
        action.pack(fill="x", pady=(12, 4))
        self.start_btn = ttk.Button(action, text="Translate", command=self.start)
        self.start_btn.pack(side="left")
        self.progress = ttk.Progressbar(action, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=12)
        self.status = ttk.Label(root, text="Ready.")
        self.status.pack(fill="x")

        log_frame = ttk.LabelFrame(root, text="Log")
        log_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.log_widget = tk.Text(log_frame, height=7, state="disabled", wrap="word")
        self.log_widget.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_widget.yview)
        log_scroll.pack(side="left", fill="y", pady=8)
        self.log_widget.config(yscrollcommand=log_scroll.set)

        self.hw_label = ttk.Label(root, text="Checking hardware...", foreground="#333333")
        self.hw_label.pack(fill="x", pady=(8, 0))

    def _on_conf(self, value: str) -> None:
        snapped = round(round(float(value) / 0.05) * 0.05, 2)
        self.conf_var.set(snapped)
        self.conf_label.config(text=f"{snapped:.2f}")

    def log(self, text: str) -> None:
        self.log_widget.config(state="normal")
        self.log_widget.insert("end", text + "\n")
        self.log_widget.see("end")
        self.log_widget.config(state="disabled")

    # --------------------------------------------------------- file list
    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Choose PDF files to translate",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        for p in paths:
            path = Path(p)
            if path not in self.files:
                self.files.append(path)
                self.listbox.insert("end", str(path))

    def remove_selected(self) -> None:
        for index in reversed(self.listbox.curselection()):
            del self.files[index]
            self.listbox.delete(index)

    def clear_files(self) -> None:
        self.files.clear()
        self.listbox.delete(0, "end")

    def pick_output(self) -> None:
        folder = filedialog.askdirectory(title="Choose the folder for translated files")
        if folder:
            self.out_var.set(folder)

    # --------------------------------------------------------- running
    def start(self) -> None:
        if self.busy:
            return
        if not self.files:
            messagebox.showwarning("No files", "Add at least one PDF file first.")
            return
        label = self.lang_var.get()
        lang = None if label == AUTO_LABEL else LABEL_TO_CODE[label]
        conf = float(self.conf_var.get())
        out_text = self.out_var.get().strip()
        out_dir = Path(out_text) if out_text else None

        self.busy = True
        self.start_btn.config(state="disabled")
        self.progress.config(maximum=len(self.files), value=0)
        self.log(f"Starting: {len(self.files)} file(s), language: {label}, "
                 f"OCR confidence threshold: {conf:.2f}")
        if lang is not None and OCR_CODE[lang] is None:
            self.log("Note: OCR is not available for this language. Scanned pages will be "
                     "left unchanged; text-based pages are still translated.")
        threading.Thread(target=self._worker, args=(list(self.files), lang, conf, out_dir),
                         daemon=True).start()

    def _worker(self, files: list[Path], lang: str | None, conf: float,
                out_dir: Path | None) -> None:
        try:
            hw = self.hardware or inspect_hardware()
            device = resolve_device("auto", hw)
            vram = hw.gpus[0].vram_mb if device == "cuda" and hw.gpus else 0
            batch = default_batch_size(device, vram)
            self.events.put(("log", format_report(hw, device)))

            from pipeline import PDFTranslator  # heavy imports happen off the UI thread
            translator = PDFTranslator(device=device, batch_size=batch, source_lang=lang,
                                       ocr_langs=ocr_languages(lang), min_ocr_conf=conf)

            for i, src in enumerate(files, 1):
                target_dir = out_dir or src.parent
                target_dir.mkdir(parents=True, exist_ok=True)
                dst = target_dir / f"{src.stem}_en.pdf"
                self.events.put(("status", f"Translating {i} of {len(files)}: {src.name}"))
                try:
                    stats = translator.translate_file(src, dst)
                    self.events.put(("log",
                                     f"Done: {src.name} -> {dst}\n"
                                     f"   {stats['translated']} text areas translated, "
                                     f"{stats['scanned_pages']} scanned page(s), "
                                     f"{stats['kept']} kept in original language, "
                                     f"{stats['seconds']} s. Report: {stats['report_path'].name}"))
                except Exception as err:
                    traceback.print_exc()
                    self.events.put(("log", f"ERROR with {src.name}: {err}"))
                self.events.put(("progress", i))
            self.events.put(("status", "Finished."))
        except Exception as err:
            traceback.print_exc()
            self.events.put(("error", str(err)))
        finally:
            self.events.put(("finished", None))

    def _load_hardware(self) -> None:
        hw = inspect_hardware()
        self.hardware = hw
        device = resolve_device("auto", hw)
        if device == "cuda":
            text = f"Processing device: GPU ({hw.gpus[0].name})"
        else:
            text = "Processing device: CPU (no usable NVIDIA GPU found)"
        self.events.put(("hw", text))

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "status":
                    self.status.config(text=payload)
                elif kind == "progress":
                    self.progress.config(value=payload)
                elif kind == "hw":
                    self.hw_label.config(text=payload)
                elif kind == "error":
                    messagebox.showerror("Translation failed", payload)
                elif kind == "finished":
                    self.busy = False
                    self.start_btn.config(state="normal")
        except queue.Empty:
            pass
        self.after(100, self._poll)


if __name__ == "__main__":
    App().mainloop()
