"""Tkinter desktop UI for toxic comment preprocessing, augmentation, and modeling."""

from __future__ import annotations

from pathlib import Path
import gc
import io
import threading
import time
from tkinter import END, Tk, filedialog, messagebox
from tkinter import ttk
import tkinter as tk
from typing import Callable

import pandas as pd

from auth import UserRecord, get_store
import tkinter.font as tkfont
import colorsys

from augmentation import AugmentConfig, TextAugmenter
from moderation import (
    LocalToxicityAnalyzer,
    ProfanitySanitizer,
    ProfanityScanner,
    ToxicityAnalysis,
    ToxicityTextMetricsBundle,
    metrics_from_llm_prediction,
    preprocess_and_analyze_single,
)
from preprocessing import PreprocessConfig, TextPreprocessor

# Rows shown in the analysis textarea; full TSV previews are very RAM-heavy in Tk.
UI_PREVIEW_MAX_ROWS = 500
# When suggesting preprocessing from stats, cap how many rows we scan at once.
SUGGEST_SAMPLE_MAX_ROWS = 30_000
# Single-message analyze: model cannot coherently classify an entire TSV preview.
MAX_SINGLE_ANALYZE_CHARS = 12_000
_HISTORY_METRIC_ORDER = (
    "Toxicity",
    "Severe Toxicity",
    "Identity Attack",
    "Insult",
    "Profanity",
    "Threat",
)


def _format_history_scores_line(*, overall: int, metrics: dict[str, int]) -> str:
    parts = [f"Overall: {int(overall)}%"]
    for name in _HISTORY_METRIC_ORDER:
        parts.append(f"{name}: {int(metrics[name])}%")
    return " | ".join(parts)


def _history_summary_column(summary: str) -> str:
    first = (summary or "").splitlines()[0].strip() if summary else ""
    if first.startswith("Overall:"):
        return first
    return first.replace("\n", " ")[:400]


def _history_analysis_preview(summary: str) -> str:
    text = (summary or "").strip()
    if not text:
        return "(No analysis result saved for this run.)"
    lines = text.splitlines()
    if lines and (lines[0].startswith("Overall:") or lines[0].startswith("Single:")):
        body = "\n".join(lines[1:]).strip()
        return body or text
    return text


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    """
    Convert HSL values (degrees, %, %) to #RRGGBB.

    Tkinter mainly wants hex colors; design spec is provided as HSL.
    """

    h_norm = (h % 360.0) / 360.0
    s_norm = max(0.0, min(1.0, s / 100.0))
    l_norm = max(0.0, min(1.0, l / 100.0))
    r, g, b = colorsys.hls_to_rgb(h_norm, l_norm, s_norm)
    return f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"


def _overall_toxicity_theme(overall: int) -> tuple[str, str]:
    """Return (accent foreground, soft card background) for overall score 0..100."""
    o = max(0, min(100, int(overall)))
    if o < 35:
        return _hsl_to_hex(142, 71, 45), _hsl_to_hex(142, 71, 95)
    if o < 70:
        return _hsl_to_hex(38, 92, 42), _hsl_to_hex(38, 86, 94)
    return _hsl_to_hex(0, 72, 46), _hsl_to_hex(0, 79, 96)


def _draw_rounded_rect(
    canvas: tk.Canvas,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    radius: int,
    *,
    fill: str,
    outline: str,
    width: int = 1,
) -> int:
    r = max(0, min(int(radius), int((x2 - x1) / 2), int((y2 - y1) / 2)))
    points = [
        x1 + r,
        y1,
        x2 - r,
        y1,
        x2,
        y1,
        x2,
        y1 + r,
        x2,
        y2 - r,
        x2,
        y2,
        x2 - r,
        y2,
        x1 + r,
        y2,
        x1,
        y2,
        x1,
        y2 - r,
        x1,
        y1 + r,
        x1,
        y1,
    ]
    return int(
        canvas.create_polygon(
            points,
            smooth=True,
            splinesteps=36,
            fill=fill,
            outline=outline,
            width=width,
        )
    )


class _RoundedCard(tk.Canvas):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        bg: str,
        border: str,
        radius: int = 12,
        border_width: int = 1,
        pad: tuple[int, int] = (14, 14),
        height: int | None = None,
    ) -> None:
        super().__init__(parent, bg=parent.cget("bg"), highlightthickness=0, bd=0)
        self._card_bg = bg
        self._border = border
        self._radius = radius
        self._border_width = border_width
        self._padx, self._pady = pad
        self._shape_id: int | None = None
        self._explicit_height = height

        if height is not None:
            self.configure(height=height)

        self.inner = tk.Frame(self, bg=self._card_bg)
        self._inner_window = self.create_window(
            (self._padx, self._pady),
            window=self.inner,
            anchor="nw",
        )

        self.bind("<Configure>", self._redraw)
        self.inner.bind("<Configure>", self._on_inner_configure)

    def set_card_background(self, bg: str) -> None:
        """Update card fill and inner frame (e.g. toxicity score band)."""
        self._card_bg = bg
        self.inner.configure(bg=bg)
        self._paint_card_shape()

    def _paint_card_shape(self) -> None:
        w = max(1, int(self.winfo_width()))
        h = max(1, int(self.winfo_height()))
        self.delete("card_shape")
        self._shape_id = _draw_rounded_rect(
            self,
            1,
            1,
            w - 1,
            h - 1,
            self._radius,
            fill=self._card_bg,
            outline=self._border,
            width=self._border_width,
        )
        self.addtag_withtag("card_shape", self._shape_id)
        self.tag_lower("card_shape")
        self.coords(self._inner_window, self._padx, self._pady)
        self.itemconfigure(self._inner_window, width=max(1, w - (self._padx * 2)))

    def _on_inner_configure(self, _event: tk.Event | None = None) -> None:
        # Canvas does not reliably grow vertically with embedded content; taller inner
        # (e.g. toxicity preview Text) clips widgets packed below unless we resize.
        if self._explicit_height is not None:
            return
        self.update_idletasks()
        inner_h = int(self.inner.winfo_reqheight())
        needed = max(1, inner_h + (self._pady * 2))
        cur = int(self.winfo_height())
        if abs(cur - needed) > 1:
            self.configure(height=needed)

    def _redraw(self, _event: tk.Event) -> None:
        self._paint_card_shape()


class _RoundedButton(tk.Canvas):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        text: str,
        command: callable,
        bg: str,
        fg: str,
        border: str | None = None,
        radius: int = 12,
        font: tuple[str, int, str] = ("Segoe UI", 10, "bold"),
        padx: int = 18,
        pady: int = 10,
        hover_bg: str | None = None,
        active_bg: str | None = None,
        enabled: bool = True,
        canvas_bg: str | None = None,
    ) -> None:
        super().__init__(
            parent,
            bg=canvas_bg or parent.cget("bg"),
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self._text = text
        self._command = command
        self._bg = bg
        self._fg = fg
        self._border = border or bg
        self._radius = radius
        self._font = font
        self._padx = padx
        self._pady = pady
        self._hover_bg = hover_bg or bg
        self._active_bg = active_bg or self._hover_bg

        self._enabled = enabled
        self._bg_enabled = self._bg
        self._fg_enabled = self._fg
        self._border_enabled = self._border
        self._hover_bg_enabled = self._hover_bg
        self._active_bg_enabled = self._active_bg
        self._disabled_bg = "#e5e7eb"
        self._disabled_fg = "#9ca3af"
        self._disabled_border = "#d1d5db"

        self._shape_id: int | None = None
        self._text_id: int | None = None

        self.bind("<Configure>", self._draw)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

        self.set_enabled(self._enabled)

    def _draw(self, _event: tk.Event) -> None:
        w = max(1, int(self.winfo_width()))
        h = max(1, int(self.winfo_height()))
        self.delete("all")
        self._shape_id = _draw_rounded_rect(
            self,
            1,
            1,
            w - 1,
            h - 1,
            self._radius,
            fill=self._bg,
            outline=self._border,
            width=1,
        )
        self._text_id = int(
            self.create_text(
                w // 2,
                h // 2,
                text=self._text,
                fill=self._fg,
                font=self._font,
            )
        )

    def _set_bg(self, color: str) -> None:
        if self._shape_id is not None:
            self.itemconfigure(self._shape_id, fill=color)

    def _on_enter(self, _event: tk.Event) -> None:
        if not self._enabled:
            return
        self._set_bg(self._hover_bg)

    def _on_leave(self, _event: tk.Event) -> None:
        if not self._enabled:
            return
        self._set_bg(self._bg)

    def _on_click(self, _event: tk.Event) -> None:
        if not self._enabled:
            return
        self._set_bg(self._active_bg)
        self.after(120, lambda: self._set_bg(self._hover_bg))
        self._command()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if self._enabled:
            self._bg = self._bg_enabled
            self._fg = self._fg_enabled
            self._border = self._border_enabled
            self._hover_bg = self._hover_bg_enabled
            self._active_bg = self._active_bg_enabled
            self.configure(cursor="hand2")
        else:
            self._bg = self._disabled_bg
            self._fg = self._disabled_fg
            self._border = self._disabled_border
            self._hover_bg = self._disabled_bg
            self._active_bg = self._disabled_bg
            self.configure(cursor="arrow")

        # Redraw immediately if already laid out.
        if self.winfo_width() > 1 and self.winfo_height() > 1:
            self._draw(tk.Event())

    def autosize(self) -> None:
        # Approximate width based on text length (keeps layout stable without measuring fonts).
        w = max(90, int(len(self._text) * 8 + (self._padx * 2)))
        h = max(34, int(18 + (self._pady * 2)))
        self.configure(width=w, height=h)

    def set_text(self, text: str) -> None:
        self._text = text
        self.autosize()
        if self.winfo_width() > 1 and self.winfo_height() > 1:
            self._draw(tk.Event())


class ToxicCommentApp:
    def __init__(self, root: Tk, *, user: UserRecord, on_logout: Callable[[], None]) -> None:
        self.root = root
        self.root.title("Toxic Comment Classification Workbench")
        self.root.geometry("1280x800")
        self.root.minsize(1100, 700)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TCombobox", padding=5, relief="flat")
        style.configure("TEntry", padding=5)
        style.configure("TButton", padding=(10, 6), font=("Segoe UI", 10))
        style.configure("TLabel", font=("Segoe UI", 10))

        self._user = user
        self._on_logout = on_logout
        self._store = get_store()

        self.df: pd.DataFrame | None = None
        self._dataset_path: Path | None = None

        self.preprocessor = TextPreprocessor()
        self.augmenter = TextAugmenter(seed=42)
        self.profanity_scanner = ProfanityScanner()
        self.profanity_sanitizer = ProfanitySanitizer(self.profanity_scanner.terms)
        # Local model weights in project model/ (see LocalToxicityAnalyzer). Falls back if load/inference fails.
        self.toxicity_analyzer = LocalToxicityAnalyzer()
        self._toxicity_analysis_busy = False
        self._toxicity_model_ready = False
        self._toxicity_cancel_event = threading.Event()
        self._toxicity_download_payload: str | None = None
        self.model_manager = None
        self._modeling_available = False
        self._modeling_error: str | None = None
        try:
            # Lazy import so the UI can start even if SciPy/sklearn aren't installed yet.
            from models import ModelManager  # type: ignore

            self.model_manager = ModelManager(random_state=42)
            self._modeling_available = True
        except Exception as exc:
            self._modeling_error = str(exc)

        self.file_var = tk.StringVar()
        self.text_col_var = tk.StringVar()
        self.label_col_var = tk.StringVar()
        self.model_var = tk.StringVar(value="Logistic Regression")
        self.aug_strength_var = tk.DoubleVar(value=0.10)
        self.clean_mode_var = tk.StringVar(value="mask")
        self.clean_mask_var = tk.StringVar(value="[censored]")

        self.prep_vars = {
            "lowercase": tk.BooleanVar(value=False),
            "remove_punctuation": tk.BooleanVar(value=False),
            "remove_stopwords": tk.BooleanVar(value=False),
            "remove_numbers": tk.BooleanVar(value=False),
            "normalize_whitespace": tk.BooleanVar(value=False),
            "normalize_unicode": tk.BooleanVar(value=False),
            "remove_noise": tk.BooleanVar(value=False),
            "reduce_elongations": tk.BooleanVar(value=False),
            "map_slang": tk.BooleanVar(value=False),
            "expand_contractions": tk.BooleanVar(value=False),
        }
        self.aug_vars = {
            "synonym_replacement": tk.BooleanVar(value=False),
            "random_insertion": tk.BooleanVar(value=False),
            "random_swap": tk.BooleanVar(value=False),
            "random_deletion": tk.BooleanVar(value=False),
            "back_translation": tk.BooleanVar(value=False),
            "paraphrasing": tk.BooleanVar(value=False),
            "sentence_shuffling": tk.BooleanVar(value=False),
            "sentence_cropping": tk.BooleanVar(value=False),
            "noise_injection": tk.BooleanVar(value=False),
        }

        self._build_ui()
        # Load the local toxicity model after login (startup). This may take a bit on CPU.
        self.root.after(50, self._warmup_local_toxicity_model)

    def _warmup_local_toxicity_model(self) -> None:
        self._log("Loading local toxicity model (background)...")
        analyzer = self.toxicity_analyzer

        def worker() -> None:
            err: BaseException | None = None
            try:
                analyzer.preload()
            except BaseException as exc:
                err = exc

            def on_main() -> None:
                if err is not None:
                    self._toxicity_model_ready = False
                    self._log(f"Local toxicity model failed to load: {err}")
                    return
                self._toxicity_model_ready = True
                self._set_toxicity_analyze_ready()
                try:
                    import torch

                    if torch.cuda.is_available():
                        self._log(f"Local toxicity model loaded on GPU: {torch.cuda.get_device_name(0)}")
                    else:
                        self._log("Local toxicity model loaded on CPU.")
                except Exception:
                    self._log("Local toxicity model loaded.")

            self.root.after(0, on_main)

        threading.Thread(target=worker, daemon=True).start()

    def _build_ui(self) -> None:
        # Shared design tokens (keep sidebar + right panel consistent)
        self._tox_blue = _hsl_to_hex(221, 83, 53)
        self._tox_green = _hsl_to_hex(142, 71, 45)
        self._tox_green_bg = _hsl_to_hex(142, 71, 95)
        self._tox_border = "#e5e7eb"
        self._tox_text = "#111827"
        self._tox_muted = "#6b7280"
        self._tox_bg = "#f9fafb"
        self._tox_panel_bg = "#f3f4f6"
        self._tox_card_bg = "#ffffff"

        self.root.configure(bg=self._tox_bg)
        ttk_style = ttk.Style(self.root)
        ttk_style.configure("TFrame", background=self._tox_bg)

        container = tk.Frame(self.root, bg=self._tox_bg, padx=10, pady=10)
        container.pack(fill="both", expand=True)

        nav = tk.Frame(container, bg=self._tox_panel_bg, highlightthickness=0, bd=0)
        nav.pack(fill="x", pady=(0, 6))

        nav_btns = tk.Frame(nav, bg=self._tox_panel_bg, highlightthickness=0, bd=0)

        def _header_btn(text: str, command: Callable[[], None]) -> _RoundedButton:
            btn = _RoundedButton(
                nav_btns,
                text=text,
                bg="#ffffff",
                fg=self._tox_text,
                command=command,
                border=self._tox_border,
                radius=12,
                hover_bg="#f3f4f6",
                active_bg="#e5e7eb",
                canvas_bg=self._tox_panel_bg,
            )
            btn.autosize()
            return btn

        _header_btn("Analysis history", self._open_history_dialog).pack(side="left", padx=(0, 6))
        if self._user.is_admin:
            _header_btn("Manage users", self._open_admin_users_dialog).pack(side="left", padx=(0, 6))
            _header_btn("System statistics", self._open_admin_stats_dialog).pack(side="left", padx=(0, 6))
        _header_btn("Log out", self._logout).pack(side="left")
        nav_btns.pack(side="right", padx=(20, 0))

        user_strip = tk.Frame(nav, bg=self._tox_card_bg, highlightthickness=0, bd=0)
        tk.Label(
            user_strip,
            text=f"Signed in as {self._user.username}"
            + (" (administrator)" if self._user.is_admin else ""),
            font=("Segoe UI", 10),
            bg=self._tox_card_bg,
            fg=self._tox_muted,
        ).pack(anchor="w", padx=12, pady=8)
        user_strip.pack(side="left", fill="both", expand=True)

        middle = tk.Frame(container, bg=self._tox_bg)
        middle.pack(fill="both", expand=True, pady=10)

        left_wrapper = tk.Frame(middle, bg=self._tox_bg)
        left_wrapper.pack(side="left", fill="y")
        self.left_canvas = tk.Canvas(left_wrapper, width=380, highlightthickness=0, bg=self._tox_bg)
        left_scrollbar = ttk.Scrollbar(
            left_wrapper, orient="vertical", command=self.left_canvas.yview
        )
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)
        self.left_canvas.pack(side="left", fill="y")
        left_scrollbar.pack(side="left", fill="y")

        left = tk.Frame(self.left_canvas, bg=self._tox_bg)
        self.left_canvas_window = self.left_canvas.create_window(
            (0, 0), window=left, anchor="nw"
        )
        left.bind("<Configure>", self._on_left_frame_configure)
        self.left_canvas.bind("<Configure>", self._on_left_canvas_configure)
        self.left_canvas.bind("<Enter>", self._bind_mousewheel)
        self.left_canvas.bind("<Leave>", self._unbind_mousewheel)

        right = tk.Frame(middle, bg=self._tox_panel_bg)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        def _sidebar_card(parent: tk.Misc, title: str, *, height: int | None = None) -> tk.Frame:
            card = _RoundedCard(
                parent,
                bg=self._tox_card_bg,
                border=self._tox_border,
                radius=12,
                pad=(12, 12),
                height=height,
            )
            card.pack(fill="x", padx=10, pady=(0, 10))
            inner = card.inner
            tk.Label(
                inner,
                text=title,
                font=("Segoe UI", 11, "bold"),
                bg=self._tox_card_bg,
                fg=self._tox_text,
            ).pack(anchor="w", pady=(0, 10))
            return inner

        def _divider(parent: tk.Misc) -> None:
            tk.Frame(parent, bg=self._tox_border, height=1).pack(fill="x", pady=(10, 10))

        prep_frame = _sidebar_card(left, "Preprocessing", height=430)
        prep_labels = [
            ("Unicode Normalize (NFKC)", "normalize_unicode"),
            ("Remove Noise (URLs, emails, HTML, @mentions)", "remove_noise"),
            ("Reduce Repeated Letters (soooo → soo)", "reduce_elongations"),
            ("Map Internet Slang (u→you, lol, …)", "map_slang"),
            ("Expand Contractions (don't→do not, …)", "expand_contractions"),
            ("Lowercasing", "lowercase"),
            ("Remove Punctuation", "remove_punctuation"),
            ("Remove Stopwords", "remove_stopwords"),
            ("Remove Numbers", "remove_numbers"),
            ("Normalize Extra Whitespace", "normalize_whitespace"),
        ]
        for label, key in prep_labels:
            tk.Checkbutton(
                prep_frame,
                text=label,
                variable=self.prep_vars[key],
                onvalue=True,
                offvalue=False,
                bg=self._tox_card_bg,
                fg=self._tox_text,
                activebackground=self._tox_card_bg,
                activeforeground=self._tox_text,
                selectcolor=self._tox_card_bg,
                anchor="w",
            ).pack(fill="x", anchor="w")

        _divider(prep_frame)
        suggest_btn = _RoundedButton(
            prep_frame,
            text="Suggest",
            bg="#ffffff",
            fg=self._tox_text,
            command=self._suggest_preprocessing,
            border=self._tox_border,
            radius=12,
            hover_bg="#f3f4f6",
            active_bg="#e5e7eb",
        )
        suggest_btn.configure(height=34)
        suggest_btn.pack(fill="x")

        self._prep_apply_btn = _RoundedButton(
            prep_frame,
            text="Apply",
            bg=self._tox_blue,
            fg="#ffffff",
            command=self._apply_preprocessing,
            border=self._tox_blue,
            radius=12,
            hover_bg=self._tox_blue,
            active_bg=self._tox_blue,
        )
        self._prep_apply_btn.configure(height=34)
        self._prep_apply_btn.pack(fill="x", pady=(8, 0))

        # Start disabled; enable once user selects any preprocessing option (or after Suggest sets options).
        self._prep_apply_btn.set_enabled(False)
        for _v in self.prep_vars.values():
            _v.trace_add("write", lambda *_: self._update_preprocessing_apply_state())

        aug_frame = _sidebar_card(left, "Data Augmentation", height=390)
        aug_labels = [
            ("Synonym Replacement", "synonym_replacement"),
            ("Random Insertion", "random_insertion"),
            ("Random Swap", "random_swap"),
            ("Random Deletion", "random_deletion"),
            ("Back Translation (Offline Approx.)", "back_translation"),
            ("Paraphrasing", "paraphrasing"),
            ("Sentence Shuffling", "sentence_shuffling"),
            ("Sentence Cropping/Truncation", "sentence_cropping"),
            ("Noise Injection", "noise_injection"),
        ]
        for label, key in aug_labels:
            tk.Checkbutton(
                aug_frame,
                text=label,
                variable=self.aug_vars[key],
                onvalue=True,
                offvalue=False,
                bg=self._tox_card_bg,
                fg=self._tox_text,
                activebackground=self._tox_card_bg,
                activeforeground=self._tox_text,
                selectcolor=self._tox_card_bg,
                anchor="w",
            ).pack(fill="x", anchor="w")

        _divider(aug_frame)
        strength_row = tk.Frame(aug_frame, bg=self._tox_card_bg)
        strength_row.pack(fill="x")
        tk.Label(
            strength_row,
            text="Strength",
            font=("Segoe UI", 9),
            bg=self._tox_card_bg,
            fg=self._tox_muted,
        ).pack(side="left")
        ttk.Spinbox(
            strength_row,
            from_=0.01,
            to=0.50,
            increment=0.01,
            textvariable=self.aug_strength_var,
            width=8,
        ).pack(side="right")

        apply_aug_btn = _RoundedButton(
            aug_frame,
            text="Apply Augmentation",
            bg=self._tox_blue,
            fg="#ffffff",
            command=self._apply_augmentation,
            border=self._tox_blue,
            radius=12,
            hover_bg=self._tox_blue,
            active_bg=self._tox_blue,
        )
        apply_aug_btn.configure(height=34)
        apply_aug_btn.pack(fill="x", pady=(10, 0))
        self._aug_apply_btn = apply_aug_btn

        if self._user.is_admin:
            model_frame = _sidebar_card(left, "Model Investigation")
            tk.Label(
                model_frame,
                text="Training dataset — load CSV/Excel/JSON with text + labels",
                font=("Segoe UI", 9),
                bg=self._tox_card_bg,
                fg=self._tox_muted,
                wraplength=320,
                justify="left",
            ).pack(anchor="w", pady=(0, 8))
            load_train_btn = _RoundedButton(
                model_frame,
                text="Load training dataset",
                bg="#ffffff",
                fg=self._tox_text,
                command=self._admin_load_training_dataset,
                border=self._tox_border,
                radius=12,
                hover_bg="#f3f4f6",
                active_bg="#e5e7eb",
            )
            load_train_btn.configure(height=34)
            load_train_btn.pack(fill="x", pady=(0, 8))
            tk.Label(
                model_frame,
                text="File:",
                font=("Segoe UI", 9),
                bg=self._tox_card_bg,
                fg=self._tox_muted,
            ).pack(anchor="w")
            tk.Label(
                model_frame,
                textvariable=self.file_var,
                font=("Segoe UI", 9),
                bg=self._tox_card_bg,
                fg=self._tox_text,
                wraplength=320,
                justify="left",
            ).pack(anchor="w", pady=(0, 10))
            tk.Label(
                model_frame,
                text="Text column",
                font=("Segoe UI", 9),
                bg=self._tox_card_bg,
                fg=self._tox_muted,
            ).pack(anchor="w")
            self.text_col_combo = ttk.Combobox(
                model_frame,
                textvariable=self.text_col_var,
                state="readonly",
                values=[],
                width=30,
            )
            self.text_col_combo.pack(fill="x", pady=(4, 8))
            tk.Label(
                model_frame,
                text="Label column",
                font=("Segoe UI", 9),
                bg=self._tox_card_bg,
                fg=self._tox_muted,
            ).pack(anchor="w")
            self.label_col_combo = ttk.Combobox(
                model_frame,
                textvariable=self.label_col_var,
                state="readonly",
                values=[],
                width=30,
            )
            self.label_col_combo.pack(fill="x", pady=(4, 12))
            _divider(model_frame)
            investigate_btn = _RoundedButton(
                model_frame,
                text="Investigate Architectures",
                bg="#ffffff",
                fg=self._tox_text,
                command=self._investigate_architectures,
                border=self._tox_border,
                radius=12,
                hover_bg="#f3f4f6",
                active_bg="#e5e7eb",
            )
            investigate_btn.configure(height=34)
            investigate_btn.pack(fill="x", pady=(0, 10))
            tk.Label(
                model_frame,
                text="Selected Architecture",
                font=("Segoe UI", 9),
                bg=self._tox_card_bg,
                fg=self._tox_muted,
            ).pack(anchor="w")
            self.model_combo = ttk.Combobox(
                model_frame,
                textvariable=self.model_var,
                state="readonly",
                values=list(self.model_manager.model_factories.keys())
                if (self._modeling_available and self.model_manager is not None)
                else [],
                width=30,
            )
            self.model_combo.pack(fill="x", pady=(6, 10))
            self.train_button = _RoundedButton(
                model_frame,
                text="Train Selected Model",
                bg=self._tox_blue,
                fg="#ffffff",
                command=self._train_selected,
                border=self._tox_blue,
                radius=12,
                hover_bg=self._tox_blue,
                active_bg=self._tox_blue,
            )
            self.train_button.configure(height=34)
            self.train_button.pack(fill="x")
            if not self._modeling_available:
                self.model_combo.configure(state="disabled")
                self.train_button.set_enabled(False)
                investigate_btn.set_enabled(False)

        # Right side: unified interface (Preview + Logs + Toxicity Analysis)
        merged = tk.Frame(right, bg=self._tox_panel_bg, highlightthickness=0, bd=0)
        merged.pack(fill="both", expand=True)
        self._build_toxicity_tab(merged)

    def _build_toxicity_tab(self, parent: ttk.Frame) -> None:
        # Design colors (from user spec)
        self._tox_blue = _hsl_to_hex(221, 83, 53)  # Analyze button
        self._tox_green = _hsl_to_hex(142, 71, 45)  # Low toxicity accent/text
        # Light background for the overall card (use same hue but much lighter)
        self._tox_green_bg = _hsl_to_hex(142, 71, 95)
        self._tox_border = "#e5e7eb"
        self._tox_text = "#111827"
        self._tox_muted = "#6b7280"
        self._tox_bg = self._tox_panel_bg
        self._tox_card_bg = "#ffffff"

        header_bar = tk.Frame(parent, bg=self._tox_bg)
        header_bar.pack(fill="x", side="top")
        header_inner = tk.Frame(header_bar, bg=self._tox_bg)
        header_inner.pack(fill="x", padx=18, pady=(10, 6))
        tk.Label(
            header_inner,
            text="Toxicity analysis",
            font=("Segoe UI", 14, "bold"),
            bg=self._tox_bg,
            fg=self._tox_text,
        ).pack(side="left")

        self.toxicity_download_btn = _RoundedButton(
            header_inner,
            text="Download",
            bg="#ffffff",
            fg=self._tox_text,
            command=self._download_toxicity_analysis,
            border=self._tox_border,
            radius=12,
            hover_bg="#f3f4f6",
            active_bg="#e5e7eb",
            canvas_bg=self._tox_bg,
            enabled=False,
        )
        self.toxicity_download_btn.autosize()
        self.toxicity_download_btn.pack(side="right")

        scroll_host = tk.Frame(parent, bg=self._tox_bg)
        scroll_host.pack(fill="both", expand=True)

        canvas = tk.Canvas(scroll_host, bg=self._tox_bg, highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(scroll_host, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Keep content centered with a max width to create left/right whitespace on wide windows.
        self._tox_max_width = 1000
        body = tk.Frame(canvas, bg=self._tox_bg)
        canvas_window = canvas.create_window((0, 0), window=body, anchor="nw")

        def on_body_configure(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event: tk.Event) -> None:
            available = max(1, int(event.width))
            content_w = min(available, int(self._tox_max_width))
            x = max(0, int((available - content_w) / 2))
            canvas.itemconfigure(canvas_window, width=content_w)
            canvas.coords(canvas_window, x, 0)

        body.bind("<Configure>", on_body_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        tk.Label(
            body,
            text="Enter a comment or upload a file to analyze for toxic content.",
            font=("Segoe UI", 10),
            bg=self._tox_bg,
            fg=self._tox_muted,
        ).pack(pady=(12, 14))

        # Input card
        input_card = _RoundedCard(body, bg=self._tox_card_bg, border=self._tox_border, radius=12, pad=(14, 14))
        input_card.pack(fill="x", padx=18, pady=(0, 16))
        input_card_inner = input_card.inner

        tox_input_wrap = tk.Frame(input_card_inner, bg=self._tox_card_bg)
        tox_input_wrap.pack(fill="x")
        tox_input_wrap.columnconfigure(0, weight=1)
        tox_input_wrap.rowconfigure(0, weight=1)

        # Text height is in lines, not pixels — grow the visible preview by ~150px using font metrics.
        _tox_font_spec = ("Segoe UI", 10)
        _line_px = max(
            1,
            int(tkfont.Font(master=body, font=_tox_font_spec).metrics("linespace")),
        )
        _extra_lines = max(1, int(round(150 / _line_px)))
        _tox_input_height_lines = 6 + _extra_lines

        self.tox_input = tk.Text(
            tox_input_wrap,
            height=_tox_input_height_lines,
            wrap="word",
            font=_tox_font_spec,
            bd=0,
            highlightthickness=1,
            highlightbackground=self._tox_border,
            highlightcolor=self._tox_border,
            padx=10,
            pady=10,
        )
        self._dataset_busy = False
        tox_input_v = ttk.Scrollbar(tox_input_wrap, orient="vertical", command=self.tox_input.yview)
        self.tox_input.configure(yscrollcommand=tox_input_v.set)
        self.tox_input.grid(row=0, column=0, sticky="nsew")
        tox_input_v.grid(row=0, column=1, sticky="ns")
        self.tox_input.bind("<KeyRelease>", self._mark_preview_dirty, add="+")
        self._dataset_progress_var = tk.StringVar(value="Ready")
        tk.Label(
            input_card_inner,
            textvariable=self._dataset_progress_var,
            anchor="w",
            bg=self._tox_card_bg,
            fg=self._tox_muted,
            font=("Segoe UI", 9),
        ).pack(fill="x", pady=(6, 0))

        # Dataset controls keep large jobs bounded while still allowing full runs.
        dataset_opts = tk.Frame(input_card_inner, bg=self._tox_card_bg)
        dataset_opts.pack(fill="x", pady=(10, 0))
        tk.Label(dataset_opts, text="Dataset mode", bg=self._tox_card_bg, fg=self._tox_muted).pack(side="left")
        self._dataset_mode_var = tk.StringVar(value="Full dataset")
        ttk.Combobox(
            dataset_opts,
            textvariable=self._dataset_mode_var,
            values=("Full dataset", "First N rows", "Random sample"),
            state="readonly",
            width=16,
        ).pack(side="left", padx=(8, 14))
        tk.Label(dataset_opts, text="Rows", bg=self._tox_card_bg, fg=self._tox_muted).pack(side="left")
        self._dataset_limit_var = tk.StringVar(value="1000")
        ttk.Entry(dataset_opts, textvariable=self._dataset_limit_var, width=8).pack(side="left", padx=(8, 0))
        self._preview_segment_var = tk.StringVar(value="Segment 1")
        ttk.Button(dataset_opts, text="Next", command=self._next_preview_segment).pack(side="right", padx=(4, 0))
        ttk.Button(dataset_opts, text="Previous", command=self._previous_preview_segment).pack(side="right", padx=(4, 0))
        tk.Label(dataset_opts, textvariable=self._preview_segment_var, bg=self._tox_card_bg, fg=self._tox_muted).pack(side="right")

        actions = tk.Frame(input_card_inner, bg=self._tox_card_bg)
        actions.pack(fill="x", pady=(12, 0))
        actions.columnconfigure(0, weight=1)

        self.analyze_btn = _RoundedButton(
            actions,
            text="Analyze",
            bg=self._tox_blue,
            fg="#ffffff",
            command=self._analyze_toxicity_text,
            border=self._tox_blue,
            radius=12,
            hover_bg=self._tox_blue,
            active_bg=self._tox_blue,
        )
        self.analyze_btn.autosize()
        self.analyze_btn.set_enabled(False)
        self.analyze_btn.grid(row=0, column=0, sticky="we", padx=(0, 10))

        self.cancel_analysis_btn = _RoundedButton(
            actions,
            text="Cancel",
            bg="#ffffff",
            fg="#b91c1c",
            command=self._cancel_toxicity_analysis,
            border="#fecaca",
            radius=12,
            hover_bg="#fef2f2",
            active_bg="#fee2e2",
            enabled=False,
        )
        self.cancel_analysis_btn.autosize()
        self.cancel_analysis_btn.grid(row=0, column=1, sticky="e", padx=(0, 10))

        self.upload_btn = _RoundedButton(
            actions,
            text="Upload File",
            bg="#ffffff",
            fg=self._tox_text,
            command=self._upload_toxicity_file,
            border=self._tox_border,
            radius=12,
            hover_bg="#f3f4f6",
            active_bg="#e5e7eb",
        )
        self.upload_btn.autosize()
        self.upload_btn.grid(row=0, column=2, sticky="e")

        # Results area (HIDDEN until user clicks Analyze)
        self._tox_results_wrap = tk.Frame(body, bg=self._tox_bg)

        tk.Label(
            self._tox_results_wrap,
            text="Analysis Results",
            font=("Segoe UI", 14, "bold"),
            bg=self._tox_bg,
            fg=self._tox_text,
        ).pack(anchor="w", padx=18, pady=(8, 10))

        # Overall card (green) - fixed height for consistent layout
        overall = _RoundedCard(
            self._tox_results_wrap,
            bg=self._tox_green_bg,
            border=self._tox_border,
            radius=12,
            pad=(0, 0),
            height=170,
        )
        overall.pack(fill="x", padx=18, pady=(0, 14))
        overall_inner = overall.inner
        self._overall_score_card = overall

        self._overall_score_title_label = tk.Label(
            overall_inner,
            text="Overall Toxicity Score",
            font=("Segoe UI", 10),
            bg=self._tox_green_bg,
            fg=self._tox_muted,
        )
        self._overall_score_title_label.pack(pady=(18, 2))

        self.overall_pct_label = tk.Label(
            overall_inner,
            text="0%",
            font=("Segoe UI", 36, "bold"),
            bg=self._tox_green_bg,
            fg=self._tox_green,
        )
        self.overall_pct_label.pack()

        self.overall_badge = tk.Label(
            overall_inner,
            text="Low Toxicity",
            font=("Segoe UI", 10, "bold"),
            bg=self._tox_green_bg,
            fg=self._tox_green,
            padx=12,
            pady=4,
        )
        self.overall_badge.pack(pady=(4, 18))

        # Metric cards (2 rows x 3)
        metrics_wrap = tk.Frame(self._tox_results_wrap, bg=self._tox_bg)
        metrics_wrap.pack(fill="x", padx=18, pady=(0, 18))
        for c in range(3):
            metrics_wrap.columnconfigure(c, weight=1, uniform="m")

        self.metric_widgets: dict[str, dict[str, object]] = {}
        metric_defs = [
            ("Toxicity", "Overall toxicity indicators"),
            ("Severe Toxicity", "Extreme toxic content signals"),
            ("Identity Attack", "Identity-based attack signals"),
            ("Insult", "Insulting or demeaning language signals"),
            ("Profanity", "Profanity usage signals"),
            ("Threat", "Threat-related signals"),
        ]

        for idx, (title, subtitle) in enumerate(metric_defs):
            r, c = divmod(idx, 3)
            card = _RoundedCard(
                metrics_wrap,
                bg=self._tox_card_bg,
                border=self._tox_border,
                radius=12,
                pad=(12, 12),
                height=140,
            )
            card.grid(row=r, column=c, sticky="nsew", padx=8, pady=8)
            inner = card.inner

            header = tk.Frame(inner, bg=self._tox_card_bg)
            header.pack(fill="x")
            tk.Label(header, text=title, font=("Segoe UI", 10, "bold"), bg=self._tox_card_bg, fg=self._tox_text).pack(
                side="left"
            )
            pct = tk.Label(header, text="0%", font=("Segoe UI", 10, "bold"), bg=self._tox_card_bg, fg=self._tox_green)
            pct.pack(side="right")

            bar_outer = tk.Frame(inner, bg="#eef2f7")
            bar_outer.pack(fill="x", pady=(10, 10))
            bar_fill = tk.Frame(bar_outer, bg=self._tox_green, width=0, height=6)
            bar_fill.pack(side="left", fill="y")
            bar_outer.pack_propagate(False)
            bar_outer.configure(height=6)

            tk.Label(inner, text=subtitle, font=("Segoe UI", 9), bg=self._tox_card_bg, fg=self._tox_muted).pack(
                anchor="w"
            )

            self.metric_widgets[title] = {"pct": pct, "bar_outer": bar_outer, "bar_fill": bar_fill}

        # Keep results hidden until first Analyze click
        self._tox_results_visible = False

        # Logs / Results (placed AFTER analysis + analysis results)
        tk.Label(
            body,
            text="Logs / Results",
            font=("Segoe UI", 14, "bold"),
            bg=self._tox_bg,
            fg=self._tox_text,
        ).pack(anchor="w", padx=18, pady=(10, 10))

        logs_card = _RoundedCard(
            body,
            bg=self._tox_card_bg,
            border=self._tox_border,
            radius=12,
            pad=(12, 12),
            height=240,
        )
        logs_card.pack(fill="x", padx=18, pady=(0, 18))
        logs_inner = logs_card.inner
        self.result_text = tk.Text(
            logs_inner,
            height=11,
            wrap="word",
            bd=0,
            padx=10,
            pady=8,
        )
        lo_fg, _ = _overall_toxicity_theme(10)
        med_fg, _ = _overall_toxicity_theme(50)
        hi_fg, _ = _overall_toxicity_theme(85)
        self.result_text.tag_configure("tox_summary_low", foreground=lo_fg)
        self.result_text.tag_configure("tox_summary_med", foreground=med_fg)
        self.result_text.tag_configure("tox_summary_high", foreground=hi_fg)
        logs_v = ttk.Scrollbar(logs_inner, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=logs_v.set)
        self.result_text.grid(row=0, column=0, sticky="nsew")
        logs_v.grid(row=0, column=1, sticky="ns")
        logs_inner.rowconfigure(0, weight=1)
        logs_inner.columnconfigure(0, weight=1)

    def _upload_toxicity_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[
                ("Text datasets", "*.txt"),
            ]
        )
        if not path:
            return
        p = Path(path)
        self._dataset_path = p
        self.file_var.set(p.name)

        # Treat .txt uploads as row-based datasets: one line = one comment to analyze.
        if p.suffix.lower() == ".txt":
            self._load_text_dataset_path(p)
            return

        # If it's a dataset, load it and show in the shared Preview section.
        if p.suffix.lower() in {".csv", ".xlsx", ".xls", ".json"}:
            self._load_dataset_path(p)
            return

        # Otherwise treat it as analysis text input.
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            messagebox.showerror("Upload Error", str(exc))
            return
        content = content.strip()
        self.tox_input.delete("1.0", END)
        self.tox_input.insert("1.0", content)
        self._show_text_preview(content)

    def _load_text_dataset_path(self, path: Path) -> None:
        if not path.exists():
            messagebox.showerror("Error", "Please select a valid text file.")
            return
        self._dataset_path = path
        self.file_var.set(path.name)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            return

        rows = content.splitlines()
        self.df = pd.DataFrame({"comment_text": rows})
        self.text_col_var.set("comment_text")
        self.label_col_var.set("")
        self._show_preview(self.df)
        self._log(f"Loaded text dataset with {len(rows):,} rows from file: {path.name}")
        self._log("Text analysis mode: one line is treated as one row/comment.")
        self._refresh_admin_training_columns()
        self._scan_profanity()

    def _set_toxicity_download_ready(self, payload: str | None) -> None:
        self._toxicity_download_payload = payload
        btn = getattr(self, "toxicity_download_btn", None)
        if btn is not None:
            btn.set_enabled(bool(payload))

    def _download_toxicity_analysis(self) -> None:
        payload = (self._toxicity_download_payload or "").strip()
        if not payload:
            messagebox.showerror("Download", "Run an analysis before exporting.")
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save toxicity analysis",
            initialfile="toxicity_analysis.txt",
        )
        if not path:
            return
        try:
            Path(path).write_text(payload, encoding="utf-8")
            self._log(f"Saved toxicity analysis to {path}")
        except Exception as exc:
            messagebox.showerror("Download", str(exc))

    def _cancel_toxicity_analysis(self) -> None:
        if not self._toxicity_analysis_busy:
            return
        self._toxicity_cancel_event.set()
        # Release the UI immediately. The worker will observe the event between
        # batches; any late result is discarded by the completion handlers.
        self._toxicity_analysis_busy = False
        self._set_toxicity_analyze_loading(False)
        btn = getattr(self, "cancel_analysis_btn", None)
        if btn is not None:
            btn.set_text("Cancelling...")
            btn.set_enabled(False)
        self._log("Analysis cancelled. Cleaning up the active worker in the background.")

    def _set_toxicity_analyze_ready(self) -> None:
        btn = getattr(self, "analyze_btn", None)
        if btn is not None and not self._toxicity_analysis_busy:
            btn.set_text(getattr(self, "_analyze_btn_idle_label", "Analyze"))
            btn.set_enabled(self._toxicity_model_ready)

    def _set_toxicity_analyze_loading(self, loading: bool) -> None:
        btn = getattr(self, "analyze_btn", None)
        if btn is None:
            return
        up = getattr(self, "upload_btn", None)
        cancel = getattr(self, "cancel_analysis_btn", None)
        if loading:
            self._analyze_btn_idle_label = getattr(self, "_analyze_btn_idle_label", "Analyze")
            btn.set_text("Analyzing...")
            btn.set_enabled(False)
            if up is not None:
                up.set_enabled(False)
            if cancel is not None:
                cancel.set_text("Cancel")
                cancel.set_enabled(True)
            self._set_toxicity_download_ready(None)
            self.root.update_idletasks()
        else:
            btn.set_text(getattr(self, "_analyze_btn_idle_label", "Analyze"))
            btn.set_enabled(self._toxicity_model_ready)
            if up is not None:
                up.set_enabled(True)
            if cancel is not None:
                cancel.set_text("Cancel")
                cancel.set_enabled(False)
            self.root.update_idletasks()

    def _text_for_single_toxicity_analyze(self, text_raw: str) -> tuple[str, str | None]:
        """If the box holds a dataset preview (TSV), classify only the first row's text column.

        Sending the entire preview makes small local models return collapsed, similar outputs.
        """
        s = text_raw.strip()
        if not s.startswith("Dataset:") or "\n\n" not in s:
            if len(s) > MAX_SINGLE_ANALYZE_CHARS:
                return (
                    s[:MAX_SINGLE_ANALYZE_CHARS],
                    f"Input truncated to {MAX_SINGLE_ANALYZE_CHARS:,} characters for the model.",
                )
            return s, None
        _, rest = s.split("\n\n", 1)
        rest = rest.strip()
        if not rest:
            return s, None
        try:
            df = pd.read_csv(io.StringIO(rest), sep="\t", nrows=1)
            text_col = self.text_col_var.get().strip()
            cell = ""
            if not df.empty and len(df.columns) > 0:
                if text_col and text_col in df.columns:
                    cell = str(df.loc[0, text_col])
                else:
                    cell = str(df.iloc[0, 0])
            if not str(cell).strip():
                lines = [ln for ln in rest.splitlines() if ln.strip()]
                if len(lines) >= 2:
                    hdr = lines[0].split("\t")
                    vals = lines[1].split("\t")
                    if text_col and text_col in hdr:
                        i = hdr.index(text_col)
                        if i < len(vals):
                            cell = vals[i]
                    elif vals:
                        cell = vals[0]
            cell = str(cell).strip()
            if not cell:
                return s, None
            note = (
                "Dataset preview detected: analyzed only the first row's text column "
                "(not the full table). Clear the box and paste one message to analyze something else."
            )
            if len(cell) > MAX_SINGLE_ANALYZE_CHARS:
                return (
                    cell[:MAX_SINGLE_ANALYZE_CHARS],
                    note + f" Row text truncated to {MAX_SINGLE_ANALYZE_CHARS:,} characters.",
                )
            return cell, note
        except Exception:
            return s, None

    def _analyze_dataset_records(
        self,
        *,
        records: list[tuple[int, str]],
        source_col: str,
        prep: PreprocessConfig,
    ) -> dict[str, object]:
        total_rows = len(records)
        metric_sums = {name: 0 for name in _HISTORY_METRIC_ORDER}
        overall_sum = 0
        analyzed = 0
        skipped = 0
        errors: list[str] = []
        row_lines = [
            "row_number\toverall\ttoxicity\tsevere_toxicity\tidentity_attack\tinsult\tprofanity\tthreat\tlevel\tcategory\ttoxicity_type\ttext_preview"
        ]
        top_examples: list[tuple[int, int, str]] = []
        progress_every = max(1, min(100, total_rows // 20))
        started_at = time.monotonic()
        cancelled = False
        processed_positions = 0
        prepared_records: list[tuple[int, str, str]] = []
        for row_number, text in records:
            raw = str(text).strip()
            if raw:
                prepared_records.append((row_number, raw, self.preprocessor.apply(raw, prep)))
            else:
                skipped += 1

        batches = self.toxicity_analyzer.token_budgeted_batches(
            [(row_number, processed) for row_number, _raw, processed in prepared_records]
        )
        raw_by_row = {row_number: raw for row_number, raw, _processed in prepared_records}
        self.root.after(
            0,
            lambda batch_count=len(batches), row_count=len(prepared_records): self._log(
                f"Batch mode enabled: {row_count:,} non-empty rows packed into {batch_count:,} model calls."
            ),
        )

        def handle_result(row_number: int, raw: str, analysis: ToxicityAnalysis) -> None:
            nonlocal analyzed, overall_sum, top_examples
            metrics, overall = metrics_from_llm_prediction(
                toxicity_level=analysis.toxicity_level,
                toxicity_types=list(analysis.toxicity_type),
                categories=list(analysis.category),
            )
            analyzed += 1
            overall_sum += int(overall)
            for metric_name in _HISTORY_METRIC_ORDER:
                metric_sums[metric_name] += int(metrics.get(metric_name, 0))

            preview = raw.replace("\n", " ").replace("\t", " ")[:180]
            cats = ", ".join(analysis.category) if analysis.category else "None"
            types = ", ".join(analysis.toxicity_type) if analysis.toxicity_type else "None"
            row_lines.append(
                "\t".join(
                    [
                        str(row_number),
                        str(int(overall)),
                        str(int(metrics["Toxicity"])),
                        str(int(metrics["Severe Toxicity"])),
                        str(int(metrics["Identity Attack"])),
                        str(int(metrics["Insult"])),
                        str(int(metrics["Profanity"])),
                        str(int(metrics["Threat"])),
                        str(analysis.toxicity_level),
                        cats,
                        types,
                        preview,
                    ]
                )
            )
            top_examples.append((int(overall), row_number, preview))
            top_examples = sorted(top_examples, key=lambda item: item[0], reverse=True)[:5]

        for batch_index, batch in enumerate(batches, start=1):
            if self._toxicity_cancel_event.is_set():
                cancelled = True
                break
            first_row = batch[0][0] if batch else 0
            batch_size = len(batch)
            if processed_positions == 0 or processed_positions % progress_every == 0:
                elapsed = max(0.001, time.monotonic() - started_at)
                pct = (processed_positions / total_rows) * 100 if total_rows else 100.0
                rate = processed_positions / elapsed
                self.root.after(
                    0,
                    lambda row=first_row,
                    done=processed_positions,
                    total=total_rows,
                    percent=pct,
                    ok=analyzed,
                    fail=len(errors),
                    seconds=elapsed,
                    rps=rate,
                    bi=batch_index,
                    bs=batch_size: self._report_dataset_progress(percent, done, total, ok, fail, seconds / 60, rps),
                )
            try:
                for row_number, analysis in self.toxicity_analyzer.analyze_batch(batch):
                    handle_result(row_number, raw_by_row.get(row_number, ""), analysis)
            except Exception as exc:
                errors.append(
                    f"Batch {batch_index} ({len(batch)} rows) failed; falling back row-by-row: {exc}"
                )
                for row_number, processed in batch:
                    if self._toxicity_cancel_event.is_set():
                        cancelled = True
                        break
                    try:
                        analysis = self.toxicity_analyzer.analyze(processed)
                        handle_result(row_number, raw_by_row.get(row_number, ""), analysis)
                    except Exception as row_exc:
                        errors.append(f"Row {row_number}: {row_exc}")
                if cancelled:
                    break

            processed_positions += len(batch)

            if self._toxicity_cancel_event.is_set():
                cancelled = True
                break

            if processed_positions == len(prepared_records) or processed_positions % progress_every == 0:
                elapsed = max(0.001, time.monotonic() - started_at)
                pct = (processed_positions / total_rows) * 100 if total_rows else 100.0
                rate = processed_positions / elapsed
                self.root.after(
                    0,
                    lambda done=processed_positions,
                    total=total_rows,
                    percent=pct,
                    ok=analyzed,
                    fail=len(errors),
                    seconds=elapsed,
                    rps=rate: self._report_dataset_progress(percent, done, total, ok, fail, seconds / 60, rps),
                )

        if analyzed <= 0:
            raise ValueError("No non-empty text rows could be analyzed.")

        avg_metrics = {
            name: int(round(metric_sums[name] / analyzed))
            for name in _HISTORY_METRIC_ORDER
        }
        avg_overall = int(round(overall_sum / analyzed))
        toxic_count = sum(
            1
            for line in row_lines[1:]
            if max(int(part) for part in line.split("\t")[1:8]) >= 20
        )
        non_toxic_count = analyzed - toxic_count

        summary = (
            "Dataset Toxicity Analysis Result (local model)\n"
            f"- Source file: {self._dataset_path.name if self._dataset_path else 'dataset'}\n"
            f"- Text column: {source_col}\n"
            f"- Status: {'Cancelled by user' if cancelled else 'Completed'}\n"
            f"- Rows in dataset: {total_rows:,}\n"
            f"- Comments analyzed: {analyzed:,}\n"
            f"- Empty/skipped rows: {skipped:,}\n"
            f"- Rows with errors: {len(errors):,}\n"
            f"- Toxic comments: {toxic_count:,}\n"
            f"- Non-toxic comments: {non_toxic_count:,}\n"
            f"- Average UI Score: {avg_overall}%\n"
            f"  - Toxicity: {avg_metrics['Toxicity']}%\n"
            f"  - Severe Toxicity: {avg_metrics['Severe Toxicity']}%\n"
            f"  - Identity Attack: {avg_metrics['Identity Attack']}%\n"
            f"  - Insult: {avg_metrics['Insult']}%\n"
            f"  - Profanity: {avg_metrics['Profanity']}%\n"
            f"  - Threat: {avg_metrics['Threat']}%"
        )
        if top_examples:
            summary += "\n\nTop toxic examples:"
            for overall, row_number, preview in top_examples:
                summary += f"\n- Row {row_number} ({overall}%): {preview}"
        if errors:
            summary += "\n\nRows that failed:"
            for err in errors[:20]:
                summary += f"\n- {err}"
            if len(errors) > 20:
                summary += f"\n- ... {len(errors) - 20:,} more"

        scores_line = _format_history_scores_line(overall=avg_overall, metrics=avg_metrics)
        download_payload = f"{scores_line}\n{summary}\n\nRow results:\n" + "\n".join(row_lines)
        return {
            "overall": avg_overall,
            "metrics": avg_metrics,
            "summary": summary,
            "download_payload": download_payload,
            "row_count": analyzed,
            "toxic_count": toxic_count,
            "non_toxic_count": non_toxic_count,
            "cancelled": cancelled,
        }

    def _set_dataset_analysis_pending_ui(self) -> None:
        self.overall_pct_label.configure(text="...", fg=self._tox_muted, bg=self._tox_green_bg)
        self.overall_badge.configure(text="Analyzing full dataset", fg=self._tox_muted, bg=self._tox_green_bg)
        self._overall_score_title_label.configure(bg=self._tox_green_bg, fg=self._tox_muted)
        self._overall_score_card.set_card_background(self._tox_green_bg)

        for widget_map in self.metric_widgets.values():
            pct_label: tk.Label = widget_map["pct"]  # type: ignore[assignment]
            fill: tk.Frame = widget_map["bar_fill"]  # type: ignore[assignment]
            pct_label.configure(text="...")
            fill.configure(width=0)

    def _analyze_toxicity_text(self) -> None:
        self._sync_preview_to_df()
        if self._toxicity_analysis_busy:
            return
        if not self._toxicity_model_ready:
            messagebox.showinfo("Analyze", "Please wait until the local toxicity model finishes loading.")
            return

        text_raw = self.tox_input.get("1.0", END).strip()
        if not text_raw:
            messagebox.showerror("Analyze Error", "Please enter text (or upload a file) first.")
            return

        if not getattr(self, "_tox_results_visible", False):
            self._tox_results_wrap.pack(fill="x", pady=(0, 6))
            self._tox_results_visible = True

        prep = self._build_prep_config()
        self._toxicity_analysis_busy = True
        self._toxicity_cancel_event.clear()
        self._set_toxicity_analyze_loading(True)

        is_dataset_preview = self.df is not None and getattr(self, "_preview_active", False)
        if is_dataset_preview:
            text_col = self.text_col_var.get().strip()
            if text_col not in self.df.columns:
                self._toxicity_analysis_busy = False
                self._set_toxicity_analyze_loading(False)
                messagebox.showerror("Analyze Error", "Please select or load a valid text column.")
                return

            records = [
                (idx + 1, value)
                for idx, value in enumerate(self.df[text_col].fillna("").astype(str).tolist())
            ]
            mode = getattr(self, "_dataset_mode_var", tk.StringVar(value="Full dataset")).get()
            if mode != "Full dataset":
                try:
                    limit = max(1, min(len(records), int(self._dataset_limit_var.get())))
                except (TypeError, ValueError):
                    messagebox.showerror("Analyze Error", "Rows must be a positive whole number.")
                    self._toxicity_analysis_busy = False
                    self._set_toxicity_analyze_loading(False)
                    return
                if mode == "Random sample":
                    import random
                    records = random.Random(42).sample(records, limit)
                    records.sort(key=lambda item: item[0])
                else:
                    records = records[:limit]
                self._log(f"Dataset mode: {mode.lower()} ({len(records):,} rows selected).")
            self._set_dataset_analysis_pending_ui()
            self._log(
                f"Dataset analysis started: analyzing all {len(records):,} rows from column '{text_col}'."
            )

            def dataset_worker() -> None:
                result: dict[str, object] | None = None
                err: BaseException | None = None
                try:
                    result = self._analyze_dataset_records(
                        records=records,
                        source_col=text_col,
                        prep=prep,
                    )
                except BaseException as exc:
                    err = exc
                rr, ee = result, err
                self.root.after(0, lambda r=rr, e=ee: self._complete_dataset_analyze_ui(r, e))

            threading.Thread(target=dataset_worker, daemon=True).start()
            return

        analyzed_text, extract_note = self._text_for_single_toxicity_analyze(text_raw)
        if extract_note:
            self._log(extract_note)

        def worker() -> None:
            bundle: ToxicityTextMetricsBundle | None = None
            err: BaseException | None = None
            try:
                bundle = preprocess_and_analyze_single(
                    analyzed_text,
                    preprocessor=self.preprocessor,
                    prep_config=prep,
                    toxicity_analyzer=self.toxicity_analyzer,
                )
            except BaseException as exc:
                err = exc
            br, er = bundle, err
            self.root.after(
                0,
                lambda tr=analyzed_text, bb=br, ee=er: self._complete_single_analyze_ui(tr, bb, ee),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _complete_dataset_analyze_ui(
        self,
        result: dict[str, object] | None,
        err: BaseException | None,
    ) -> None:
        if not self._toxicity_analysis_busy:
            return
        self._toxicity_analysis_busy = False
        self._set_toxicity_analyze_loading(False)
        if err is not None:
            messagebox.showerror("Analyze Error", str(err))
            self._log(f"Dataset Analyze Error: {err}")
            return
        assert result is not None

        overall = int(result["overall"])
        metrics = result["metrics"]
        assert isinstance(metrics, dict)
        clean_metrics = {str(k): int(v) for k, v in metrics.items()}
        summary = str(result["summary"])
        self._update_toxicity_ui(overall=overall, metrics=clean_metrics)
        self._log_toxicity_summary(summary, overall)
        if bool(result.get("cancelled")):
            self._log(f"Dataset analysis cancelled: {int(result['row_count']):,} rows analyzed.")
        else:
            self._log(f"Dataset analysis completed: {int(result['row_count']):,} rows analyzed.")

        self._store.insert_run(
            user_id=self._user.id,
            run_type="dataset",
            source_filename=self._dataset_path.name if self._dataset_path else None,
            row_count=int(result["row_count"]),
            toxic_count=int(result["toxic_count"]),
            non_toxic_count=int(result["non_toxic_count"]),
            summary=f"{_format_history_scores_line(overall=overall, metrics=clean_metrics)}\n{summary}",
        )
        self._set_toxicity_download_ready(str(result["download_payload"]))

    def _complete_single_analyze_ui(
        self,
        text_raw: str,
        bundle: ToxicityTextMetricsBundle | None,
        err: BaseException | None,
    ) -> None:
        """Apply analysis on the Tk main thread after background work completes."""
        if not self._toxicity_analysis_busy:
            return
        self._toxicity_analysis_busy = False
        self._set_toxicity_analyze_loading(False)
        if err is not None:
            messagebox.showerror("Analyze Error", str(err))
            self._log(f"Analyze Error: {err}")
            return
        assert bundle is not None

        metrics = bundle.metrics
        overall = bundle.overall
        lvl = bundle.lvl
        cats = bundle.cats_csv
        types_j = bundle.types_csv
        expl = bundle.explanation

        self._update_toxicity_ui(overall=overall, metrics=metrics)

        summary = (
            "Toxicity Analysis Result (local model)\n"
            f"- Category: {cats or 'None'}\n"
            f"- Toxicity Level: {lvl}\n"
            f"- Toxicity Type: {types_j or 'None'}\n"
        )
        if expl:
            summary += f"- Explanation: {expl}\n"
        summary += (
            f"- UI Score (derived): {overall}%\n"
            f"  - Toxicity: {metrics['Toxicity']}%\n"
            f"  - Severe Toxicity: {metrics['Severe Toxicity']}%\n"
            f"  - Identity Attack: {metrics['Identity Attack']}%\n"
            f"  - Insult: {metrics['Insult']}%\n"
            f"  - Profanity: {metrics['Profanity']}%\n"
            f"  - Threat: {metrics['Threat']}%"
        )
        self._log_toxicity_summary(summary, overall)

        self._record_single_run(
            text_raw=text_raw,
            overall=overall,
            metrics=metrics,
            analysis_summary=summary,
        )

        snippet = text_raw.strip().replace("\n", " ")[:500]
        scores_line = _format_history_scores_line(overall=overall, metrics=metrics)
        self._set_toxicity_download_ready(f"{scores_line}\n{summary.strip()}\nText preview: {snippet}")

    def _update_toxicity_ui(self, overall: int, metrics: dict[str, int]) -> None:
        overall = max(0, min(100, int(overall)))
        accent, card_bg = _overall_toxicity_theme(overall)
        self.overall_pct_label.configure(text=f"{overall}%", fg=accent, bg=card_bg)
        label = "Low Toxicity" if overall < 35 else ("Medium Toxicity" if overall < 70 else "High Toxicity")
        self.overall_badge.configure(text=label, fg=accent, bg=card_bg)
        self._overall_score_title_label.configure(bg=card_bg, fg=self._tox_muted)
        self._overall_score_card.set_card_background(card_bg)

        # Update metric cards + bars
        for title, value in metrics.items():
            w = self.metric_widgets.get(title)
            if not w:
                continue
            v = max(0, min(100, int(value)))
            pct_label: tk.Label = w["pct"]  # type: ignore[assignment]
            pct_label.configure(text=f"{v}%")

            outer: tk.Frame = w["bar_outer"]  # type: ignore[assignment]
            fill: tk.Frame = w["bar_fill"]  # type: ignore[assignment]

            # Ensure geometry is computed before sizing the fill.
            outer.update_idletasks()
            width = max(1, int(outer.winfo_width()))
            fill_w = int(width * (v / 100.0))
            fill.configure(width=fill_w)

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ]
        )
        if path:
            p = Path(path)
            self._dataset_path = p
            self.file_var.set(p.name)

    def _admin_load_training_dataset(self) -> None:
        """Admin: load CSV/Excel/JSON with text + labels for classical model training."""
        if not self._user.is_admin:
            return
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load training dataset",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_dataset_path(Path(path))

    def _refresh_admin_training_columns(self) -> None:
        if not hasattr(self, "text_col_combo"):
            return
        if self.df is None:
            self.text_col_combo.configure(values=())
            self.label_col_combo.configure(values=())
            return
        cols = tuple(self.df.columns.astype(str))
        self.text_col_combo.configure(values=cols)
        self.label_col_combo.configure(values=cols)
        tc = self.text_col_var.get().strip()
        if cols and tc not in cols:
            self.text_col_var.set(cols[0])
            tc = cols[0]
        lc = self.label_col_var.get().strip()
        if cols and lc and lc not in cols:
            self.label_col_var.set("")
        if cols and not self.label_col_var.get().strip():
            det = self._detect_label_column(self.df, tc)
            if det in cols:
                self.label_col_var.set(det)

    def _load_dataset_path(self, path: Path) -> None:
        if not path.exists():
            messagebox.showerror("Error", "Please select a valid dataset file.")
            return
        self._dataset_path = path
        self._set_dataset_busy(True)
        self.file_var.set(path.name)
        self._log(f"Loading dataset in background: {path.name}")

        def worker() -> None:
            try:
                if path.suffix.lower() == ".csv":
                    df = pd.read_csv(path, low_memory=True)
                elif path.suffix.lower() in {".xlsx", ".xls"}:
                    df = pd.read_excel(path)
                elif path.suffix.lower() == ".json":
                    df = pd.read_json(path)
                else:
                    raise ValueError("Unsupported file format.")
                self.root.after(0, lambda: self._finish_dataset_load(path, df, None))
            except Exception as exc:
                self.root.after(0, lambda: self._finish_dataset_load(path, None, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_dataset_load(self, path: Path, df: pd.DataFrame | None, error: Exception | None) -> None:
        if error is not None:
            self._set_dataset_busy(False)
            messagebox.showerror("Load Error", str(error))
            return
        assert df is not None
        self.df = df
        text_col = self._detect_text_column(self.df)
        label_col = self._detect_label_column(self.df, text_col)
        if text_col:
            self.text_col_var.set(text_col)
        if label_col:
            self.label_col_var.set(label_col)

        self._refresh_admin_training_columns()

        self._show_preview(self.df)
        self._log(f"Loaded dataset with shape: {self.df.shape}")
        if text_col:
            self._log(f"Detected text column: {text_col}")
        if label_col:
            self._log(f"Detected label column: {label_col}")
        # Do not block the UI while scanning every row.
        text_col_for_scan = text_col
        def scan_worker() -> None:
            try:
                result = self._scan_profanity_values(df, text_col_for_scan)
                self.root.after(0, lambda: self._finish_profanity_scan(result))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Scan Error", str(exc)))
        threading.Thread(target=scan_worker, daemon=True).start()

    def _scan_profanity_values(self, df: pd.DataFrame, text_col: str) -> tuple[list[bool], list[int], list[str]]:
        text_series = df[text_col].astype(str)
        has_pf: list[bool] = []
        prof_counts: list[int] = []
        prof_matches: list[str] = []
        for value in text_series:
            result = self.profanity_scanner.scan_text(value)
            has_pf.append(result.has_profanity)
            prof_counts.append(result.profanity_count)
            prof_matches.append(", ".join(result.matches))
        return has_pf, prof_counts, prof_matches

    def _finish_profanity_scan(self, result: tuple[list[bool], list[int], list[str]]) -> None:
        if self.df is None:
            return
        has_pf, prof_counts, prof_matches = result
        self.df["has_profanity"] = has_pf
        self.df["profanity_count"] = prof_counts
        self.df["profanity_matches"] = prof_matches
        flagged = int(sum(has_pf))
        total = len(has_pf)
        pct = (flagged / total * 100.0) if total else 0.0
        self._show_preview(self.df)
        self._log(f"Profanity scan completed on column: {self.text_col_var.get()}")
        self._log(f"Flagged rows: {flagged:,} / {total:,} ({pct:.2f}%)")
        self._set_dataset_busy(False)

    def _set_dataset_busy(self, busy: bool) -> None:
        self._dataset_busy = busy
        if hasattr(self, "_dataset_progress_var"):
            self._dataset_progress_var.set("Working… controls are temporarily locked" if busy else "Ready")
        state = "disabled" if busy else "normal"
        if hasattr(self, "tox_input"):
            self.tox_input.configure(state=state)
        for name in ("_prep_apply_btn", "_aug_apply_btn", "analyze_btn", "upload_btn"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.set_enabled(not busy)

    def _load_dataset(self) -> None:
        if self._dataset_path is None:
            raw_path = self.file_var.get().strip()
            if raw_path:
                self._dataset_path = Path(raw_path)

        if self._dataset_path is None:
            messagebox.showerror("Error", "Please browse file first.")
            return
        self._load_dataset_path(self._dataset_path)

    def _show_text_preview(self, text: str) -> None:
        # Preview panel removed; show in analysis input instead.
        if hasattr(self, "tox_input") and self.tox_input is not None:
            self._preview_df = None
            self._preview_active = False
            self._preview_text_snapshot = ""
            self._preview_dirty = False
            self.tox_input.delete("1.0", END)
            self.tox_input.insert("1.0", text)
            self.tox_input.see("1.0")

    def _show_preview(self, df: pd.DataFrame) -> None:
        # Preview panel removed; show dataset preview inside the analysis input.
        if not hasattr(self, "tox_input") or self.tox_input is None:
            return
        same_dataset = getattr(self, "_preview_df", None) is df
        self._preview_df = df
        self._preview_active = True
        if not same_dataset:
            self._preview_segment = 0
        self._preview_columns = None
        self.tox_input.delete("1.0", END)
        # Use TSV-style preview to avoid padded right-aligned DataFrame formatting.
        start = getattr(self, "_preview_segment", 0) * UI_PREVIEW_MAX_ROWS
        n_show = min(max(0, len(df) - start), UI_PREVIEW_MAX_ROWS)
        preview_df = df.iloc[start : start + n_show].fillna("").astype(str)
        # The editor shows only the source text column. Derived text and label
        # columns remain in the dataframe but are not editable preview content.
        source_preview_col = self.text_col_var.get().strip()
        if source_preview_col in preview_df.columns:
            preview_df = preview_df[[source_preview_col]]
        preview_df = preview_df.apply(
            lambda col: col.str.replace("\n", "\\n", regex=False).str.replace("\t", " ", regex=False)
        )
        # Hide profanity scan columns from preview (keep them in self.df).
        profanity_cols = ["has_profanity", "profanity_count", "profanity_matches"]
        drop_cols = [c for c in profanity_cols if c in preview_df.columns]
        if drop_cols:
            preview_df = preview_df.drop(columns=drop_cols)

        cols = list(preview_df.columns)
        text_col = self.text_col_var.get().strip()
        priority: list[str] = []
        if text_col and text_col in cols:
            priority.append(text_col)
        for name in ("processed_text", "augmented_text"):
            if name in cols and name not in priority:
                priority.append(name)
        for name in ("clean_text", "clean_replacements"):
            if name in cols and name not in priority:
                priority.append(name)
        rest = [c for c in cols if c not in priority]
        if priority:
            preview_df = preview_df[priority + rest]
        self._preview_columns = list(preview_df.columns)
        self._preview_next_row = n_show
        if len(preview_df.columns) == 1:
            preview_str = "\n".join(preview_df.iloc[:, 0].astype(str).tolist()) + "\n"
        else:
            preview_str = preview_df.to_csv(sep="\t", index=False, header=False)
        segment_end = min(len(df), start + n_show)
        extra = f" (rows {start + 1:,}–{segment_end:,})"
        header = (
            f"Dataset: {len(df):,} rows, {len(df.columns):,} columns{extra}\n\n"
        )
        self.tox_input.insert("1.0", preview_str)
        total_segments = max(1, (len(df) + UI_PREVIEW_MAX_ROWS - 1) // UI_PREVIEW_MAX_ROWS)
        self._preview_segment_var.set(f"Segment {getattr(self, '_preview_segment', 0) + 1} / {total_segments}")
        self._preview_text_snapshot = self.tox_input.get("1.0", END).strip()
        self._preview_dirty = False
        self.tox_input.see("1.0")

    def _mark_preview_dirty(self, _event: tk.Event | None = None) -> None:
        self._preview_dirty = True

    def _sync_preview_to_df(self) -> None:
        """Persist edits made in the paginated TSV preview into the in-memory dataframe."""
        if self.df is None or not hasattr(self, "tox_input"):
            return
        current = self.tox_input.get("1.0", END).strip()
        snapshot = getattr(self, "_preview_text_snapshot", "")
        if not getattr(self, "_preview_active", False) or (not getattr(self, "_preview_dirty", False) and current == snapshot):
            return
        try:
            start = getattr(self, "_preview_segment", 0) * UI_PREVIEW_MAX_ROWS
            label_col = self.label_col_var.get().strip()
            editable_names = {
                self.text_col_var.get().strip(),
                "processed_text",
                "augmented_text",
                "clean_text",
                "clean_replacements",
            }
            columns = [
                c for c in (getattr(self, "_preview_columns", None) or [])
                if c in self.df.columns and c != label_col and c in editable_names
            ]
            rows = [line.split("\t") for line in current.splitlines() if line.strip()]
            segment_len = min(UI_PREVIEW_MAX_ROWS, len(self.df) - start)
            count = min(len(rows), segment_len)
            for offset in range(segment_len):
                row_index = self.df.index[start + offset]
                values = rows[offset] if offset < count else []
                for col_index, column in enumerate(columns):
                    value = values[col_index] if col_index < len(values) else ""
                    self.df.at[row_index, column] = value
            self._log(f"Synchronized {count:,} edited preview rows to the in-memory dataset.")
            self._preview_text_snapshot = current
            self._preview_dirty = False
        except Exception as exc:
            self._log(f"Preview synchronization skipped: {exc}")

    def _preview_scroll_command(self, *args: str) -> None:
        self.tox_input.yview(*args)

    def _preview_mousewheel(self, event: tk.Event) -> None:
        if getattr(self, "_preview_df", None) is None:
            return None
        self.tox_input.yview_scroll(-max(1, int(event.delta / 120)), "units")
        _first, last = self.tox_input.yview()
        return "break"

    def _report_dataset_progress(self, percent: float, done: int, total: int, ok: int, fail: int, minutes: float, rate: float) -> None:
        self._dataset_progress_var.set(
            f"Analysis: {percent:.1f}% — {done:,}/{total:,} rows — {rate:.2f} rows/sec"
        )
        self._log(
            f"Dataset analysis progress: {percent:.2f}% complete "
            f"({done:,}/{total:,} rows checked, {ok:,} analyzed, {fail:,} errors, "
            f"{minutes:.1f} min, {rate:.2f} rows/sec)"
        )

    def _append_preview_page(self) -> None:
        df = getattr(self, "_preview_df", None)
        start = getattr(self, "_preview_next_row", 0)
        cols = getattr(self, "_preview_columns", None)
        if df is None or cols is None or start >= len(df):
            return
        end = min(len(df), start + UI_PREVIEW_MAX_ROWS)
        page = df.iloc[start:end].reindex(columns=cols).fillna("").astype(str)
        page = page.apply(lambda col: col.str.replace("\n", "\\n", regex=False).str.replace("\t", " ", regex=False))
        self.tox_input.insert(END, page.to_csv(sep="\t", index=False, header=False))
        self._preview_next_row = end

    def _processing_chunk_rows(self, text_sample: pd.Series) -> int:
        """Adaptive batch size: smaller chunks when texts are long (limits peak RAM)."""
        n = len(text_sample)
        if n <= 0:
            return 4096
        sample_n = min(512, n)
        lens = text_sample.iloc[:sample_n].astype(str).str.len()
        mean_len = float(lens.mean()) if len(lens) else 0.0
        if mean_len > 3000:
            return 256
        if mean_len > 1200:
            return 512
        if mean_len > 350:
            return 2048
        return min(8192, max(1024, n))

    def _detect_text_column(self, df: pd.DataFrame) -> str:
        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not object_cols:
            return str(df.columns[0])
        best_col = object_cols[0]
        best_len = -1.0
        for col in object_cols:
            mean_len = (
                df[col].astype(str).str.len().replace([float("inf")], 0).fillna(0).mean()
            )
            if mean_len > best_len:
                best_len = mean_len
                best_col = col
        return str(best_col)

    def _detect_label_column(self, df: pd.DataFrame, text_col: str) -> str:
        candidates = []
        for col in df.columns:
            if str(col) == text_col:
                continue
            nunique = df[col].nunique(dropna=True)
            if 2 <= nunique <= min(50, max(2, int(len(df) * 0.3))):
                candidates.append((col, nunique))
        if candidates:
            candidates.sort(key=lambda item: item[1])
            return str(candidates[0][0])
        return str(df.columns[1]) if len(df.columns) > 1 else str(df.columns[0])

    def _get_text_series(self) -> pd.Series:
        if self.df is None:
            raise ValueError("Dataset not loaded.")
        col = self.text_col_var.get().strip()
        if col not in self.df.columns:
            raise ValueError("Please select a valid text column.")
        return self.df[col].astype(str)

    def _get_label_series(self) -> pd.Series:
        if self.df is None:
            raise ValueError("Dataset not loaded.")
        col = self.label_col_var.get().strip()
        if col not in self.df.columns:
            raise ValueError("Please select a valid label column.")
        return self.df[col].astype(str)

    def _validate_training_pair(self, texts: list[str], labels: list[str]) -> tuple[bool, str]:
        if len(texts) != len(labels):
            return False, "Text and label lists have different lengths."
        text_col = self.text_col_var.get().strip()
        label_col = self.label_col_var.get().strip()
        if text_col == label_col:
            return (
                False,
                "Text column and label column must be different. "
                "Choose the categorical label column (for example toxic: 0/1), not the comment body.",
            )
        n = len(labels)
        if n == 0:
            return False, "No rows to train on."
        uniq = len({str(y) for y in labels})
        if n >= 20 and uniq / n >= 0.85:
            return (
                False,
                f"The label column has {uniq:,} unique values across {n:,} rows. "
                "That usually means the comment text was selected as labels. "
                "Pick the label column (for example toxic), not the text column.",
            )
        sample = [str(y) for y in labels[: min(200, n)]]
        mean_len = sum(len(s) for s in sample) / max(len(sample), 1)
        if uniq > 50 and mean_len > 80:
            return (
                False,
                "Label values look like long text, not class names or numbers. "
                "Check that the label column is categorical (for example 0/1 or toxic/nontoxic).",
            )
        return True, ""

    def _suggest_preprocessing(self) -> None:
        if self._dataset_busy:
            return
        self._set_dataset_busy(True)
        try:
            series = self._get_text_series()
            texts = series.tolist()
            self._log(f"Suggest preprocessing: analyzed all {len(texts):,} rows.")
        except Exception as exc:
            self._set_dataset_busy(False)
            messagebox.showerror("Suggestion Error", str(exc))
            return

        def worker() -> None:
            try:
                cfg, stats = self.preprocessor.suggest(texts)
                self.root.after(0, lambda: self._finish_preprocessing_suggestion(cfg, stats))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Suggestion Error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _finish_preprocessing_suggestion(self, cfg: PreprocessConfig, stats: dict[str, object]) -> None:
        self._set_dataset_busy(False)

        self.prep_vars["normalize_unicode"].set(cfg.normalize_unicode)
        self.prep_vars["remove_noise"].set(cfg.remove_noise)
        self.prep_vars["reduce_elongations"].set(cfg.reduce_elongations)
        self.prep_vars["map_slang"].set(cfg.map_slang)
        self.prep_vars["expand_contractions"].set(cfg.expand_contractions)
        self.prep_vars["lowercase"].set(cfg.lowercase)
        self.prep_vars["remove_punctuation"].set(cfg.remove_punctuation)
        self.prep_vars["remove_stopwords"].set(cfg.remove_stopwords)
        self.prep_vars["remove_numbers"].set(cfg.remove_numbers)
        self.prep_vars["normalize_whitespace"].set(cfg.normalize_whitespace)
        self._update_preprocessing_apply_state()

        self._log("Suggested preprocessing updated from dataset statistics:")
        for key, value in stats.items():
            self._log(f"- {key}: {value}")

    def _build_prep_config(self) -> PreprocessConfig:
        return PreprocessConfig(
            normalize_unicode=self.prep_vars["normalize_unicode"].get(),
            remove_noise=self.prep_vars["remove_noise"].get(),
            reduce_elongations=self.prep_vars["reduce_elongations"].get(),
            map_slang=self.prep_vars["map_slang"].get(),
            expand_contractions=self.prep_vars["expand_contractions"].get(),
            lowercase=self.prep_vars["lowercase"].get(),
            remove_punctuation=self.prep_vars["remove_punctuation"].get(),
            remove_stopwords=self.prep_vars["remove_stopwords"].get(),
            remove_numbers=self.prep_vars["remove_numbers"].get(),
            normalize_whitespace=self.prep_vars["normalize_whitespace"].get(),
        )

    def _apply_preprocessing(self) -> None:
        if self._dataset_busy:
            return
        self._sync_preview_to_df()
        try:
            text_series = self._get_text_series()
        except Exception as exc:
            messagebox.showerror("Preprocessing Error", str(exc))
            return
        if self.df is None:
            return

        cfg = self._build_prep_config()
        n = len(self.df)
        self._set_dataset_busy(True)
        self._prep_apply_btn.set_enabled(False)
        def worker() -> None:
            out = [self.preprocessor.apply(str(t), cfg) for t in text_series]
            self.root.after(0, lambda: self._finish_preprocessing(out))
        threading.Thread(target=worker, daemon=True).start()

    def _finish_preprocessing(self, out: list[str]) -> None:
        self._set_dataset_busy(False)
        if self.df is None:
            return
        self.df["processed_text"] = out
        gc.collect()
        self._show_preview(self.df)
        self._log("Applied preprocessing. Output column: processed_text")
        self._update_preprocessing_apply_state()

    def _update_preprocessing_apply_state(self) -> None:
        btn = getattr(self, "_prep_apply_btn", None)
        if btn is None:
            return
        enabled = any(v.get() for v in self.prep_vars.values())
        btn.set_enabled(enabled)

    def _build_aug_config(self) -> AugmentConfig:
        strength = float(self.aug_strength_var.get())
        strength = min(0.50, max(0.01, strength))
        return AugmentConfig(
            synonym_replacement=self.aug_vars["synonym_replacement"].get(),
            random_insertion=self.aug_vars["random_insertion"].get(),
            random_swap=self.aug_vars["random_swap"].get(),
            random_deletion=self.aug_vars["random_deletion"].get(),
            back_translation=self.aug_vars["back_translation"].get(),
            paraphrasing=self.aug_vars["paraphrasing"].get(),
            sentence_shuffling=self.aug_vars["sentence_shuffling"].get(),
            sentence_cropping=self.aug_vars["sentence_cropping"].get(),
            noise_injection=self.aug_vars["noise_injection"].get(),
            strength=strength,
        )

    def _apply_augmentation(self) -> None:
        if self._dataset_busy:
            return
        self._sync_preview_to_df()
        if self.df is None:
            messagebox.showerror("Augmentation Error", "Dataset not loaded.")
            return
        source_col = "processed_text" if "processed_text" in self.df.columns else self.text_col_var.get()
        cfg = self._build_aug_config()
        self._set_dataset_busy(True)
        src = self.df[source_col].astype(str)
        if hasattr(self, "_aug_apply_btn"):
            self._aug_apply_btn.set_enabled(False)
        def worker() -> None:
            out = [self.augmenter.apply(str(t), cfg) for t in src]
            self.root.after(0, lambda: finish(out))
        def finish(out: list[str]) -> None:
            self._set_dataset_busy(False)
            if self.df is None:
                return
            self.df["augmented_text"] = out
            gc.collect()
            self._show_preview(self.df)
            self._log(f"Applied augmentation on {source_col}. Output column: augmented_text")
            if hasattr(self, "_aug_apply_btn"):
                self._aug_apply_btn.set_enabled(True)
        threading.Thread(target=worker, daemon=True).start()

    def _training_text_column(self) -> str:
        if self.df is None:
            raise ValueError("Dataset not loaded.")
        if "augmented_text" in self.df.columns:
            return "augmented_text"
        if "processed_text" in self.df.columns:
            return "processed_text"
        return self.text_col_var.get()

    def _scan_profanity(self) -> None:
        self._sync_preview_to_df()
        if self.df is None:
            messagebox.showerror("Scan Error", "Dataset not loaded.")
            return
        try:
            text_col = self._training_text_column()
            if text_col not in self.df.columns:
                # Fallback to the selected text column if the derived name is missing.
                text_col = self.text_col_var.get().strip()
            if text_col not in self.df.columns:
                raise ValueError("Please select a valid text column.")

            text_values = self.df[text_col].astype(str).tolist()
        except Exception as exc:
            messagebox.showerror("Scan Error", str(exc))
            return
        def worker() -> None:
            result = self._scan_profanity_values(pd.DataFrame({text_col: text_values}), text_col)
            self.root.after(0, lambda: self._finish_profanity_scan(result))
        threading.Thread(target=worker, daemon=True).start()

    def _clean_profanity(self) -> None:
        if self.df is None:
            messagebox.showerror("Clean Error", "Dataset not loaded.")
            return
        try:
            source_col = self._training_text_column()
            if source_col not in self.df.columns:
                source_col = self.text_col_var.get().strip()
            if source_col not in self.df.columns:
                raise ValueError("Please select a valid text column.")

            mode = self.clean_mode_var.get().strip().lower() or "mask"
            mask_token = self.clean_mask_var.get().strip() or "[censored]"

            # Small default replacement map. Expand for your project/domain as needed.
            replacement_map = {
                "shit": "bad",
                "fuck": "bad",
                "bitch": "rude person",
                "asshole": "rude person",
                "dick": "rude person",
                "whore": "person",
                "slut": "person",
            }

            text_series = self.df[source_col].astype(str)
            n_rows = len(text_series)
            chunk = self._processing_chunk_rows(text_series)
            clean_texts: list[str] = []
            clean_repls: list[str] = []
            changed = 0
            for start in range(0, n_rows, chunk):
                end = min(start + chunk, n_rows)
                for idx in range(start, end):
                    r = self.profanity_sanitizer.sanitize(
                        text_series.iloc[idx],
                        mode=mode,
                        mask_token=mask_token,
                        replacement_map=replacement_map,
                    )
                    clean_texts.append(r.clean_text)
                    clean_repls.append(", ".join(r.replacements))
                    if r.replacements:
                        changed += 1
        except Exception as exc:
            messagebox.showerror("Clean Error", str(exc))
            return

        self.df["clean_text"] = clean_texts
        self.df["clean_replacements"] = clean_repls
        gc.collect()

        total = n_rows

        self._show_preview(self.df)
        self._log(f"Cleaning completed on column: {source_col}")
        self._log(f"Output column: clean_text (mode={mode})")
        self._log(f"Rows changed: {changed:,} / {total:,}")

    def _investigate_architectures(self) -> None:
        self._sync_preview_to_df()
        if not self._user.is_admin:
            return
        if self.df is None:
            messagebox.showerror("Model Error", "Dataset not loaded.")
            return
        if not self.label_col_var.get().strip():
            messagebox.showerror(
                "Investigation Error",
                "Choose a label column. Load a training CSV with labels (see Model Investigation).",
            )
            return
        try:
            text_col = self._training_text_column()
            texts = self.df[text_col].astype(str).tolist()
            labels = self._get_label_series().tolist()
            ok, err = self._validate_training_pair(texts, labels)
            if not ok:
                messagebox.showerror("Investigation Error", err)
                return
            result = self.model_manager.investigate(texts, labels)
        except Exception as exc:
            messagebox.showerror("Investigation Error", str(exc))
            return

        self.model_var.set(result.best_model_name)
        self._log("Architecture investigation completed:")
        self._log(result.table.to_string(index=False))
        self._log(
            f"Recommended development architecture: {result.best_model_name} "
            f"(F1_macro={result.best_f1_macro:.4f})"
        )

    def _train_selected(self) -> None:
        self._sync_preview_to_df()
        if not self._user.is_admin:
            return
        if self.df is None:
            messagebox.showerror("Train Error", "Dataset not loaded.")
            return
        model_name = self.model_var.get().strip()
        if not model_name:
            messagebox.showerror("Train Error", "Please choose an architecture.")
            return
        if not self.label_col_var.get().strip():
            messagebox.showerror(
                "Train Error",
                "Choose a label column. Training requires a CSV (or Excel/JSON) with a label column "
                "(e.g. toxic). Plain .txt uploads have comments only.",
            )
            return
        try:
            text_col = self._training_text_column()
            texts = self.df[text_col].astype(str).tolist()
            labels = self._get_label_series().tolist()
            ok, err = self._validate_training_pair(texts, labels)
            if not ok:
                messagebox.showerror("Train Error", err)
                return
            metrics, report = self.model_manager.train_selected(texts, labels, model_name)
        except Exception as exc:
            messagebox.showerror("Train Error", str(exc))
            return

        self._log(f"Trained model: {model_name} using column: {text_col}")
        self._log(f"Metrics: {metrics}")
        self._log("Classification report:")
        self._log(report)

    def _logout(self) -> None:
        self._on_logout()

    def _record_single_run(
        self,
        *,
        text_raw: str,
        overall: int,
        metrics: dict[str, int],
        analysis_summary: str,
    ) -> None:
        toxic = 1 if (max((int(v) for v in metrics.values()), default=0) >= 20) else 0
        snippet = text_raw.strip().replace("\n", " ")[:500]
        scores_line = _format_history_scores_line(overall=overall, metrics=metrics)
        summary = f"{scores_line}\n{analysis_summary.strip()}\nText preview: {snippet}"
        self._store.insert_run(
            user_id=self._user.id,
            run_type="single",
            source_filename=self._dataset_path.name if self._dataset_path else None,
            row_count=1,
            toxic_count=toxic,
            non_toxic_count=1 - toxic,
            summary=summary,
        )

    def _open_history_dialog(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Analysis history")
        win.geometry("1040x460")
        win.transient(self.root)

        cols = ("created", "type", "rows", "toxic", "nontoxic", "summary")
        tree = ttk.Treeview(win, columns=cols, show="headings", height=14)
        tree.heading("created", text="When (UTC)")
        tree.heading("type", text="Type")
        tree.heading("rows", text="Rows")
        tree.heading("toxic", text="Toxic")
        tree.heading("nontoxic", text="Non-toxic")
        tree.heading("summary", text="Summary")
        tree.column("created", width=150)
        tree.column("type", width=70)
        tree.column("rows", width=55)
        tree.column("toxic", width=55)
        tree.column("nontoxic", width=70)
        tree.column("summary", width=620)
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        def refresh() -> None:
            for item in tree.get_children():
                tree.delete(item)
            for r in self._store.list_runs_for_user(self._user.id):
                tree.insert(
                    "",
                    "end",
                    iid=str(r["id"]),
                    values=(
                        r.get("created_at") or "",
                        r.get("run_type") or "",
                        r.get("row_count"),
                        r.get("toxic_count"),
                        r.get("non_toxic_count"),
                        _history_summary_column(r.get("summary") or ""),
                    ),
                )

        refresh()

        def show_analysis_preview(run_id: int) -> None:
            row = self._store.get_run(run_id, self._user.id)
            if not row:
                return
            preview = _history_analysis_preview(row.get("summary") or "")
            rp = row.get("results_path")
            if rp and Path(rp).is_file():
                preview = f"{preview}\n\nSaved results file:\n{rp}"

            detail_win = tk.Toplevel(win)
            detail_win.title("Analysis result preview")
            detail_win.geometry("760x520")
            detail_win.transient(win)

            text = tk.Text(detail_win, wrap="word", font=("Segoe UI", 10), padx=12, pady=12)
            text.pack(fill="both", expand=True)
            text.insert("1.0", preview)
            text.configure(state="disabled")

            ttk.Button(detail_win, text="Close", command=detail_win.destroy).pack(pady=(0, 10))

        def on_summary_click(event: tk.Event) -> None:
            if tree.identify_region(event.x, event.y) != "cell":
                return
            if tree.identify_column(event.x) != "#6":
                return
            row_id = tree.identify_row(event.y)
            if not row_id:
                return
            show_analysis_preview(int(row_id))

        tree.bind("<ButtonRelease-1>", on_summary_click)

        btn_row = ttk.Frame(win)
        btn_row.pack(fill="x", padx=10, pady=(0, 10))

        def clear_history() -> None:
            if not messagebox.askyesno(
                "Clear history",
                "Delete all analysis history for your account? This cannot be undone.",
            ):
                return
            removed = self._store.clear_runs_for_user(self._user.id)
            refresh()
            messagebox.showinfo("Clear history", f"Removed {removed} entr{'y' if removed == 1 else 'ies'}.")

        ttk.Button(btn_row, text="Clear history", command=clear_history).pack(side="left")
        ttk.Button(btn_row, text="Close", command=win.destroy).pack(side="right")

    def _open_admin_users_dialog(self) -> None:
        if not self._user.is_admin:
            return
        win = tk.Toplevel(self.root)
        win.title("Manage users")
        win.geometry("720x380")
        win.transient(self.root)

        cols = ("id", "username", "admin", "blocked", "created")
        tree = ttk.Treeview(win, columns=cols, show="headings", height=12)
        for c, t, w in (
            ("id", "ID", 50),
            ("username", "Username", 180),
            ("admin", "Admin", 60),
            ("blocked", "Blocked", 70),
            ("created", "Created (UTC)", 200),
        ):
            tree.heading(c, text=t)
            tree.column(c, width=w)
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        def refresh() -> None:
            for item in tree.get_children():
                tree.delete(item)
            for u in self._store.list_users():
                tree.insert(
                    "",
                    "end",
                    iid=str(u["id"]),
                    values=(
                        u["id"],
                        u["username"],
                        "yes" if u["is_admin"] else "no",
                        "yes" if u["is_blocked"] else "no",
                        u.get("created_at") or "",
                    ),
                )

        refresh()

        btn_row = ttk.Frame(win)
        btn_row.pack(fill="x", padx=10, pady=(0, 10))

        def selected_id() -> int | None:
            sel = tree.selection()
            return int(sel[0]) if sel else None

        def do_block() -> None:
            uid = selected_id()
            if uid is None:
                return
            self._store.set_blocked(uid, True)
            refresh()

        def do_unblock() -> None:
            uid = selected_id()
            if uid is None:
                return
            self._store.set_blocked(uid, False)
            refresh()

        def do_delete() -> None:
            uid = selected_id()
            if uid is None:
                return
            if uid == self._user.id:
                messagebox.showerror("Manage users", "You cannot delete your own account while signed in.")
                return
            if not messagebox.askyesno("Manage users", "Delete this user and all of their analysis history?"):
                return
            self._store.delete_user(uid)
            refresh()

        ttk.Button(btn_row, text="Block selected", command=do_block).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Unblock selected", command=do_unblock).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Delete selected", command=do_delete).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Close", command=win.destroy).pack(side="right", padx=4)

    def _open_admin_stats_dialog(self) -> None:
        if not self._user.is_admin:
            return
        win = tk.Toplevel(self.root)
        win.title("System statistics")
        win.geometry("900x560")
        win.transient(self.root)

        stats = self._store.admin_aggregate_stats()
        total = int(stats["comments_analyzed"])
        toxic = int(stats["toxic_comments"])
        nontoxic = int(stats["non_toxic_comments"])

        summary = ttk.LabelFrame(win, text="Totals (all users)", padding=10)
        summary.pack(fill="x", padx=10, pady=10)
        ttk.Label(
            summary,
            text=f"Comments analyzed (recorded runs): {total:,}\n"
            f"Toxic (score ≥ 35): {toxic:,}\n"
            f"Non-toxic: {nontoxic:,}",
        ).pack(anchor="w")

        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            ttk.Label(
                win,
                text="Install matplotlib to see charts (pip install matplotlib).",
            ).pack(pady=10)
            ttk.Button(win, text="Close", command=win.destroy).pack(pady=10)
            return

        fig = Figure(figsize=(9, 4.2), dpi=100)
        ax1 = fig.add_subplot(121)
        timeline = stats.get("timeline_days") or []
        if timeline:
            days, counts = zip(*timeline)
            ax1.bar(days, counts, color="#2563eb")
            ax1.set_title("Comments analyzed per day")
            ax1.tick_params(axis="x", rotation=45)
            ax1.set_ylabel("Rows")
        else:
            ax1.text(0.5, 0.5, "No history yet", ha="center", va="center")
            ax1.axis("off")

        ax2 = fig.add_subplot(122)
        if toxic + nontoxic > 0:
            ax2.pie(
                [toxic, nontoxic],
                labels=["Toxic (≥35)", "Non-toxic"],
                autopct="%1.1f%%",
                colors=["#dc2626", "#16a34a"],
                startangle=90,
            )
            ax2.set_title("Toxic vs non-toxic (recorded)")
        else:
            ax2.text(0.5, 0.5, "No classified rows yet", ha="center", va="center")
            ax2.axis("off")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 10))

    def _log(self, text: str) -> None:
        self.result_text.insert(END, f"{text}\n")
        self.result_text.see(END)

    def _log_toxicity_summary(self, text: str, overall: int) -> None:
        """Append analysis summary to Logs / Results with color from overall score."""
        overall = max(0, min(100, int(overall)))
        if overall < 35:
            tag = "tox_summary_low"
        elif overall < 70:
            tag = "tox_summary_med"
        else:
            tag = "tox_summary_high"
        self.result_text.insert(END, f"{text}\n", (tag,))
        self.result_text.see(END)

    def _on_left_frame_configure(self, _event: tk.Event) -> None:
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _on_left_canvas_configure(self, event: tk.Event) -> None:
        self.left_canvas.itemconfigure(self.left_canvas_window, width=event.width)

    def _bind_mousewheel(self, _event: tk.Event) -> None:
        self.left_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event: tk.Event) -> None:
        self.left_canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_preview_mousewheel(self, event: tk.Event) -> str:
        # Preview panel removed
        return "break"
        return "break"

    def _preview_text_focus_in(self, _event: tk.Event) -> None:
        # Preview panel removed
        return

    def _next_preview_segment(self) -> None:
        df = getattr(self, "_preview_df", None)
        if df is None:
            return
        self._sync_preview_to_df()
        total = max(1, (len(df) + UI_PREVIEW_MAX_ROWS - 1) // UI_PREVIEW_MAX_ROWS)
        if self._preview_segment + 1 < total:
            self._preview_segment += 1
            self._show_preview(df)

    def _previous_preview_segment(self) -> None:
        df = getattr(self, "_preview_df", None)
        if df is None:
            return
        self._sync_preview_to_df()
        if self._preview_segment > 0:
            self._preview_segment -= 1
            self._show_preview(df)
