"""Tkinter desktop UI for toxic comment preprocessing, augmentation, and modeling."""

from __future__ import annotations

from pathlib import Path
from tkinter import END, Tk, filedialog, messagebox
from tkinter import ttk
import tkinter as tk
import tkinter.font as tkfont
import colorsys

import pandas as pd

from augmentation import AugmentConfig, TextAugmenter
from moderation import (
    LocalToxicityAnalyzer,
    ProfanitySanitizer,
    ProfanityScanner,
    ToxicityModelError,
)
from preprocessing import PreprocessConfig, TextPreprocessor

# Max rows rendered in Dataset Preview (full file still loaded; raise if UI tolerates it).
PREVIEW_MAX_ROWS = 50_000


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
    ) -> None:
        super().__init__(parent, bg=parent.cget("bg"), highlightthickness=0, bd=0, cursor="hand2")
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


class ToxicCommentApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Toxic Comment Classification Workbench")
        self.root.geometry("1280x800")

        self.df: pd.DataFrame | None = None
        self._dataset_path: Path | None = None

        self.preprocessor = TextPreprocessor()
        self.augmenter = TextAugmenter(seed=42)
        self.profanity_scanner = ProfanityScanner()
        self.profanity_sanitizer = ProfanitySanitizer(self.profanity_scanner.terms)
        # FYP model weights in project modal/ (see LocalToxicityAnalyzer). Falls back if load/inference fails.
        self.toxicity_analyzer = LocalToxicityAnalyzer()
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
        if self._modeling_available and self.model_manager is not None:
            self._log(self.model_manager.architecture_guidance())
        else:
            msg = "Modeling features unavailable (SciPy/scikit-learn not installed or incompatible)."
            if self._modeling_error:
                msg += f" Error: {self._modeling_error}"
            self._log(msg)

    def _build_ui(self) -> None:
        # Shared design tokens (keep sidebar + right panel consistent)
        self._tox_blue = _hsl_to_hex(221, 83, 53)
        self._tox_green = _hsl_to_hex(142, 71, 45)
        self._tox_green_bg = _hsl_to_hex(142, 71, 95)
        self._tox_border = "#e5e7eb"
        self._tox_text = "#111827"
        self._tox_muted = "#6b7280"
        self._tox_bg = "#f9fafb"
        self._tox_card_bg = "#ffffff"

        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        top = ttk.LabelFrame(container, text="Dataset", padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Text Column").grid(row=0, column=0, sticky="w")
        self.text_col_combo = ttk.Combobox(
            top, textvariable=self.text_col_var, state="readonly", width=35
        )
        self.text_col_combo.grid(row=0, column=1, sticky="w")

        ttk.Label(top, text="Label Column").grid(
            row=0, column=2, sticky="e", padx=(15, 6)
        )
        self.label_col_combo = ttk.Combobox(
            top, textvariable=self.label_col_var, state="readonly", width=30
        )
        self.label_col_combo.grid(row=0, column=3, sticky="w")

        middle = ttk.Frame(container)
        middle.pack(fill="both", expand=True, pady=10)

        left_wrapper = ttk.Frame(middle)
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

        right = ttk.Frame(middle)
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

        model_frame = _sidebar_card(left, "Model Investigation")
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
        save_btn = _RoundedButton(
            model_frame,
            text="Save Processed Dataset",
            bg="#ffffff",
            fg=self._tox_text,
            command=self._save_dataset,
            border=self._tox_border,
            radius=12,
            hover_bg="#f3f4f6",
            active_bg="#e5e7eb",
        )
        save_btn.configure(height=34)
        save_btn.pack(fill="x", pady=(8, 0))
        if not self._modeling_available:
            self.model_combo.configure(state="disabled")
            self.train_button.configure(state="disabled")

            # Disable the investigate button too.
            for child in model_frame.winfo_children():
                if isinstance(child, _RoundedButton) and getattr(child, "_text", "") == "Investigate Architectures":
                    child.configure(state="disabled")
                    break

        # Right side: unified interface (Preview + Logs + Toxicity Analysis)
        merged = ttk.Frame(right, padding=0)
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
        self._tox_bg = "#f9fafb"
        self._tox_card_bg = "#ffffff"

        canvas = tk.Canvas(parent, bg=self._tox_bg, highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
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

        # Title
        tk.Label(
            body,
            text="Toxicity Analysis",
            font=("Segoe UI", 20, "bold"),
            bg=self._tox_bg,
            fg=self._tox_text,
        ).pack(pady=(18, 4))
        tk.Label(
            body,
            text="Enter a comment or upload a file to analyze for toxic content",
            font=("Segoe UI", 10),
            bg=self._tox_bg,
            fg=self._tox_muted,
        ).pack(pady=(0, 14))

        # Toxicity analysis section (analysis shown above Logs/Results)
        tk.Label(
            body,
            text="Toxicity Analysis",
            font=("Segoe UI", 14, "bold"),
            bg=self._tox_bg,
            fg=self._tox_text,
        ).pack(anchor="w", padx=18, pady=(4, 10))

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
        tox_input_v = ttk.Scrollbar(tox_input_wrap, orient="vertical", command=self.tox_input.yview)
        self.tox_input.configure(yscrollcommand=tox_input_v.set)
        self.tox_input.grid(row=0, column=0, sticky="nsew")
        tox_input_v.grid(row=0, column=1, sticky="ns")

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
        self.analyze_btn.grid(row=0, column=0, sticky="we", padx=(0, 10))

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
        self.upload_btn.grid(row=0, column=1, sticky="e")

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
                ("Datasets", "*.csv *.xlsx *.xls *.json"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return
        p = Path(path)
        self._dataset_path = p
        self.file_var.set(p.name)

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

    def _analyze_toxicity_text(self) -> None:
        text_raw = self.tox_input.get("1.0", END).strip()
        if not text_raw:
            messagebox.showerror("Analyze Error", "Please enter text (or upload a file) first.")
            return

        try:
            if not getattr(self, "_tox_results_visible", False):
                # Show results UI only when analysis actually runs.
                self._tox_results_wrap.pack(fill="x", pady=(0, 6))
                self._tox_results_visible = True

            # Same preprocessing pipeline as dataset Apply (quote fences, noise, etc.).
            text = self.preprocessor.apply(text_raw, self._build_prep_config())

            # Primary: local modal/FYP-model.safetensors (FYP toxicity model).
            try:
                analysis = self.toxicity_analyzer.analyze(text)
                metrics, overall = self._metrics_from_llm_analysis(
                    toxicity_level=analysis.toxicity_level,
                    toxicity_types=analysis.toxicity_type,
                )
                self._update_toxicity_ui(overall=overall, metrics=metrics)

                summary = (
                    "Toxicity Analysis Result (local model)\n"
                    f"- Category: {', '.join(analysis.category) if analysis.category else 'None'}\n"
                    f"- Toxicity Level: {analysis.toxicity_level}\n"
                    f"- Toxicity Type: {', '.join(analysis.toxicity_type) if analysis.toxicity_type else 'None'}\n"
                )
                if analysis.explanation:
                    summary += f"- Explanation: {analysis.explanation}\n"
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
            except ToxicityModelError as exc:
                # Fallback: heuristic scoring using profanity scanner (keeps UI usable offline).
                scan = self.profanity_scanner.scan_text(text)
                prof = int(scan.profanity_count)
                toxic = min(100, prof * 15)
                severe = 0 if prof == 0 else min(100, max(0, (prof - 2) * 18))
                identity = 0
                insult = min(100, prof * 6)
                profanity = min(100, prof * 20)
                threat = 0

                metrics = {
                    "Toxicity": toxic,
                    "Severe Toxicity": severe,
                    "Identity Attack": identity,
                    "Insult": insult,
                    "Profanity": profanity,
                    "Threat": threat,
                }
                overall = int(round(sum(metrics.values()) / len(metrics)))
                self._update_toxicity_ui(overall=overall, metrics=metrics)
                hf_summary = (
                    "Toxicity Analysis Result (heuristic fallback)\n"
                    f"- UI Score (derived): {overall}%\n"
                    f"  - Toxicity: {metrics['Toxicity']}%\n"
                    f"  - Severe Toxicity: {metrics['Severe Toxicity']}%\n"
                    f"  - Identity Attack: {metrics['Identity Attack']}%\n"
                    f"  - Insult: {metrics['Insult']}%\n"
                    f"  - Profanity: {metrics['Profanity']}%\n"
                    f"  - Threat: {metrics['Threat']}%"
                )
                self._log_toxicity_summary(hf_summary, overall)
                self._log(f"Local toxicity model unavailable; used heuristic fallback. Reason: {exc}")
        except Exception as exc:
            messagebox.showerror("Analyze Error", str(exc))
            self._log(f"Analyze Error: {exc}")
            return

    def _metrics_from_llm_analysis(
        self,
        *,
        toxicity_level: str,
        toxicity_types: list[str],
    ) -> tuple[dict[str, int], int]:
        # Convert model labels to the existing UI's 6 metrics.
        lvl = (toxicity_level or "").strip().lower()
        base = {"low": 20, "medium": 55, "high": 80, "severe": 95}.get(lvl, 40)
        severe = {"low": 0, "medium": 35, "high": 70, "severe": 95}.get(lvl, 30)

        types = {t.strip().lower(): t for t in (toxicity_types or []) if t and t.strip()}
        identity = 85 if "identity attack" in types else 0
        threat = 85 if "threat" in types else 0
        profanity = 80 if "profanity" in types else 0
        insult = 70 if ("insult" in types or "bullying" in types or "political abuse" in types) else 0

        # If the model says it's toxic but didn't name any type, keep something non-zero.
        if base >= 55 and max(identity, threat, profanity, insult) == 0:
            insult = 55

        metrics = {
            "Toxicity": int(max(0, min(100, base))),
            "Severe Toxicity": int(max(0, min(100, severe))),
            "Identity Attack": int(max(0, min(100, identity))),
            "Insult": int(max(0, min(100, insult))),
            "Profanity": int(max(0, min(100, profanity))),
            "Threat": int(max(0, min(100, threat))),
        }
        overall = int(round(sum(metrics.values()) / len(metrics)))
        return metrics, overall

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

    def _load_dataset_path(self, path: Path) -> None:
        if not path.exists():
            messagebox.showerror("Error", "Please select a valid dataset file.")
            return
        self._dataset_path = path
        self.file_var.set(path.name)
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(path)
            elif path.suffix.lower() == ".json":
                df = pd.read_json(path)
            else:
                messagebox.showerror("Error", "Unsupported file format.")
                return
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            return

        self.df = df.copy()
        cols = list(self.df.columns.astype(str))
        self.text_col_combo["values"] = cols
        self.label_col_combo["values"] = cols

        text_col = self._detect_text_column(self.df)
        label_col = self._detect_label_column(self.df, text_col)
        if text_col:
            self.text_col_var.set(text_col)
        if label_col:
            self.label_col_var.set(label_col)

        self._show_preview(self.df)
        self._log(f"Loaded dataset with shape: {self.df.shape}")
        if text_col:
            self._log(f"Detected text column: {text_col}")
        if label_col:
            self._log(f"Detected label column: {label_col}")
        self._scan_profanity()

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
            self.tox_input.delete("1.0", END)
            self.tox_input.insert("1.0", text)
            self.tox_input.see("1.0")

    def _show_preview(self, df: pd.DataFrame) -> None:
        # Preview panel removed; show dataset preview inside the analysis input.
        if not hasattr(self, "tox_input") or self.tox_input is None:
            return
        self.tox_input.delete("1.0", END)
        # Use TSV-style preview to avoid padded right-aligned DataFrame formatting.
        n_show = min(len(df), PREVIEW_MAX_ROWS)
        preview_df = df.head(n_show).fillna("").astype(str)
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
        preview_str = preview_df.to_csv(sep="\t", index=False)
        header = f"Dataset loaded: {len(df):,} rows, {len(df.columns):,} columns\n\n"
        self.tox_input.insert("1.0", header + preview_str)
        self.tox_input.see("1.0")

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

    def _suggest_preprocessing(self) -> None:
        try:
            texts = self._get_text_series().tolist()
            cfg, stats = self.preprocessor.suggest(texts)
        except Exception as exc:
            messagebox.showerror("Suggestion Error", str(exc))
            return

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
        try:
            text_series = self._get_text_series()
        except Exception as exc:
            messagebox.showerror("Preprocessing Error", str(exc))
            return
        if self.df is None:
            return

        cfg = self._build_prep_config()
        self.df["processed_text"] = text_series.apply(lambda t: self.preprocessor.apply(t, cfg))
        self._show_preview(self.df)
        self._log("Applied preprocessing. Output column: processed_text")

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
        if self.df is None:
            messagebox.showerror("Augmentation Error", "Dataset not loaded.")
            return
        source_col = "processed_text" if "processed_text" in self.df.columns else self.text_col_var.get()
        cfg = self._build_aug_config()
        self.df["augmented_text"] = self.df[source_col].astype(str).apply(
            lambda t: self.augmenter.apply(t, cfg)
        )
        self._show_preview(self.df)
        self._log(f"Applied augmentation on {source_col}. Output column: augmented_text")

    def _training_text_column(self) -> str:
        if self.df is None:
            raise ValueError("Dataset not loaded.")
        if "augmented_text" in self.df.columns:
            return "augmented_text"
        if "processed_text" in self.df.columns:
            return "processed_text"
        return self.text_col_var.get()

    def _scan_profanity(self) -> None:
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

            texts = self.df[text_col].astype(str).tolist()
            results = self.profanity_scanner.scan_texts(texts)
        except Exception as exc:
            messagebox.showerror("Scan Error", str(exc))
            return

        self.df["has_profanity"] = [r.has_profanity for r in results]
        self.df["profanity_count"] = [r.profanity_count for r in results]
        self.df["profanity_matches"] = [", ".join(r.matches) for r in results]

        flagged = int(sum(self.df["has_profanity"].astype(bool)))
        total = int(len(self.df))
        pct = (flagged / total * 100.0) if total else 0.0

        self._show_preview(self.df)
        self._log(f"Profanity scan completed on column: {text_col}")
        self._log(f"Flagged rows: {flagged:,} / {total:,} ({pct:.2f}%)")

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

            texts = self.df[source_col].astype(str).tolist()
            results = self.profanity_sanitizer.sanitize_texts(
                texts,
                mode=mode,
                mask_token=mask_token,
                replacement_map=replacement_map,
            )
        except Exception as exc:
            messagebox.showerror("Clean Error", str(exc))
            return

        self.df["clean_text"] = [r.clean_text for r in results]
        self.df["clean_replacements"] = [", ".join(r.replacements) for r in results]

        changed = int(sum(1 for r in results if r.replacements))
        total = int(len(results))

        self._show_preview(self.df)
        self._log(f"Cleaning completed on column: {source_col}")
        self._log(f"Output column: clean_text (mode={mode})")
        self._log(f"Rows changed: {changed:,} / {total:,}")

    def _investigate_architectures(self) -> None:
        if self.df is None:
            messagebox.showerror("Model Error", "Dataset not loaded.")
            return
        try:
            text_col = self._training_text_column()
            texts = self.df[text_col].astype(str).tolist()
            labels = self._get_label_series().tolist()
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
        if self.df is None:
            messagebox.showerror("Train Error", "Dataset not loaded.")
            return
        model_name = self.model_var.get().strip()
        if not model_name:
            messagebox.showerror("Train Error", "Please choose an architecture.")
            return
        try:
            text_col = self._training_text_column()
            texts = self.df[text_col].astype(str).tolist()
            labels = self._get_label_series().tolist()
            metrics, report = self.model_manager.train_selected(texts, labels, model_name)
        except Exception as exc:
            messagebox.showerror("Train Error", str(exc))
            return

        self._log(f"Trained model: {model_name} using column: {text_col}")
        self._log(f"Metrics: {metrics}")
        self._log("Classification report:")
        self._log(report)

    def _save_dataset(self) -> None:
        if self.df is None:
            messagebox.showerror("Save Error", "Dataset not loaded.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.df.to_csv(path, index=False)
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))
            return
        self._log(f"Saved dataset to: {path}")

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


def run() -> None:
    root = Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    ToxicCommentApp(root)
    root.mainloop()
