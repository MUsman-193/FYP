"""Tkinter desktop UI for toxic comment preprocessing, augmentation, and modeling."""

from __future__ import annotations

from pathlib import Path
import threading
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
    ToxicityTextMetricsBundle,
    preprocess_and_analyze_single,
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
                    self._log(f"Local toxicity model failed to load: {err}")
                    return
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
        self._tox_card_bg = "#ffffff"

        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        nav = tk.Frame(container, bg=self._tox_bg)
        nav.pack(fill="x", pady=(0, 6))
        tk.Label(
            nav,
            text=f"Signed in as {self._user.username}"
            + (" (administrator)" if self._user.is_admin else ""),
            font=("Segoe UI", 10),
            bg=self._tox_bg,
            fg=self._tox_muted,
        ).pack(side="left")
        nav_btns = tk.Frame(nav, bg=self._tox_bg)
        nav_btns.pack(side="right")
        ttk.Button(nav_btns, text="Analysis history", command=self._open_history_dialog).pack(
            side="left", padx=(0, 6)
        )
        if self._user.is_admin:
            ttk.Button(nav_btns, text="Manage users", command=self._open_admin_users_dialog).pack(
                side="left", padx=(0, 6)
            )
            ttk.Button(nav_btns, text="System statistics", command=self._open_admin_stats_dialog).pack(
                side="left", padx=(0, 6)
            )
        ttk.Button(nav_btns, text="Log out", command=self._logout).pack(side="left")

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
        if not self._modeling_available:
            self.model_combo.configure(state="disabled")
            self.train_button.set_enabled(False)

            # Disable the investigate button too.
            for child in model_frame.winfo_children():
                if isinstance(child, _RoundedButton) and getattr(child, "_text", "") == "Investigate Architectures":
                    child.set_enabled(False)
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

    def _set_toxicity_analyze_loading(self, loading: bool) -> None:
        btn = getattr(self, "analyze_btn", None)
        if btn is None:
            return
        up = getattr(self, "upload_btn", None)
        if loading:
            self._analyze_btn_idle_label = getattr(self, "_analyze_btn_idle_label", "Analyze")
            btn.set_text("Analyzing...")
            btn.set_enabled(False)
            if up is not None:
                up.set_enabled(False)
            self.root.update_idletasks()
        else:
            btn.set_text(getattr(self, "_analyze_btn_idle_label", "Analyze"))
            btn.set_enabled(True)
            if up is not None:
                up.set_enabled(True)
            self.root.update_idletasks()

    def _analyze_toxicity_text(self) -> None:
        if self._toxicity_analysis_busy:
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
        self._set_toxicity_analyze_loading(True)

        def worker() -> None:
            bundle: ToxicityTextMetricsBundle | None = None
            err: BaseException | None = None
            try:
                bundle = preprocess_and_analyze_single(
                    text_raw,
                    preprocessor=self.preprocessor,
                    prep_config=prep,
                    toxicity_analyzer=self.toxicity_analyzer,
                )
            except BaseException as exc:
                err = exc
            br, er = bundle, err
            self.root.after(
                0,
                lambda tr=text_raw, bb=br, ee=er: self._complete_single_analyze_ui(tr, bb, ee),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _complete_single_analyze_ui(
        self,
        text_raw: str,
        bundle: ToxicityTextMetricsBundle | None,
        err: BaseException | None,
    ) -> None:
        """Apply analysis on the Tk main thread after background work completes."""
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
            summary_line=(
                "Single: "
                f"top={max(metrics.items(), key=lambda kv: int(kv[1]))[0]} "
                f"({max((int(v) for v in metrics.values()), default=0)}%) "
                f"level={lvl}"
            ),
        )

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
        # Header column pickers were removed from the UI, so these comboboxes may not exist.
        # Keep dataset loading functional by only updating them when present.
        if hasattr(self, "text_col_combo"):
            self.text_col_combo["values"] = cols  # type: ignore[attr-defined]
        if hasattr(self, "label_col_combo"):
            self.label_col_combo["values"] = cols  # type: ignore[attr-defined]

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

    def _logout(self) -> None:
        self._on_logout()

    def _record_single_run(
        self,
        *,
        text_raw: str,
        overall: int,
        metrics: dict[str, int],
        summary_line: str,
    ) -> None:
        toxic = 1 if (max((int(v) for v in metrics.values()), default=0) >= 20) else 0
        snippet = text_raw.strip().replace("\n", " ")[:500]
        summary = f"{summary_line}\nText preview: {snippet}"
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
        win.geometry("900x420")
        win.transient(self.root)

        cols = ("created", "type", "rows", "toxic", "nontoxic", "summary")
        tree = ttk.Treeview(win, columns=cols, show="headings", height=14)
        tree.heading("created", text="When (UTC)")
        tree.heading("type", text="Type")
        tree.heading("rows", text="Rows")
        tree.heading("toxic", text="Toxic")
        tree.heading("nontoxic", text="Non-toxic")
        tree.heading("summary", text="Summary")
        tree.column("created", width=160)
        tree.column("type", width=70)
        tree.column("rows", width=60)
        tree.column("toxic", width=60)
        tree.column("nontoxic", width=70)
        tree.column("summary", width=420)
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
                        (r.get("summary") or "").replace("\n", " ")[:200],
                    ),
                )

        refresh()

        def on_details(_event: tk.Event | None = None) -> None:
            sel = tree.selection()
            if not sel:
                return
            rid = int(sel[0])
            row = self._store.get_run(rid, self._user.id)
            if not row:
                return
            rp = row.get("results_path")
            detail = (row.get("summary") or "").strip()
            if rp and Path(rp).is_file():
                detail += f"\n\nSaved results file:\n{rp}"
            messagebox.showinfo("Run details", detail or "(No summary)")

        tree.bind("<Double-1>", on_details)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 10))

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
