"""Tkinter desktop UI for toxic comment preprocessing, augmentation, and modeling."""

from __future__ import annotations

from pathlib import Path
from tkinter import END, Tk, filedialog, messagebox
from tkinter import ttk
import tkinter as tk

import pandas as pd

from augmentation import AugmentConfig, TextAugmenter
from moderation import ProfanitySanitizer, ProfanityScanner
from preprocessing import PreprocessConfig, TextPreprocessor

# Max rows rendered in Dataset Preview (full file still loaded; raise if UI tolerates it).
PREVIEW_MAX_ROWS = 50_000


class ToxicCommentApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Toxic Comment Classification Workbench")
        self.root.geometry("1280x800")

        self.df: pd.DataFrame | None = None

        self.preprocessor = TextPreprocessor()
        self.augmenter = TextAugmenter(seed=42)
        self.profanity_scanner = ProfanityScanner()
        self.profanity_sanitizer = ProfanitySanitizer(self.profanity_scanner.terms)
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
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        top = ttk.LabelFrame(container, text="Dataset", padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="File").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.file_var, width=90).grid(
            row=0, column=1, padx=6, sticky="we"
        )
        ttk.Button(top, text="Browse", command=self._browse_file).grid(row=0, column=2)
        ttk.Button(top, text="Load Dataset", command=self._load_dataset).grid(
            row=0, column=3, padx=6
        )

        ttk.Label(top, text="Text Column").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.text_col_combo = ttk.Combobox(
            top, textvariable=self.text_col_var, state="readonly", width=35
        )
        self.text_col_combo.grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(top, text="Label Column").grid(
            row=1, column=2, sticky="e", pady=(8, 0), padx=(15, 6)
        )
        self.label_col_combo = ttk.Combobox(
            top, textvariable=self.label_col_var, state="readonly", width=30
        )
        self.label_col_combo.grid(row=1, column=3, sticky="w", pady=(8, 0))

        middle = ttk.Frame(container)
        middle.pack(fill="both", expand=True, pady=10)

        left_wrapper = ttk.Frame(middle)
        left_wrapper.pack(side="left", fill="y")
        self.left_canvas = tk.Canvas(left_wrapper, width=380, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(
            left_wrapper, orient="vertical", command=self.left_canvas.yview
        )
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)
        self.left_canvas.pack(side="left", fill="y")
        left_scrollbar.pack(side="left", fill="y")

        left = ttk.Frame(self.left_canvas)
        self.left_canvas_window = self.left_canvas.create_window(
            (0, 0), window=left, anchor="nw"
        )
        left.bind("<Configure>", self._on_left_frame_configure)
        self.left_canvas.bind("<Configure>", self._on_left_canvas_configure)
        self.left_canvas.bind("<Enter>", self._bind_mousewheel)
        self.left_canvas.bind("<Leave>", self._unbind_mousewheel)

        right = ttk.Frame(middle)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        prep_frame = ttk.LabelFrame(left, text="Preprocessing", padding=8)
        prep_frame.pack(fill="x")
        prep_labels = [
            ("Lowercasing", "lowercase"),
            ("Remove Punctuation", "remove_punctuation"),
            ("Remove Stopwords", "remove_stopwords"),
            ("Remove Numbers", "remove_numbers"),
            ("Normalize Extra Whitespace", "normalize_whitespace"),
        ]
        for idx, (label, key) in enumerate(prep_labels):
            ttk.Checkbutton(
                prep_frame,
                text=label,
                variable=self.prep_vars[key],
                onvalue=True,
                offvalue=False,
            ).grid(row=idx, column=0, sticky="w")
        ttk.Button(
            prep_frame, text="Suggest Preprocessing", command=self._suggest_preprocessing
        ).grid(row=len(prep_labels), column=0, sticky="we", pady=(8, 0))
        ttk.Button(
            prep_frame, text="Apply Selected Preprocessing", command=self._apply_preprocessing
        ).grid(row=len(prep_labels) + 1, column=0, sticky="we", pady=(6, 0))

        aug_frame = ttk.LabelFrame(left, text="Data Augmentation", padding=8)
        aug_frame.pack(fill="x", pady=10)
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
        for idx, (label, key) in enumerate(aug_labels):
            ttk.Checkbutton(
                aug_frame,
                text=label,
                variable=self.aug_vars[key],
                onvalue=True,
                offvalue=False,
            ).grid(row=idx, column=0, sticky="w")
        ttk.Label(aug_frame, text="Strength (0.01 - 0.5)").grid(
            row=len(aug_labels), column=0, sticky="w", pady=(6, 0)
        )
        ttk.Spinbox(
            aug_frame,
            from_=0.01,
            to=0.50,
            increment=0.01,
            textvariable=self.aug_strength_var,
            width=8,
        ).grid(row=len(aug_labels) + 1, column=0, sticky="w")
        ttk.Button(
            aug_frame, text="Apply Selected Augmentation", command=self._apply_augmentation
        ).grid(row=len(aug_labels) + 2, column=0, sticky="we", pady=(8, 0))

        model_frame = ttk.LabelFrame(left, text="Model Investigation", padding=8)
        model_frame.pack(fill="x")
        ttk.Button(
            model_frame, text="Investigate Architectures", command=self._investigate_architectures
        ).grid(row=0, column=0, sticky="we")
        ttk.Label(model_frame, text="Selected Architecture").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            state="readonly",
            values=list(self.model_manager.model_factories.keys())
            if (self._modeling_available and self.model_manager is not None)
            else [],
            width=30,
        )
        self.model_combo.grid(row=2, column=0, sticky="we")
        self.train_button = ttk.Button(
            model_frame, text="Train Selected Model", command=self._train_selected
        )
        self.train_button.grid(row=3, column=0, sticky="we", pady=(8, 0))
        ttk.Button(model_frame, text="Save Processed Dataset", command=self._save_dataset).grid(
            row=4, column=0, sticky="we", pady=(8, 0)
        )
        if not self._modeling_available:
            self.model_combo.configure(state="disabled")
            self.train_button.configure(state="disabled")
            # The investigate button is created earlier; disable it too.
            for child in model_frame.winfo_children():
                if isinstance(child, ttk.Button) and child.cget("text") == "Investigate Architectures":
                    child.configure(state="disabled")
                    break

        scan_frame = ttk.LabelFrame(left, text="Profanity Scan", padding=8)
        scan_frame.pack(fill="x", pady=10)
        ttk.Button(
            scan_frame, text="Scan Dataset (Vulgar Words)", command=self._scan_profanity
        ).grid(row=0, column=0, sticky="we")
        ttk.Label(
            scan_frame,
            text="Adds columns: has_profanity, profanity_count, profanity_matches",
            wraplength=340,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        clean_frame = ttk.LabelFrame(left, text="Clean / Replace Profanity", padding=8)
        clean_frame.pack(fill="x")
        ttk.Radiobutton(
            clean_frame, text="Mask", value="mask", variable=self.clean_mode_var
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            clean_frame, text="Replace (mapped)", value="replace", variable=self.clean_mode_var
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))
        ttk.Label(clean_frame, text="Mask token").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(clean_frame, textvariable=self.clean_mask_var, width=18).grid(
            row=1, column=1, sticky="w", pady=(6, 0)
        )
        ttk.Button(clean_frame, text="Apply Cleaning", command=self._clean_profanity).grid(
            row=2, column=0, columnspan=2, sticky="we", pady=(8, 0)
        )

        preview_frame = ttk.LabelFrame(right, text="Dataset Preview", padding=8)
        preview_frame.pack(fill="both", expand=True)
        preview_grid = ttk.Frame(preview_frame)
        preview_grid.pack(fill="both", expand=True)
        self.preview_text = tk.Text(
            preview_grid,
            height=18,
            wrap="none",
            undo=False,
            exportselection=False,
            padx=20,
        )
        self.preview_text.tag_configure("left", justify="left")
        preview_v = ttk.Scrollbar(
            preview_grid, orient="vertical", command=self.preview_text.yview
        )
        preview_h = ttk.Scrollbar(
            preview_grid, orient="horizontal", command=self.preview_text.xview
        )
        self.preview_text.configure(
            yscrollcommand=preview_v.set, xscrollcommand=preview_h.set
        )
        self.preview_text.grid(row=0, column=0, sticky="nsew")
        preview_v.grid(row=0, column=1, sticky="ns")
        preview_h.grid(row=1, column=0, sticky="ew")
        preview_grid.rowconfigure(0, weight=1)
        preview_grid.columnconfigure(0, weight=1)
        self.preview_text.bind("<MouseWheel>", self._on_preview_mousewheel)
        self.preview_text.bind("<Enter>", self._preview_text_focus_in)

        result_frame = ttk.LabelFrame(right, text="Logs / Results", padding=8)
        result_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.result_text = tk.Text(result_frame, height=16, wrap="word")
        self.result_text.pack(fill="both", expand=True)

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
            self.file_var.set(path)

    def _load_dataset(self) -> None:
        raw_path = self.file_var.get().strip()
        if not raw_path:
            messagebox.showerror("Error", "Please browse file first.")
            return
        path = Path(raw_path)
        if not path.exists():
            messagebox.showerror("Error", "Please select a valid dataset file.")
            return
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
        # Main app goal: scan immediately after loading.
        self._scan_profanity()

    def _show_preview(self, df: pd.DataFrame) -> None:
        self.preview_text.delete("1.0", END)
        # Use TSV-style preview to avoid padded right-aligned DataFrame formatting.
        n_show = min(len(df), PREVIEW_MAX_ROWS)
        preview_df = df.head(n_show).fillna("").astype(str)
        preview_df = preview_df.apply(
            lambda col: col.str.replace("\n", "\\n", regex=False).str.replace("\t", " ", regex=False)
        )
        cols = list(preview_df.columns)
        text_col = self.text_col_var.get().strip()
        priority: list[str] = []
        if text_col and text_col in cols:
            priority.append(text_col)
        for name in ("processed_text", "augmented_text"):
            if name in cols and name not in priority:
                priority.append(name)
        for name in ("has_profanity", "profanity_count", "profanity_matches"):
            if name in cols and name not in priority:
                priority.append(name)
        for name in ("clean_text", "clean_replacements"):
            if name in cols and name not in priority:
                priority.append(name)
        rest = [c for c in cols if c not in priority]
        if priority:
            preview_df = preview_df[priority + rest]
        preview_str = preview_df.to_csv(sep="\t", index=False)
        self.preview_text.insert("1.0", preview_str, "left")
        parent = self.preview_text.master.master
        if isinstance(parent, ttk.LabelFrame):
            if len(df) > n_show:
                parent.configure(
                    text=f"Dataset Preview (first {n_show:,} of {len(df):,} rows — scroll to view)"
                )
            else:
                parent.configure(
                    text=f"Dataset Preview ({len(df):,} rows — scroll to view)"
                )

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

        self.prep_vars["lowercase"].set(cfg.lowercase)
        self.prep_vars["remove_punctuation"].set(cfg.remove_punctuation)
        self.prep_vars["remove_stopwords"].set(cfg.remove_stopwords)
        self.prep_vars["remove_numbers"].set(cfg.remove_numbers)
        self.prep_vars["normalize_whitespace"].set(cfg.normalize_whitespace)

        self._log("Suggested preprocessing updated from dataset statistics:")
        for key, value in stats.items():
            self._log(f"- {key}: {value}")

    def _build_prep_config(self) -> PreprocessConfig:
        return PreprocessConfig(
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
        self.preview_text.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

    def _preview_text_focus_in(self, _event: tk.Event) -> None:
        self.preview_text.focus_set()


def run() -> None:
    root = Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    ToxicCommentApp(root)
    root.mainloop()
