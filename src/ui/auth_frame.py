"""Login and registration gate shown before the main workbench."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from auth import UserRecord, get_store


class AuthFrame(ttk.Frame):
    """Username/password login with optional registration."""

    # Tailwind-ish palette (to mimic the React/Tailwind design)
    _BG = "#E6E6E7"
    _CARD_BG = "#ffffff"
    _BORDER = "#e5e7eb"
    _TEXT = "#111827"
    _MUTED = "#6b7280"
    _PRIMARY = "#2563eb"
    _PRIMARY_HOVER = "#1d4ed8"
    _DESTRUCTIVE = "#dc2626"

    def _rounded_rect_points(self, x1: int, y1: int, x2: int, y2: int, r: int) -> list[int]:
        r = max(0, min(int(r), int((x2 - x1) / 2), int((y2 - y1) / 2)))
        return [
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

    def _build_rounded_shadow_card(
        self,
        parent: tk.Misc,
        *,
        width: int,
        radius: int = 18,
        shadow_dx: int = 0,
        shadow_dy: int = 0,
        shadow_color: str = "#000000",
        shadow_stipple: str = "gray50",
        padding: int = 22,
    ) -> tuple[tk.Canvas, tk.Frame]:
        """
        Create a rounded 'card' with a soft shadow using a Canvas.

        Tkinter doesn't support real border-radius or alpha shadows for Frames,
        so we approximate via Canvas polygons + stipple.
        """
        canvas = tk.Canvas(parent, bg=self._BG, highlightthickness=0, bd=0)
        canvas.configure(width=width + (padding * 2), height=300)

        inner = tk.Frame(canvas, bg=self._CARD_BG)
        inner_id = canvas.create_window((padding, padding), window=inner, anchor="nw", width=width)

        def redraw(_event: tk.Event | None = None) -> None:
            canvas.delete("card")
            canvas.update_idletasks()
            w = width + (padding * 2)
            content_h = max(1, int(inner.winfo_reqheight()))
            h = content_h + (padding * 2)
            canvas.configure(width=w, height=h + max(0, shadow_dy) + 2)

            x1, y1 = 1, 1
            x2, y2 = w - 1 - max(0, shadow_dx), h - 1 - max(0, shadow_dy)

            # Shadow removed (requested).

            # Card fill
            canvas.create_polygon(
                self._rounded_rect_points(x1, y1, x2, y2, radius),
                smooth=True,
                splinesteps=36,
                fill=self._CARD_BG,
                outline=self._BORDER,
                width=1,
                tags=("card",),
            )
            canvas.tag_lower("card")
            canvas.coords(inner_id, padding, padding)
            canvas.itemconfigure(inner_id, width=width)

        inner.bind("<Configure>", redraw)
        redraw()
        return canvas, inner

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_success: Callable[[UserRecord], None],
        store=None,
    ) -> None:
        super().__init__(master, padding=0)
        self._on_success = on_success
        self._store = store or get_store()
        self._register_mode = tk.BooleanVar(value=False)
        self._is_loading = False

        self.pack(fill="both", expand=True)

        bg = tk.Frame(self, bg=self._BG)
        bg.pack(fill="both", expand=True)
        bg.columnconfigure(0, weight=1)
        bg.rowconfigure(0, weight=1)
        bg.rowconfigure(2, weight=1)

        card_canvas, card_inner = self._build_rounded_shadow_card(
            bg,
            width=440,
            radius=18,
            shadow_dx=0,
            shadow_dy=0,
            padding=22,
        )
        card_canvas.grid(row=1, column=0, padx=18, pady=18, sticky="n")
        content = card_inner

        tk.Label(content, text="Welcome back", bg=self._CARD_BG, fg=self._TEXT, font=("Segoe UI", 18, "bold")).pack(
            anchor="w"
        )
        tk.Label(
            content,
            text="Sign in to your ToxiGuard account",
            bg=self._CARD_BG,
            fg=self._MUTED,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(4, 16))

        form = tk.Frame(content, bg=self._CARD_BG)
        form.pack(fill="x")

        self._user_placeholder = "name@example.com"
        self._pass_placeholder = "••••••••"
        self._confirm_placeholder = "••••••••"

        self.user_entry, self._user_err = self._add_text_input(
            form, label="Username", icon="@", placeholder=self._user_placeholder
        )
        tk.Frame(form, bg=self._CARD_BG, height=10).pack()
        self.pass_entry, self._pass_toggle_btn, self._pass_err = self._add_password_with_toggle(
            form, label="Password", icon="🔒", placeholder=self._pass_placeholder
        )

        self.confirm_row = tk.Frame(form, bg=self._CARD_BG)
        tk.Frame(form, bg=self._CARD_BG, height=10).pack()
        self.confirm_entry, self._confirm_toggle_btn, self._confirm_err = self._add_password_with_toggle(
            self.confirm_row, label="Confirm password", icon="🔒", placeholder=self._confirm_placeholder
        )

        self._primary = self._btn_primary(content, text="Sign in", command=self._submit)
        self._primary.pack(fill="x", pady=(12, 0))

        footer = tk.Frame(content, bg=self._CARD_BG)
        footer.pack(fill="x", pady=(14, 0))
        self._footer_text = tk.Label(
            footer,
            text="Don't have an account?",
            bg=self._CARD_BG,
            fg=self._MUTED,
            font=("Segoe UI", 9),
        )
        self._footer_text.pack(side="left")
        self._footer_link = self._link(footer, text="Register", command=self._toggle_register_link)
        self._footer_link.pack(side="left", padx=(6, 0))

        # Keep old flow: Enter submits
        self._toggle_register()
        self.user_entry.focus_set()
        self.bind("<Return>", lambda _e: self._submit())

    def _btn_primary(self, parent: tk.Misc, *, text: str, command: Callable[[], None]) -> tk.Button:
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=self._PRIMARY,
            fg="#ffffff",
            activebackground=self._PRIMARY_HOVER,
            activeforeground="#ffffff",
            bd=0,
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            padx=14,
            pady=10,
        )
        return btn

    def _link(self, parent: tk.Misc, *, text: str, command: Callable[[], None]) -> tk.Label:
        lbl = tk.Label(
            parent,
            text=text,
            bg=self._CARD_BG,
            fg=self._PRIMARY,
            cursor="hand2",
            font=("Segoe UI", 9, "underline"),
        )
        lbl.bind("<Button-1>", lambda _e: command())
        return lbl

    def _get_entry_value(self, entry: tk.Entry, placeholder: str) -> str:
        v = entry.get().strip()
        if v == placeholder and entry.cget("fg") == self._MUTED:
            return ""
        return v

    def _add_text_input(
        self,
        parent: tk.Misc,
        *,
        label: str,
        icon: str,
        placeholder: str,
    ) -> tuple[tk.Entry, tk.Label]:
        tk.Label(parent, text=label, bg=self._CARD_BG, fg=self._TEXT, font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=(0, 6)
        )

        row = tk.Frame(parent, bg=self._CARD_BG, highlightthickness=1, highlightbackground=self._BORDER)
        row.pack(fill="x")
        row.columnconfigure(1, weight=1)

        tk.Label(row, text=icon, bg=self._CARD_BG, fg=self._MUTED, font=("Segoe UI", 11)).grid(
            row=0, column=0, padx=(10, 6), pady=10, sticky="w"
        )

        entry = tk.Entry(
            row,
            bd=0,
            relief="flat",
            font=("Segoe UI", 10),
            fg=self._TEXT,
            bg=self._CARD_BG,
            insertbackground=self._TEXT,
        )
        entry.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=10)
        entry.insert(0, placeholder)
        entry.configure(fg=self._MUTED)

        def on_focus_in(_e: tk.Event) -> None:
            if entry.get() == placeholder and entry.cget("fg") == self._MUTED:
                entry.delete(0, "end")
                entry.configure(fg=self._TEXT)

        def on_focus_out(_e: tk.Event) -> None:
            if not entry.get().strip():
                entry.insert(0, placeholder)
                entry.configure(fg=self._MUTED)

        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

        err = tk.Label(parent, text="", bg=self._CARD_BG, fg=self._DESTRUCTIVE, font=("Segoe UI", 9))
        err.pack(anchor="w", pady=(6, 0))
        return entry, err

    def _add_password_with_toggle(
        self,
        parent: tk.Misc,
        *,
        label: str,
        icon: str,
        placeholder: str,
    ) -> tuple[tk.Entry, tk.Button, tk.Label]:
        tk.Label(parent, text=label, bg=self._CARD_BG, fg=self._TEXT, font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=(0, 6)
        )

        row = tk.Frame(parent, bg=self._CARD_BG, highlightthickness=1, highlightbackground=self._BORDER)
        row.pack(fill="x")
        row.columnconfigure(1, weight=1)

        tk.Label(row, text=icon, bg=self._CARD_BG, fg=self._MUTED, font=("Segoe UI", 11)).grid(
            row=0, column=0, padx=(10, 6), pady=10, sticky="w"
        )

        entry = tk.Entry(
            row,
            bd=0,
            relief="flat",
            show="*",
            font=("Segoe UI", 10),
            fg=self._TEXT,
            bg=self._CARD_BG,
            insertbackground=self._TEXT,
        )
        entry.grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=10)
        entry.insert(0, placeholder)
        entry.configure(fg=self._MUTED)

        def on_focus_in(_e: tk.Event) -> None:
            if entry.get() == placeholder and entry.cget("fg") == self._MUTED:
                entry.delete(0, "end")
                entry.configure(fg=self._TEXT)

        def on_focus_out(_e: tk.Event) -> None:
            if not entry.get().strip():
                entry.insert(0, placeholder)
                entry.configure(fg=self._MUTED)

        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

        btn = tk.Button(
            row,
            text="Show",
            bd=0,
            relief="flat",
            bg=self._CARD_BG,
            fg=self._MUTED,
            activebackground=self._CARD_BG,
            activeforeground=self._TEXT,
            cursor="hand2",
            font=("Segoe UI", 9, "bold"),
        )
        btn.grid(row=0, column=2, padx=(0, 10), pady=6, sticky="e")

        def toggle() -> None:
            showing = entry.cget("show") == ""
            entry.configure(show="*" if showing else "")
            btn.configure(text="Show" if showing else "Hide")

        btn.configure(command=toggle)

        err = tk.Label(parent, text="", bg=self._CARD_BG, fg=self._DESTRUCTIVE, font=("Segoe UI", 9))
        err.pack(anchor="w", pady=(6, 0))

        return entry, btn, err

    def _toggle_register_link(self) -> None:
        self._register_mode.set(not self._register_mode.get())
        self._toggle_register()

    def _toggle_register(self) -> None:
        reg = self._register_mode.get()
        if reg:
            self.confirm_row.pack(fill="x")
            self._primary.configure(text="Register")
            self._footer_text.configure(text="Already have an account?")
            self._footer_link.configure(text="Sign in")
        else:
            self.confirm_row.pack_forget()
            self._primary.configure(text="Sign in")
            self._footer_text.configure(text="Don't have an account?")
            self._footer_link.configure(text="Register")
            self._confirm_err.configure(text="")

    def _validate_login_form(self) -> bool:
        ok = True
        self._user_err.configure(text="")
        self._pass_err.configure(text="")

        user = self._get_entry_value(self.user_entry, self._user_placeholder)
        pw = self._get_entry_value(self.pass_entry, self._pass_placeholder)

        if not user:
            self._user_err.configure(text="Username is required")
            ok = False
        if not pw:
            self._pass_err.configure(text="Password is required")
            ok = False
        elif len(pw) < 6:
            self._pass_err.configure(text="Password must be at least 6 characters")
            ok = False
        return ok

    def _validate_register_form(self) -> bool:
        ok = self._validate_login_form()
        self._confirm_err.configure(text="")
        pw = self._get_entry_value(self.pass_entry, self._pass_placeholder)
        pw2 = self._get_entry_value(self.confirm_entry, self._confirm_placeholder)
        if not pw2:
            self._confirm_err.configure(text="Confirm password is required")
            return False
        if pw != pw2:
            self._confirm_err.configure(text="Passwords do not match")
            return False
        return ok

    def _set_loading(self, loading: bool) -> None:
        self._is_loading = bool(loading)
        if self._is_loading:
            self._primary.configure(
                text="Signing in..." if not self._register_mode.get() else "Creating account...",
                state="disabled",
            )
        else:
            self._primary.configure(
                text="Register" if self._register_mode.get() else "Sign in",
                state="normal",
            )

    def _submit(self) -> None:
        if self._is_loading:
            return

        if self._register_mode.get():
            if not self._validate_register_form():
                return
        else:
            if not self._validate_login_form():
                return

        self._set_loading(True)
        self.after(30, self._submit_impl)

    def _submit_impl(self) -> None:
        user = self._get_entry_value(self.user_entry, self._user_placeholder)
        pw = self._get_entry_value(self.pass_entry, self._pass_placeholder)

        try:
            if self._register_mode.get():
                ok, msg = self._store.register(user, pw)
                if not ok:
                    messagebox.showerror("Registration", msg)
                    return
                messagebox.showinfo("Registration", msg)
                rec = self._store.authenticate(user, pw)
                if rec is not None:
                    self._on_success(rec)
                return

            rec = self._store.authenticate(user, pw)
            if rec is None:
                messagebox.showerror(
                    "Sign in",
                    "Could not sign in. Check your username and password, or whether the account is disabled.",
                )
                return
            self._on_success(rec)
        finally:
            self._set_loading(False)
