"""Login and registration gate shown before the main workbench."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from auth import UserRecord, get_store


class AuthFrame(ttk.Frame):
    """Username/password login with optional registration."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_success: Callable[[UserRecord], None],
        store=None,
    ) -> None:
        super().__init__(master, padding=24)
        self._on_success = on_success
        self._store = store or get_store()
        self._register_mode = tk.BooleanVar(value=False)

        self.pack(fill="both", expand=True)

        title = ttk.Label(self, text="Toxic Comment Workbench", font=("Segoe UI", 16, "bold"))
        title.pack(anchor="center", pady=(0, 4))
        sub = ttk.Label(self, text="Sign in to analyze content and keep a personal history.", font=("Segoe UI", 10))
        sub.pack(anchor="center", pady=(0, 18))

        form = ttk.Frame(self)
        form.pack(fill="x", pady=8)
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Username").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=4)
        self.user_entry = ttk.Entry(form, width=36)
        self.user_entry.grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(form, text="Password").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=4)
        self.pass_entry = ttk.Entry(form, width=36, show="*")
        self.pass_entry.grid(row=1, column=1, sticky="ew", pady=4)

        self.confirm_row = ttk.Frame(form)
        ttk.Label(self.confirm_row, text="Confirm").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=4)
        self.confirm_entry = ttk.Entry(self.confirm_row, width=36, show="*")
        self.confirm_entry.grid(row=0, column=1, sticky="ew", pady=4)
        self.confirm_row.columnconfigure(1, weight=1)

        reg_chk = ttk.Checkbutton(
            self,
            text="Create a new account",
            variable=self._register_mode,
            command=self._toggle_register,
        )
        reg_chk.pack(anchor="w", pady=(12, 8))

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", pady=(8, 0))
        self._primary = ttk.Button(btn_row, text="Sign in", command=self._submit)
        self._primary.pack(side="left")
        ttk.Button(btn_row, text="Quit", command=self._quit_app).pack(side="right")

        self._toggle_register()
        self.user_entry.focus_set()
        self.bind("<Return>", lambda _e: self._submit())

    def _quit_app(self) -> None:
        root = self.winfo_toplevel()
        root.destroy()

    def _toggle_register(self) -> None:
        reg = self._register_mode.get()
        if reg:
            self.confirm_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)
            self._primary.configure(text="Register")
        else:
            self.confirm_row.grid_remove()
            self._primary.configure(text="Sign in")

    def _submit(self) -> None:
        user = self.user_entry.get().strip()
        pw = self.pass_entry.get()
        if self._register_mode.get():
            pw2 = self.confirm_entry.get()
            if pw != pw2:
                messagebox.showerror("Registration", "Passwords do not match.")
                return
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
