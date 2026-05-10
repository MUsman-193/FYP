"""UI package."""

import tkinter as tk
from tkinter import ttk

from auth import UserRecord

from .auth_frame import AuthFrame
from .tk_app import ToxicCommentApp


def run() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    root.title("Toxic Comment Classification Workbench")
    root.geometry("520x420")

    def open_app(user: UserRecord) -> None:
        root.geometry("1280x800")
        for w in root.winfo_children():
            w.destroy()
        ToxicCommentApp(root, user=user, on_logout=lambda: show_login(root))

    def show_login(r: tk.Tk) -> None:
        r.geometry("520x420")
        for w in r.winfo_children():
            w.destroy()
        AuthFrame(r, on_success=open_app)

    show_login(root)
    root.mainloop()


__all__ = ["run"]
