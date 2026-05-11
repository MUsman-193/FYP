"""UI package."""

import tkinter as tk
from tkinter import ttk

from auth import UserRecord, get_store

from .auth_frame import AuthFrame
from .tk_app import ToxicCommentApp


def run() -> None:
    root = tk.Tk()
    store = get_store()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    root.title("Toxic Comment Classification Workbench")
    root.geometry("800x700")

    def open_app(user: UserRecord) -> None:
        store.save_session(user.id)
        root.geometry("1280x800")
        for w in root.winfo_children():
            w.destroy()
        ToxicCommentApp(root, user=user, on_logout=lambda: show_login(root, clear_session=True))

    def show_login(r: tk.Tk, *, clear_session: bool = False) -> None:
        if clear_session:
            store.clear_session()
        r.geometry("800x700")
        for w in r.winfo_children():
            w.destroy()
        AuthFrame(r, on_success=open_app)

    session_user = store.load_session()
    if session_user is not None:
        open_app(session_user)
    else:
        show_login(root)
    root.mainloop()


__all__ = ["run"]
