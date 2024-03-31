import tkinter as tk
from tkinter import filedialog
from source.app import Application

app_ico = 'media/app_leyser_logo.png'
app_title = 'Face Blendshapes Generation - ARLiveLink UE5'
app_size = '400x400'

def _set_app_icon(master: tk, photo_path):
    logo = tk.PhotoImage(file=photo_path)
    master.wm_iconphoto(False, logo)
    
if __name__ == "__main__":
    root = tk.Tk()
    # Adjust size
    root.geometry(app_size)
    root.iconbitmap(app_ico)
    _set_app_icon(root, app_ico)
    root.title(app_title)
    root.config(bg='#222831')
    Application(root)    
    root.mainloop()


