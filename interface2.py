
import tkinter as tk
from PIL import Image, ImageTk
import subprocess
import os

class SpellCorrectorApp:
    def __init__(self, root):
        self.root = root

        self.root.title("Spell Corrector")
        self.root.geometry("800x600")
        self.root.configure(bg="#f8f8f6")

  
        self.title_label = tk.Label(self.root, text="Spell Corrector App",
                                    font=("Arial", 22, "bold"), bg="#f8f8f6", fg="#5222aa")
        self.title_label.place(relx=0.5, rely=0.05, anchor="center")

        
        self.logo_image = Image.open("logo3.jpeg")
        target_size = 300
        self.logo_photo = ImageTk.PhotoImage(self.logo_image.resize((target_size, target_size)))
        self.logo_label = tk.Label(self.root, image=self.logo_photo, bg="#f8f8f6")
        self.logo_label.place(relx=0.55, rely=0.3, anchor="center")
        self.title_label.lift()

        self.show_main_menu()
