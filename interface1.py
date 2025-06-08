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

        self.logo_image = Image.open("logo3.jpeg")
        self.start_size = 800
        self.target_size = 300
        self.current_size = self.start_size

        self.y_pos = 0.5
        self.target_y = 0.2

        self.logo_photo = ImageTk.PhotoImage(self.logo_image.resize((self.current_size, self.current_size)))
        self.logo_label = tk.Label(self.root, image=self.logo_photo, bg="#f8f8f6")
        self.logo_label.place(relx=0.55, rely=self.y_pos, anchor="center")

        self.title_label = tk.Label(self.root, text="Welcome to Spell Corrector App",
                                    font=("Arial", 22, "bold"), bg="#f8f8f6", fg="#5222aa")

        self.animate_logo()