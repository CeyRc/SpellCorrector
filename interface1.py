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

       def animate_logo(self):
        size_step = 4
        y_step = 0.012

        size_done = False
        y_done = False

        if self.current_size > self.target_size:
            self.current_size -= size_step
            if self.current_size < self.target_size:
                self.current_size = self.target_size
        else:
            size_done = True

        if self.y_pos > self.target_y:
            self.y_pos -= y_step
            if self.y_pos < self.target_y:
                self.y_pos = self.target_y
        else:
            y_done = True

        resized = self.logo_image.resize((self.current_size, self.current_size))
        self.logo_photo = ImageTk.PhotoImage(resized)
        self.logo_label.config(image=self.logo_photo)
        self.logo_label.place(relx=0.55, rely=self.y_pos, anchor="center")

        if self.y_pos <= 0.1:
            self.title_label.place(relx=0.5, rely=self.y_pos + 0.2, anchor="center")  # Ortalandı

        if not (size_done and y_done):
            self.root.after(30, self.animate_logo)
        else:
            self.logo_label.place(relx=0.55, rely=self.target_y, anchor="center")
            self.title_label.place(relx=0.5, rely=self.target_y + 0.2, anchor="center")  # Ortalandı
            self.show_main_menu()