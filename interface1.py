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

    def show_main_menu(self):
        word_icon = ImageTk.PhotoImage(Image.open("word_icon.jpg").resize((100, 100)))
        self.word_button = tk.Button(self.root, image=word_icon,
                                     command=self.open_word_screen, bg="#f8f8f6", bd=0,
                                     highlightthickness=0, relief="flat", activebackground="#f8f8f6")
        self.word_button.image = word_icon
        self.word_button.place(relx=0.3, rely=0.60, anchor="center")

        self.word_label = tk.Label(self.root, text="For Word Correction",
                                   font=("Arial", 15), bg="#f8f8f6", fg="#444444")
        self.word_label.place(relx=0.3, rely=0.72, anchor="center")

        sentence_icon = ImageTk.PhotoImage(Image.open("sentence_icon.png").resize((100, 100)))
        self.sentence_button = tk.Button(self.root, image=sentence_icon,
                                         command=self.open_sentence_screen, bg="#f8f8f6", bd=0,
                                         highlightthickness=0, relief="flat", activebackground="#f8f8f6")
        self.sentence_button.image = sentence_icon
        self.sentence_button.place(relx=0.7, rely=0.60, anchor="center")

        self.sentence_label = tk.Label(self.root, text="For Sentence Correction",
                                       font=("Arial", 15), bg="#f8f8f6", fg="#444444")
        self.sentence_label.place(relx=0.7, rely=0.72, anchor="center")

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
            self.title_label.place(relx=0.5, rely=self.y_pos + 0.2, anchor="center")

        if not (size_done and y_done):
            self.root.after(30, self.animate_logo)
        else:
            self.logo_label.place(relx=0.55, rely=self.target_y, anchor="center")
            self.title_label.place(relx=0.5, rely=self.target_y + 0.2, anchor="center")
            self.show_main_menu()

    # Bu iki fonksiyonu kendin doldurmalısın.
    def open_word_screen(self):
        # Word correction ekranını açacak fonksiyon
        pass

    def open_sentence_screen(self):
        # Sentence correction ekranını açacak fonksiyon
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = SpellCorrectorApp(root)
    root.mainloop()
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

    def show_main_menu(self):
        word_icon = ImageTk.PhotoImage(Image.open("word_icon.jpg").resize((100, 100)))
        self.word_button = tk.Button(self.root, image=word_icon,
                                     command=self.open_word_screen, bg="#f8f8f6", bd=0,
                                     highlightthickness=0, relief="flat", activebackground="#f8f8f6")
        self.word_button.image = word_icon
        self.word_button.place(relx=0.3, rely=0.60, anchor="center")

        self.word_label = tk.Label(self.root, text="For Word Correction",
                                   font=("Arial", 15), bg="#f8f8f6", fg="#444444")
        self.word_label.place(relx=0.3, rely=0.72, anchor="center")

        sentence_icon = ImageTk.PhotoImage(Image.open("sentence_icon.png").resize((100, 100)))
        self.sentence_button = tk.Button(self.root, image=sentence_icon,
                                         command=self.open_sentence_screen, bg="#f8f8f6", bd=0,
                                         highlightthickness=0, relief="flat", activebackground="#f8f8f6")
        self.sentence_button.image = sentence_icon
        self.sentence_button.place(relx=0.7, rely=0.60, anchor="center")

        self.sentence_label = tk.Label(self.root, text="For Sentence Correction",
                                       font=("Arial", 15), bg="#f8f8f6", fg="#444444")
        self.sentence_label.place(relx=0.7, rely=0.72, anchor="center")



    def open_sentence_screen(self):
        print("🟡 Sentence ekranı başlatılıyor...")
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        subprocess.Popen(["python", "sentence_gui.py"], env=env)
        self.root.withdraw()  # Ana pencereyi gizler, kapatmaz

    def open_word_screen(self):
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        subprocess.Popen(["python", "wordgui.py"], env=env)
        self.root.destroy()  # arayüz kapanacak

    def new_window(self, title, label_text):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("400x200")
        win.configure(bg="#f8f8f6")
        tk.Label(win, text=label_text, bg="#f8f8f6").pack(pady=10)
        tk.Entry(win, width=40).pack(pady=5)
        tk.Button(win, text="Correct", command=lambda: print("Correction logic here")).pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = SpellCorrectorApp(root)
    root.mainloop()