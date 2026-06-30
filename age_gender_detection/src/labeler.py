import tkinter as tk
from PIL import Image, ImageTk
import os
import csv

class HairLabeler:
    def __init__(self, image_folder, output_csv="hair_labels.csv"):
        self.image_folder = image_folder
        self.output_csv   = output_csv
        self.images       = [f for f in os.listdir(image_folder)
                             if f.lower().endswith(('.jpg','.jpeg','.png'))]
        self.index        = 0
        self.labels       = []

        # Resume if already started
        if os.path.exists(output_csv):
            with open(output_csv, 'r') as f:
                reader = csv.DictReader(f)
                self.labels = list(reader)
                self.index  = len(self.labels)
                print(f"Resuming from {self.index}")

        # Window
        self.root = tk.Tk()
        self.root.title("Hair Labeler")
        self.root.geometry("620x820")
        self.root.configure(bg='#1a1a1a')

        # Progress text
        self.progress_label = tk.Label(
            self.root, text="",
            font=("Arial", 14, "bold"),
            bg='#1a1a1a', fg='white'
        )
        self.progress_label.pack(pady=8)

        # Progress bar
        self.bar_canvas = tk.Canvas(
            self.root, width=500, height=18,
            bg='#333', highlightthickness=0
        )
        self.bar_canvas.pack(pady=4)

        # Image display
        self.img_label = tk.Label(self.root, bg='#1a1a1a')
        self.img_label.pack(pady=8)

        # File info
        self.file_label = tk.Label(
            self.root, text="",
            font=("Arial", 9),
            bg='#1a1a1a', fg='#666'
        )
        self.file_label.pack()

        # Age + gender from filename
        self.info_label = tk.Label(
            self.root, text="",
            font=("Arial", 13, "bold"),
            bg='#1a1a1a', fg='#FFD700'
        )
        self.info_label.pack(pady=4)

        # Suggestion based on gender
        self.suggestion_label = tk.Label(
            self.root, text="",
            font=("Arial", 12),
            bg='#1a1a1a', fg='white'
        )
        self.suggestion_label.pack(pady=2)

        # Instructions
        tk.Label(
            self.root,
            text="L = Long Hair     S = Short Hair     Space = Skip",
            font=("Arial", 11),
            bg='#1a1a1a', fg='#aaa'
        ).pack(pady=6)

        # Buttons
        btn_frame = tk.Frame(self.root, bg='#1a1a1a')
        btn_frame.pack(pady=8)

        tk.Button(
            btn_frame,
            text="💇 LONG HAIR\n(press L)",
            command=lambda: self.label('long'),
            bg='#E8593C', fg='white',
            font=("Arial", 14, "bold"),
            width=14, height=3,
            relief='flat', cursor='hand2'
        ).grid(row=0, column=0, padx=15)

        tk.Button(
            btn_frame,
            text="✂️ SHORT HAIR\n(press S)",
            command=lambda: self.label('short'),
            bg='#3B8BD4', fg='white',
            font=("Arial", 14, "bold"),
            width=14, height=3,
            relief='flat', cursor='hand2'
        ).grid(row=0, column=1, padx=15)

        tk.Button(
            btn_frame,
            text="SKIP (Space)",
            command=lambda: self.label('skip'),
            bg='#444', fg='#aaa',
            font=("Arial", 10),
            width=12, height=2,
            relief='flat', cursor='hand2'
        ).grid(row=1, column=0, columnspan=2, pady=6)

        # Stats
        self.stats_label = tk.Label(
            self.root, text="",
            font=("Arial", 11),
            bg='#1a1a1a', fg='#3B8BD4'
        )
        self.stats_label.pack(pady=4)

        # Keyboard shortcuts
        self.root.bind('l', lambda e: self.label('long'))
        self.root.bind('L', lambda e: self.label('long'))
        self.root.bind('s', lambda e: self.label('short'))
        self.root.bind('S', lambda e: self.label('short'))
        self.root.bind('<space>', lambda e: self.label('skip'))

        self.show_image()
        self.root.mainloop()

    def show_image(self):
        if self.index >= len(self.images):
            self.finish()
            return

        img_file = self.images[self.index]
        img_path = os.path.join(self.image_folder, img_file)

        # Load image
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((380, 380))
        self.tk_img = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tk_img)
        self.file_label.config(text=img_file)

        # Parse age + gender from UTKFace filename
        try:
            parts      = img_file.split('_')
            age        = int(parts[0])
            gender     = int(parts[1])
            gender_str = "Female" if gender == 1 else "Male"
            self.info_label.config(
                text=f"Age: {age}  |  Biological: {gender_str}"
            )
            # Suggest based on gender
            if gender == 1:
                self.suggestion_label.config(
                    text="💡 Suggested → LONG (female)",
                    fg='#E8593C'
                )
            else:
                self.suggestion_label.config(
                    text="💡 Suggested → SHORT (male)",
                    fg='#3B8BD4'
                )
        except:
            self.info_label.config(text="")
            self.suggestion_label.config(text="")

        # Progress
        pct = int((self.index / len(self.images)) * 100)
        self.progress_label.config(
            text=f"Image {self.index+1} / {len(self.images)}  ({pct}%)"
        )

        # Progress bar
        self.bar_canvas.delete("all")
        fill_width = int(500 * self.index / len(self.images))
        self.bar_canvas.create_rectangle(
            0, 0, fill_width, 18,
            fill='#E8593C', outline=''
        )

        # Stats
        long_c  = sum(1 for l in self.labels if l['hair_length']=='long')
        short_c = sum(1 for l in self.labels if l['hair_length']=='short')
        skip_c  = sum(1 for l in self.labels if l['hair_length']=='skip')
        self.stats_label.config(
            text=f"Long: {long_c}  |  Short: {short_c}  |  Skipped: {skip_c}"
        )

    def label(self, hair_length):
        if self.index >= len(self.images):
            return
        img_file = self.images[self.index]
        self.labels.append({
            'filename'   : img_file,
            'img_path'   : os.path.join(self.image_folder, img_file),
            'hair_length': hair_length
        })
        self.index += 1
        if self.index % 50 == 0:
            self.save_labels()
            print(f"Auto saved — {self.index} done")
        self.show_image()

    def save_labels(self):
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['filename','img_path','hair_length']
            )
            writer.writeheader()
            writer.writerows(self.labels)
        print(f"Saved {len(self.labels)} labels")

    def finish(self):
        self.save_labels()
        for widget in self.root.winfo_children():
            widget.destroy()

        long_c  = sum(1 for l in self.labels if l['hair_length']=='long')
        short_c = sum(1 for l in self.labels if l['hair_length']=='short')
        skip_c  = sum(1 for l in self.labels if l['hair_length']=='skip')

        tk.Label(
            self.root, text="✅ Done!",
            font=("Arial", 24, "bold"),
            bg='#1a1a1a', fg='#3B8BD4'
        ).pack(pady=30)

        tk.Label(
            self.root,
            text=f"Long   : {long_c}\nShort  : {short_c}\nSkipped: {skip_c}\n\nSaved to:\n{self.output_csv}",
            font=("Arial", 14),
            bg='#1a1a1a', fg='white',
            justify='left'
        ).pack(pady=10)

# ── Run ────────────────────────────────────────────────────
labeler = HairLabeler(
    image_folder = r"C:\age_gender_detection\to_label",
    output_csv   = r"C:\age_gender_detection\hair_labels.csv"
)