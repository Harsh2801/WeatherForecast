import os.path
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import pandas as pd
from PIL import ImageTk, Image

big_heading_font = ('Helvetica', 28, 'bold')
welcome_default_background = 'white'
button_font = ('Lao UI', 13)
df = pd.DataFrame()


def open_dataset():
    global df
    dataframe_path = filedialog.askopenfile(mode='r', filetypes=[('CSV files', '*.csv')])
    if dataframe_path:
        dataframe_path = os.path.abspath(dataframe_path.name)
        df = pd.read_csv(dataframe_path, sep=';')


class WelcomeWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        self.geometry('720x450')
        self.title('All in One Data Cleaner')
        self.resizable(False, False)
        self.configure(bg=welcome_default_background)

        top_frame = LabelFrame(self, bg=welcome_default_background, height=100, width=720, bd=0)
        Label(top_frame, text='Happy Training Machines', font=big_heading_font,
              bg=welcome_default_background, fg='SlateBlue2') \
            .place(relwidth=.8, relx=.1, y=20)
        top_frame.pack()

        mid_frame = LabelFrame(self, width=720, bd=0, height=300)
        mid_frame.pack()

        left_button_frame = LabelFrame(mid_frame, bg=welcome_default_background, width=180, bd=0)
        self.dataset_btn = Button(left_button_frame, text='Choose Dataset', font=button_font, activebackground='#ffcc66',
                                  bg='#141414', relief='groove', activeforeground='#141414', fg='#ffcc66',
                                  border=0, command=open_dataset)
        self.dataset_btn.place(relx=.2, rely=.45)
        left_button_frame.pack(side=LEFT, fill=Y)

        display_image = ImageTk.PhotoImage(Image.open("Images/pipes.jpg")
                                           .resize((350, 300), Image.ANTIALIAS))
        image_frame = Label(mid_frame, image=display_image, width=360, background=welcome_default_background, bd=0)
        image_frame.image = display_image
        image_frame.pack(side=LEFT, fill=Y)

        right_button_frame = LabelFrame(mid_frame, bg=welcome_default_background, width=180, bd=0)
        self.begin_btn = Button(right_button_frame, text='Begin Cleaning', font=button_font, activebackground='white',
                                bg='orange red', relief='groove', activeforeground='orange red', fg='white', border=0,
                                command=self.destroy)
        self.begin_btn.place(relx=.2, rely=.45)
        right_button_frame.pack(side=LEFT, fill=Y)

        Button(self, text='EXIT', command=self.destroy, font=button_font, width=8, activebackground='firebrick1',
               relief='groove').place(rely=.90, relx=.45)

        self.df = pd.DataFrame()
