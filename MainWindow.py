import threading
import tkinter as tk
from tkinter import *
from tkinter import ttk
from functools import partial
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from GandhiWeather import IntroWindow
from sklearn.model_selection import train_test_split
from tkinter import messagebox

button_font = ('arial', 13)
small_btn_font = ('arial', 10)
default_text_font = ('Courier ', 10)
default_text_font_bold = ('Courier ', 10, 'bold')

default_button_options = {'activebackground': 'white', 'bg': 'RoyalBlue3', 'relief': 'groove',
                          'activeforeground': 'RoyalBlue3', 'width': '16',
                          'fg': 'white', 'font': button_font, 'bd': 1}

another_button_options = {'activebackground': 'black', 'bg': 'springgreen2', 'relief': 'groove',
                          'activeforeground': 'springgreen2', 'width': '10',
                          'fg': 'black', 'font': small_btn_font, 'bd': 1}


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.predict_window = None
        self.knn = None
        self.svm = None
        self.dtr = None
        self.one_hot_encode_window = None
        self.impute_window = None
        self.change_datatype_window = None
        self.get_column_window = None

        self.undo_stack = []
        self.redo_stack = []

        self.right_btn_list = None
        self.left_btn_list = None
        self.check_btn_vars = None
        self.selected_columns = []

        self.df = IntroWindow.df
        # self.df = pd.read_csv("Dataset/temps2.csv", sep=';')
        self.original_df = self.df.copy()
        self.columns = self.df.columns

        screen_width = 1366
        screen_height = 750

        self.resizable(False, False)
        self.geometry("%dx%d" % (screen_width, screen_height))
        self.title('Happy Data Cleaning')

        # label for showing rows x columns
        self.row_col_display = Label(fg='gray7', font=('Helvetica', 12, 'bold'))
        self.row_col_display.place(x=1360//2+120, y=2)

        # drop down list to get the target Column
        Label(self, text='Select Target Column: ').place(x=(1360//3), y=8)
        self.target_col = StringVar()
        self.target_col.set('None')
        self.target_col_option_menu = OptionMenu(self, self.target_col, *self.columns)
        self.target_col_option_menu.place(x=1360//3+120, y=2)

        # to show dataframe
        dataset_frame = LabelFrame(self, text='DataFrame', border=0, padx=10)
        dataset_frame.place(y=40, height=screen_height * 4 / 5 - 40, width=screen_width)
        self.tree_view = ttk.Treeview(dataset_frame)
        self.tree_view.place(relheight=1, relwidth=1)
        treescrolly = Scrollbar(dataset_frame, orient="vertical",
                                command=self.tree_view.yview)  # command means update the y-axis view of the widget
        treescrollx = Scrollbar(dataset_frame, orient="horizontal",
                                command=self.tree_view.xview)  # command means update the x-axis view of the widget
        self.tree_view.configure(xscrollcommand=treescrollx.set,
                                 yscrollcommand=treescrolly.set)  # assign the scrollbars to the Treeview Widget
        treescrollx.pack(side="bottom", fill="x")  # make the scrollbar fill the x-axis with the Treeview widget
        treescrolly.pack(side="right", fill="y")  # make the scrollbar fill the y-axis with the Treeview widget
        self.show_dataset()

        # bottom Frame
        manipulate_button_frame = LabelFrame(self, border=0, text='Manipulate DataFrame', pady=10, padx=10)
        btn_padding_x = 7
        btn_padding_y = 12
        Button(manipulate_button_frame, default_button_options, text='Delete NaN Rows', command=self.remove_nans) \
            .grid(row=0, column=0, padx=btn_padding_x, pady=btn_padding_y)
        Button(manipulate_button_frame, default_button_options, text='Delete Column', command=self.delete_cols) \
            .grid(row=0, column=1, padx=btn_padding_x, pady=btn_padding_y)
        Button(manipulate_button_frame, default_button_options, text='Change Datatype', command=self.change_dtype) \
            .grid(row=0, column=2, padx=btn_padding_x, pady=btn_padding_y)
        Button(manipulate_button_frame, default_button_options, text='One-Hot Encode', command=self.one_hot_encode) \
            .grid(row=0, column=3, padx=btn_padding_x, pady=btn_padding_y)
        Button(manipulate_button_frame, default_button_options, text='Impute', command=self.impute) \
            .grid(row=0, column=4, padx=btn_padding_x, pady=btn_padding_y)
        Button(manipulate_button_frame, default_button_options, text='Correlation', command=self.correlation_map) \
            .grid(row=0, column=5, padx=btn_padding_x, pady=btn_padding_y)
        Button(manipulate_button_frame, default_button_options, text='Train SVM', command=self.model_svm) \
            .grid(row=0, column=6, padx=btn_padding_x, pady=btn_padding_y)
        Button(manipulate_button_frame, default_button_options, text='Train KNN', command=self.model_knn) \
            .grid(row=0, column=7, padx=btn_padding_x, pady=btn_padding_y)
        Button(manipulate_button_frame, default_button_options, text='Train DecisionTree', command=self.model_decision_tree) \
            .grid(row=1, column=0, padx=btn_padding_x, pady=btn_padding_y)
        Button(manipulate_button_frame, default_button_options, text='Compare Models') \
            .grid(row=1, column=1, padx=btn_padding_x, pady=btn_padding_y)
        Button(manipulate_button_frame, default_button_options, text='Predict', command=self.predict) \
            .grid(row=1, column=2, padx=btn_padding_x, pady=btn_padding_y)
        Button(manipulate_button_frame, default_button_options, text='Label Encoding') \
            .grid(row=1, column=3, padx=btn_padding_x, pady=btn_padding_y)
        Button(manipulate_button_frame, default_button_options, text='Dummy') \
            .grid(row=1, column=4, padx=btn_padding_x, pady=btn_padding_y)
        Button(manipulate_button_frame, default_button_options, text='Redo', command=self.redo) \
            .grid(row=1, column=5, padx=btn_padding_x, pady=btn_padding_y)
        Button(manipulate_button_frame, default_button_options, text='Undo', command=self.undo) \
            .grid(row=1, column=6, padx=btn_padding_x, pady=btn_padding_y)
        Button(manipulate_button_frame, default_button_options, text='Original Dataset', command=self.show_original) \
            .grid(row=1, column=7, padx=btn_padding_x, pady=btn_padding_y)

        manipulate_button_frame.place(y=screen_height * 4 / 5, height=screen_height * 1 / 5, width=screen_width)

    def predict(self):
        if self.target_col.get() == 'None':
            messagebox.showerror('Target Missing', 'Please choose a target col from above')
            return

        self.predict_window = Toplevel(self)
        predictors = self.df.drop(self.target_col.get(), axis=1)
        predictors_var = [StringVar() for _ in predictors]
        container = Frame(self.predict_window)

        for i, column_name in enumerate(predictors):
            Label(container, text=column_name).grid(row=i, column=0)
            Entry(container, textvariable=predictors_var[i]).grid(row=i, column=1)
            container.pack(padx=20)

        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)
        # np.self.df[predictor].dtype(variable.get())

        def predict_helper():
            var_list = []
            for variable, predictor in zip(predictors_var, predictors):
                var = variable.get()
                data_type = self.df[predictor].dtype
                var = np.dtype(str(data_type)).type(var)
                var_list.append(var)

            np_arr = np.array(var_list).reshape(1, -1)
            print(self.knn.predict(np_arr))

        Button(self.predict_window, text='Predict', command=predict_helper).pack(pady=10)

    def model_svm(self):
        if self.target_col.get() == 'None':
            messagebox.showerror('Target Missing', 'Please choose a target col from above')
            return

        svm_thread = threading.Thread(target=self.model_svm_thread_func, args=())
        svm_thread.start()

    def model_svm_thread_func(self):
        labels = self.df[self.target_col.get()]
        predictors = self.df.drop(self.target_col.get(), axis=1)

        X_train, X_test, y_train, y_test = train_test_split(predictors, labels, train_size=0.3, random_state=4)
        from sklearn.svm import SVR
        from sklearn.metrics import mean_squared_error

        self.svm = SVR()
        self.svm.fit(X_train, y_train)

        test_predictions = self.svm.predict(X_test)
        mse = mean_squared_error(y_test, test_predictions)
        messagebox.showinfo('Training Completed', 'Model trained with {} Mean Squared Error'.format(mse))

    def model_knn(self):
        if self.target_col.get() == 'None':
            messagebox.showerror('Target Missing', 'Please choose a target col from above')
            return

        knn_thread = threading.Thread(target=self.model_knn_thread_func, args=())
        knn_thread.start()

    def model_knn_thread_func(self):
        labels = self.df[self.target_col.get()]
        predictors = self.df.drop(self.target_col.get(), axis=1)

        X_train, X_test, y_train, y_test = train_test_split(predictors, labels, train_size=0.3, random_state=4)
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.metrics import mean_squared_error

        self.knn = KNeighborsRegressor(n_neighbors=10)
        self.knn.fit(X_train, y_train)

        test_predictions = self.knn.predict(X_test)
        mse = mean_squared_error(y_test, test_predictions)
        messagebox.showinfo('Training Completed', 'Model trained with {} Mean Squared Error'.format(mse))

    def model_decision_tree(self):
        if self.target_col.get() == 'None':
            messagebox.showerror('Target Missing', 'Please choose a target col from above')
            return

        decision_tree_thread = threading.Thread(target=self.model_decision_tree_thread_func, args=())
        decision_tree_thread.start()

    def model_decision_tree_thread_func(self):
        labels = self.df[self.target_col.get()]
        predictors = self.df.drop(self.target_col.get(), axis=1)

        X_train, X_test, y_train, y_test = train_test_split(predictors, labels, train_size=0.3, random_state=4)
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_squared_error

        self.dtr = DecisionTreeRegressor(max_depth=10)
        self.dtr.fit(X_train, y_train)

        test_predictions = self.dtr.predict(X_test)
        mse = mean_squared_error(y_test, test_predictions)
        messagebox.showinfo('Training Completed', 'Model trained with {} Mean Squared Error'.format(mse))

    def correlation_map(self):
        corr = self.df.corr()
        for col in corr.columns:
            total_row = corr.shape[0]
            total_na = corr[col].isna().sum()
            if total_na == total_row:
                corr.drop(col, inplace=True, axis=1)
                corr.drop(col, inplace=True, axis=0)
        sns.heatmap(corr, center=0, cmap='bwr')
        plt.show()

    def one_hot_encode(self):
        category_cols = self.df.select_dtypes(include='O').keys()

        if len(category_cols) == 0:
            messagebox.showinfo('Info!', 'No Categorical Column Found')
            return

        self.one_hot_encode_window = Toplevel(self)
        self.one_hot_encode_window.title('One Hot Encoder')
        width = 360
        height = min(len(category_cols) * 30 + 60, 460)
        self.one_hot_encode_window.geometry('{}x{}'.format(width, height))
        self.one_hot_encode_window.resizable(False, False)

        top_list_frame = LabelFrame(self.one_hot_encode_window, width=width)
        scroll_bar = Scrollbar(top_list_frame)
        scroll_bar.pack(side=RIGHT, fill=Y)

        check_list = Text(top_list_frame, height=height-50)
        check_list.pack(fill=X)

        self.left_btn_list = [Button for _ in range(len(category_cols))]
        self.right_btn_list = [Button for _ in range(len(category_cols))]

        for i in range(len(category_cols)):
            container = LabelFrame(check_list)
            self.left_btn_list[i] = Button(container, another_button_options, text='Select', cursor='hand2',
                                           command=partial(self.change_state, i, 0, category_cols))
            self.left_btn_list[i].pack(side=LEFT)
            Label(container, text=category_cols[i].upper(), width=24).pack(side=LEFT)
            self.right_btn_list[i] = Button(container, another_button_options, text='Remove', cursor='hand2',
                                            state=DISABLED, command=partial(self.change_state, i, 1, category_cols))
            self.right_btn_list[i].pack(side=LEFT)
            container.pack(fill=X)

            check_list.window_create('end', window=container)
            check_list.insert('end', '\n')

        check_list.config(yscrollcommand=scroll_bar.set)
        scroll_bar.config(command=check_list.yview)
        check_list.configure(state='disabled')

        top_list_frame.pack()

        bottom_button_frame = LabelFrame(self.one_hot_encode_window)
        Button(bottom_button_frame, default_button_options, text='Encode', bg='red3',
               command=self.one_hot_encode_helper).pack(side=LEFT)
        bottom_button_frame.place(relx=.5, y=height-25, anchor=CENTER)

    def one_hot_encode_helper(self):
        self.one_hot_encode_window.destroy()
        self.save_state()
        encoder = preprocessing.OneHotEncoder()
        for i, col in enumerate(self.selected_columns):
            encoded_series = encoder.fit_transform(self.df[col].values.reshape(-1, 1)).toarray().reshape(-1)
            self.df[col] = pd.Series(encoded_series)
        self.update_table()
        self.selected_columns = []

    def impute(self):
        numerical_data_types = ['int64', 'float64']
        numerical_cols = []
        for col in self.columns:
            if self.df[col].dtype in numerical_data_types:
                numerical_cols.append(col)
        nan_cols = [col for col in self.df.columns if self.df[col].isnull().any()]

        if len(nan_cols) == 0:
            messagebox.showinfo('Info!', 'No Numerical Column has NaN value')
            return

        options = ['none', 'mean', 'median', 'most_frequent']
        clicked_arr = [StringVar() for _ in range(len(nan_cols))]

        self.impute_window = Toplevel(self)
        self.impute_window.title('Imputer')
        self.impute_window.configure(background='bisque')
        width = 380
        height = min(len(nan_cols) * 30 + 80, 460)
        self.impute_window.geometry('{}x{}'.format(width, height))
        self.impute_window.resizable(False, False)

        for var in clicked_arr:
            var.set(options[0])

        top_list_frame = LabelFrame(self.impute_window, border=0)

        scroll_bar = Scrollbar(top_list_frame)
        scroll_bar.pack(side=RIGHT, fill=Y)

        check_list = Text(top_list_frame, height=height-50)
        check_list.pack(fill=X)

        container = LabelFrame(check_list)
        Label(container, text='COLUMN', width=34).pack(side=LEFT)
        Label(container, text='STRATEGY', width=16).pack(side=LEFT)
        container.pack(fill=X)
        check_list.window_create('end', window=container)
        check_list.insert('end', '\n')

        for i in range(len(nan_cols)):
            container = LabelFrame(check_list, border=0)
            cur_col = nan_cols[i]
            Label(container, text=cur_col.upper(), width=29, font=default_text_font_bold).pack(side=LEFT)
            option_menu = OptionMenu(container, clicked_arr[i], *options)
            option_menu.configure(width=10, font=default_text_font)
            option_menu.pack(side=LEFT, fill=X)
            container.pack(fill=X)

            check_list.window_create('end', window=container)
            check_list.insert('end', '\n')

        check_list.config(yscrollcommand=scroll_bar.set)
        scroll_bar.config(command=check_list.yview)
        check_list.configure(state='disabled')
        top_list_frame.pack()
        Button(self.impute_window, another_button_options, text='Impute',
               command=partial(self.impute_helper, clicked_arr, nan_cols)).place(relx=.5, y=height-25, anchor=CENTER)

    def impute_helper(self, arr_list, col_list):
        self.impute_window.destroy()

        for i, col in enumerate(col_list):
            strategy = arr_list[i].get()
            if strategy != 'none':
                imputer = SimpleImputer(strategy=strategy)
                self.df[col] = imputer.fit_transform(self.df[col].values.reshape(-1, 1))

        self.save_state()
        self.update_table()

    def change_dtype(self):
        if len(self.columns) == 0:
            messagebox.showinfo('Info!', 'No Column to change datatype of')
            return

        data_types = ['object', 'int64', 'float64', 'bool', 'datetime64', 'timedelta[ns]', 'category']

        clicked_arr = [StringVar() for _ in range(len(self.columns))]
        for i in range(len(self.columns)):
            clicked_arr[i].set(self.df[self.columns[i]].dtype)

        self.change_datatype_window = Toplevel(self)
        self.change_datatype_window.title('Change Datatype')
        width = 380
        height = min(len(self.columns) * 30 + 80, 460)
        self.change_datatype_window.geometry('{}x{}'.format(width, height))
        self.change_datatype_window.resizable(False, False)

        top_list_frame = LabelFrame(self.change_datatype_window)

        scroll_bar = Scrollbar(top_list_frame)
        scroll_bar.pack(side=RIGHT, fill=Y)

        check_list = Text(top_list_frame)
        check_list.pack(fill=X)

        container = LabelFrame(check_list, border=0)
        Label(container, text='Column', width=36).pack(side=LEFT)
        Label(container, text='Change to Dtype', width=14).pack(side=LEFT)
        container.pack(fill=X)
        check_list.window_create('end', window=container)
        check_list.insert('end', '\n')

        for i in range(len(self.columns)):
            container = LabelFrame(check_list, border=0)
            cur_col = self.columns[i]
            Label(container, text=cur_col.upper(), width=29, font=default_text_font_bold).pack(side=LEFT)
            option_menu = OptionMenu(container, clicked_arr[i], *data_types)
            option_menu.configure(width=10, font=default_text_font)
            option_menu.pack(side=LEFT, fill=X)
            container.pack(fill=X)

            check_list.window_create('end', window=container)
            check_list.insert('end', '\n')

        check_list.config(yscrollcommand=scroll_bar.set)
        scroll_bar.config(command=check_list.yview)
        check_list.configure(state='disabled')
        top_list_frame.pack()
        Button(self.change_datatype_window, another_button_options, text='Change',
               command=partial(self.change_dtype_helper, clicked_arr)).place(relx=.5, y=height-25, anchor=CENTER)

    def change_dtype_helper(self, arr_list):
        self.change_datatype_window.destroy()
        self.save_state()
        for i, column in enumerate(self.columns):
            cur_col_dtype = self.df[self.columns[i]].dtype
            if arr_list[i].get() != cur_col_dtype:
                try:
                    self.df[self.columns[i]] = self.df[self.columns[i]].astype(arr_list[i].get())
                except Exception as e:
                    messagebox.showinfo('Error', 'While Changing type of {}\n{}\nException Occurred\n'
                                                 'Other Columns Type has been changed'
                                        .format(self.columns[i].upper(), e))
        self.update_table()

    def update_table(self):
        self.columns = self.df.columns
        self.show_dataset()

    def save_state(self):
        # first condition makes sure that the current dataframe has some changes made to it
        # second condition takes care of situation when we have not select any columns to manipulate
        if (len(self.undo_stack) and self.undo_stack[-1].equals(self.df)) or (len(self.undo_stack) and self.df.equals(self.original_df)):
            return

        self.undo_stack.append(self.df.copy())

    def redo(self):
        if len(self.redo_stack):
            self.undo_stack.append(self.df)
            self.df = self.redo_stack[-1]
            self.redo_stack.pop()
            self.update_table()

    def undo(self):
        if len(self.undo_stack):
            self.redo_stack.append(self.df)
            self.df = self.undo_stack[-1]
            self.undo_stack.pop()
            self.update_table()

    def show_original(self):
        # condition to check if user is trying to see the original while current df is original
        if not self.df.equals(self.original_df):
            self.undo_stack.append(self.df)
            self.df = self.original_df
            self.update_table()

    def clear_all(self):
        for item in self.tree_view.get_children():
            self.tree_view.delete(item)

    def show_dataset(self):
        self.clear_all()
        num_rows, num_cols = self.df.shape
        self.row_col_display['text'] = '{}x{}'.format(num_rows, num_cols)
        self.tree_view["column"] = list(self.df.columns)
        self.tree_view["show"] = "headings"
        for column in self.tree_view["columns"]:
            self.tree_view.heading(column, text=column)  # let the column heading = column name
        df_rows = self.df.to_numpy().tolist()  # turns the dataframe into a list of lists
        for row in df_rows:
            self.tree_view.insert("", "end", values=row)

    def delete_cols(self):
        self.selected_columns = []

        self.get_column_window = Toplevel(self)
        self.get_column_window.title('Select Columns')
        self.get_column_window.geometry('380x460')

        total_columns = len(self.columns)

        top_list_frame = LabelFrame(self.get_column_window, height=400, width=380)

        scroll_bar = Scrollbar(top_list_frame)
        scroll_bar.pack(side=RIGHT, fill=Y)

        check_list = Text(top_list_frame)
        check_list.pack(fill=X)

        self.left_btn_list = [None for i in range(total_columns)]
        self.right_btn_list = [None for i in range(total_columns)]

        for i in range(total_columns):
            container = LabelFrame(check_list)
            self.left_btn_list[i] = Button(container, another_button_options, text='Select', width=10, cursor='hand2',
                                           command=partial(self.change_state, i, 0, self.columns))
            self.left_btn_list[i].pack(side=LEFT)
            Label(container, text=self.columns[i].upper(), width=24).pack(side=LEFT)
            self.right_btn_list[i] = Button(container, another_button_options, text='Remove', width=10, cursor='hand2',
                                            state=DISABLED, command=partial(self.change_state, i, 1, self.columns))
            self.right_btn_list[i].pack(side=LEFT)
            container.pack(fill=X)

            check_list.window_create('end', window=container)
            check_list.insert('end', '\n')

        check_list.config(yscrollcommand=scroll_bar.set)
        scroll_bar.config(command=check_list.yview)
        check_list.configure(state='disabled')

        top_list_frame.pack()

        bottom_button_frame = LabelFrame(self.get_column_window)
        Button(bottom_button_frame, default_button_options, text='Delete', bg='red3',
               command=partial(self.delete_cols_helper)).pack(side=LEFT)
        bottom_button_frame.pack(pady=14)

    def change_state(self, idx, right, cols):
        if right == 1:
            self.selected_columns.remove(cols[idx])
            self.left_btn_list[idx]['state'] = NORMAL
            self.right_btn_list[idx]['state'] = DISABLED
        else:
            self.selected_columns.append(cols[idx])
            self.left_btn_list[idx]['state'] = DISABLED
            self.right_btn_list[idx]['state'] = NORMAL

    def delete_cols_helper(self):
        self.save_state()
        self.df.drop(self.selected_columns, inplace=True, axis=1)
        self.selected_columns = []
        self.get_column_window.destroy()
        self.update_table()

    def remove_nans(self):
        self.save_state()
        self.df.dropna(inplace=True)
        self.update_table()
