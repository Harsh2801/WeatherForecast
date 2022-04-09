from MainWindow import *
from IntroWindow import *

welcome_win = WelcomeWindow()
welcome_win.mainloop()

try:
    main_win = MainWindow()
    main_win.mainloop()
except TypeError:
    pass
