#!/usr/bin/env python3
"""
A more responsive version of equalize_it.py for Linux that uses two
histogram plots on a tkinter canvas to update just the CLAHE histogram.
Uses a matplotlib TkAgg backend and tkinter.
Fully functional only on Linux systems.

USAGE Example command lines, from within the image-processor-main folder:
python3 -m equalize_tk --help
python3 -m equalize_tk --about
python3 -m equalize_tk --input images/sample2.jpg

Quit program with Esc or Q key; may need to first select a window other
than Histograms. Or quit from command line with Ctrl-C.

Requires Python3.7 or later and the packages opencv-python and numpy.
Developed in Python 3.8-3.9.
"""
# Copyright (C) 2022-2023 C.S. Echt, under GNU General Public License

# Standard library imports.
import sys
import threading
from pathlib import Path

# Third party imports.
try:
    import cv2
    import matplotlib.backends.backend_tkagg as backend
    import tkinter as tk
    from matplotlib import pyplot as plt
except (ImportError, ModuleNotFoundError) as import_err:
    print('*** OpenCV, Numpy, Matplotlib or tkinter (tk/tcl) was not found or needs an update:\n\n'
          'To install: from the current folder, run this command'
          ' for the Python package installer (PIP):\n'
          '   python3 -m pip install -r requirements.txt\n\n'
          'Alternative command formats (system dependent):\n'
          '   py -m pip install -r requirements.txt (Windows)\n'
          '   pip install -r requirements.txt\n\n'
          'A package may already be installed, but needs an update;\n'
          '   this may be the case when the error message (below) is a bit cryptic\n'
          '   Example update command:\n'
          '   python3 -m pip install -U matplotlib\n'
          'On Linux, if tkinter is the problem, then you may need to run:\n'
          '   sudo apt-get install python3-tk\n'
          '   See also: https://tkdocs.com/tutorial/install.html \n\n'
          f'Error message:\n{import_err}')
    sys.exit(1)

# Local application imports
from contour_modules import (vcheck, manage, utils, constants as const)


# noinspection PyUnresolvedReferences
class ProcessImage:
    __slots__ = ('clahe_img', 'clahe_mean', 'clahe_sd', 'clip_limit',
                 'gray_img', 'input_img', 'input_mean',
                 'input_sd', 'settings_txt',
                 'settings_win', 'tile_size',
                 'fig', 'ax1', 'ax2')

    def __init__(self):

        # The np.ndarray arrays for images to be processed.
        self.input_img = None
        self.gray_img = None
        self.clahe_img = None

        if utils.MY_OS == 'lin':
            # Matplotlib plotting with live updates.
            plt.style.use(('bmh', 'fast'))
            self.fig, (self.ax1, self.ax2) = plt.subplots(
                nrows=2,
                num='Histograms',  # Provide a window title to replace 'Figure 1'.
                sharex='all',
                sharey='all',
                clear=True
            )
            # Note that plt.ion() needs to be called
            # AFTER subplots(), otherwise
            #   a "Segmentation fault (core dumped)" error is raised.
            # plt.ion() is used with fig.canvas.start_event_loop(0.1);
            #   it is not needed if fig.canvas.draw_idle() is used.
            # matplotlib.get_backend()
            plt.ion()

        # Image processing parameters amd metrics.
        self.clip_limit = 2.0  # Default trackbar value.
        self.tile_size = (8, 8)  # Default trackbar value.
        self.input_sd = 0
        self.input_mean = 0
        self.clahe_sd = 0
        self.clahe_mean = 0

        self.settings_txt = ''
        self.settings_win = ''

        # Note: This order of calls in Linus and Windows is
        #  needed for histograms to display at start; it's an
        #  event issue (where setup_trackbars() triggers an event
        #  that prompts drawing of histogram window).
        self.manage_input()
        if utils.MY_OS == 'lin':
            self.setup_canvas_window()
            self.show_input_histogram()
        self.setup_trackbars()

    def manage_input(self):
        """
        Reads input images, creates grayscale image and its flattened
        array, adjusts displayed image size, displays input and grayscale
        side-by-side in one window.

        Returns: None
        """

        # utils.args_handler() has verified image path, so read from it.
        self.input_img = cv2.imread(arguments['input'])
        self.gray_img = cv2.imread(arguments['input'], cv2.IMREAD_GRAYSCALE)

        cv2.namedWindow(const.WIN_NAME['input+gray'],
                        flags=cv2.WINDOW_GUI_NORMAL)

        # NOTE: In Windows, w/o scaling, window may be expanded to full screen
        #   if system is set to remember window positions.
        if utils.MY_OS == 'win':
            cv2.resizeWindow(const.WIN_NAME['input+gray'], 1000, 500)

        # Need to scale only images to display, not those to be processed.
        #   Default --scale arg is 1.0, so no scaling when option not used.
        input_img_scaled = utils.scale_img(self.input_img, arguments['scale'])
        gray_img_scaled = utils.scale_img(self.gray_img, arguments['scale'])
        side_by_side = cv2.hconcat(
            [input_img_scaled, cv2.cvtColor(gray_img_scaled, cv2.COLOR_GRAY2RGB)])
        cv2.imshow(const.WIN_NAME['input+gray'], side_by_side)

    @staticmethod
    def setup_canvas_window() -> None:
        """
        A tkinter window for the Matplotlib figure canvas.
        """

        # histogram_window is the Tk mainloop defined in if __name__ == "__main__".
        canvas_window.title('Histograms')
        canvas_window.resizable(False, False)

        canvas_window.bind_all('<Escape>', utils.quit_keys)
        canvas_window.bind('<Control-q>', utils.quit_keys)

        canvas = backend.FigureCanvasTkAgg(plt.gcf(), canvas_window)
        toolbar = backend.NavigationToolbar2Tk(canvas, canvas_window)

        # Need to remove navigation button.
        # Source: https://stackoverflow.com/questions/59155873/
        #   how-to-remove-toolbar-button-from-navigationtoolbar2tk-figurecanvastkagg
        # Remove all tools from toolbar because the Histograms window is
        #   non-responsive while in event_loop.
        for child, widget in toolbar.children.items():
            widget.pack_forget()

        # Now display remaining widgets in histogram_window.
        # NOTE: toolbar must be gridded BEFORE canvas to prevent
        #   FigureCanvasTkAgg from preempting window geometry with its pack().
        toolbar.grid(row=1, column=0,
                     padx=5, pady=(0, 5),  # Put a border around toolbar.
                     sticky=tk.NSEW,
                     )
        canvas.get_tk_widget().grid(row=0, column=0,
                                    ipady=10, ipadx=10,
                                    padx=5, pady=5,  # Put a border around plot.
                                    sticky=tk.NSEW,
                                    )

    def setup_trackbars(self) -> None:
        """
        All trackbars that go in a separate window of image processing
        settings.

        Returns: None
        """

        if utils.MY_OS in 'lin, win':
            self.settings_win = "cv2.createCLAHE settings (dbl-click text to save)"
        else:  # is macOS
            self.settings_win = "cv2.createCLAHE settings (rt-click text to save)"

        # Move the control window away from the processing windows.
        # Place window at right edge of screen by using an excessive x-coordinate.
        if utils.MY_OS == 'lin':
            cv2.namedWindow(self.settings_win, flags=cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self.settings_win, 800, 35)
        elif utils.MY_OS == 'dar':
            cv2.namedWindow(self.settings_win)
            cv2.moveWindow(self.settings_win, 600, 300)
        else:  # is Windows
            # Need to compensate for WINDOW_AUTOSIZE not working in Windows10.
            cv2.namedWindow(self.settings_win, flags=cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(self.settings_win, 900, 500)

        cv2.setMouseCallback(self.settings_win,
                             self.save_with_click)

        if utils.MY_OS == 'lin':
            clip_tb_name = 'Clip limit\n10X'
            tile_tb_name = 'Tile size (N, N)\n'
        elif utils.MY_OS == 'win':  # is WindowsS, limited to 10 characters
            clip_tb_name = 'Clip, 10X'
            tile_tb_name = 'Tile size'
        else:  # is macOS
            clip_tb_name = 'Clip limit, (10X)'
            tile_tb_name = 'Tile size, (N, N)'

        cv2.createTrackbar(clip_tb_name,
                           self.settings_win,
                           20,
                           50,
                           self.clip_selector)
        cv2.setTrackbarMin(clip_tb_name,
                           self.settings_win,
                           1)

        cv2.createTrackbar(tile_tb_name,
                           self.settings_win,
                           8,
                           200,
                           self.tile_selector)
        cv2.setTrackbarMin(tile_tb_name,
                           self.settings_win,
                           1)

    def save_with_click(self, event, *args):
        """
        Double-click on the namedWindow calls module that saves the image
        and settings.
        Calls utils.save_settings_and_img.
        Called by cv2.setMouseCallback event.

        Args:
            event: The implicit mouse event.
            *args: Return values from setMouseCallback(); not used here.

        Returns: *event* as a formality.

        """
        implicit_list_of_6_elements = args
        if utils.MY_OS in 'lin, win':
            mouse_event = cv2.EVENT_LBUTTONDBLCLK
        else:
            mouse_event = cv2.EVENT_RBUTTONDOWN

        if event == mouse_event:
            utils.save_settings_and_img(self.clahe_img,
                                        self.settings_txt,
                                        f'{Path(__file__).stem}')
        return event

    def clip_selector(self, c_val) -> None:
        """
        The "CLAHE clip limit (10X)" trackbar handler. Limits tile_size
        to greater than zero.

        Args:
            c_val: The integer value passed from trackbar.
        Returns: None
        """

        self.clip_limit = c_val / 10

        self.apply_clahe()

    def tile_selector(self, t_val) -> None:
        """
        The "CLAHE tile size" trackbar handler. Limits tile_size
        to greater than zero.

        Args:
            t_val: The integer value passed from trackbar.
        Returns: None
        """

        self.tile_size = t_val, t_val

        self.apply_clahe()

    def apply_clahe(self) -> None:
        """
        Applies CLAHE adjustments to image and calculates pixel values
        for reporting.

        Returns: None
        """

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=self.tile_size,
                                )
        self.clahe_img = clahe.apply(self.gray_img)

        self.input_sd = int(self.gray_img.std())
        self.input_mean = int(self.gray_img.mean())
        self.clahe_sd = int(self.clahe_img.std())
        self.clahe_mean = int(self.clahe_img.mean())

        if utils.MY_OS == 'lin':
            self.show_clahe_histogram()
        self.show_settings()

        cv2.namedWindow(const.WIN_NAME['clahe'],
                        flags=cv2.WINDOW_GUI_NORMAL)
        clahe_img_scaled = utils.scale_img(self.clahe_img, arguments['scale'])
        cv2.imshow(const.WIN_NAME['clahe'], clahe_img_scaled)

    def show_input_histogram(self) -> None:
        """
        Allows a one-time rendering of the input histogram, thus
        providing a faster response for updating the histogram Figure
        with CLAHE Trackbar changes.
        Called from __init__().

        Returns: None
        """

        # hist() returns tuple of (counts(n), bins(edges), patches(artists)
        # histtype='step' draws a line, 'stepfilled' fills under the line;
        #   both are patches.Polygon artists that provide faster rendering
        #   than the default 'bar', which is a BarContainer object of
        #   Rectangle artists.
        # Need to match these parameters with those for ax2.hist().
        self.ax1.hist(self.gray_img.ravel(),
                      bins=255,
                      range=[0, 256],
                      color='blue',
                      alpha=0.4,
                      histtype='stepfilled',
                      )
        self.ax1.set_ylabel("Pixel count")
        self.ax1.set_title('Input (grayscale)')

    def show_clahe_histogram(self) -> None:
        """
        Updates CLAHE adjusted histogram plot with Matplotlib from
        trackbar changes. Called from apply_clahe().

        Returns: None
        """

        # Need to clear prior histograms before drawing new ones.
        self.ax2.clear()

        self.ax2.hist(self.clahe_img.ravel(),
                      bins=255,
                      range=[0, 256],
                      color='orange',
                      histtype='stepfilled',  # 'step' draws a line.
                      # linewidth=1.2
                      )
        self.ax2.set_title('CLAHE adjusted')
        self.ax2.set_xlabel("Pixel value")
        self.ax2.set_ylabel("Pixel count")

        # From: https://stackoverflow.com/questions/28269157/
        #  plotting-in-a-non-blocking-way-with-matplotlib
        # and, https://github.com/matplotlib/matplotlib/issues/11131
        # Note that start_event_loop is needed for live updates of clahe histograms.
        self.fig.canvas.start_event_loop(0.1)

    def show_settings(self) -> None:
        """
        Display name of file and processing parameters in contour_tb_win
        window. Displays real-time parameter changes.
        Calls module utils.text_array() in contour_modules directory.

        Returns: None
        """

        the_text = (
            f'Input image: {arguments["input"]}\n'
            f'Input grayscale pixel value: mean {self.input_mean},'
            f' stdev {self.input_sd}\n'
            f'cv2.createCLAHE cliplimit={self.clip_limit}, tileGridSize{self.tile_size}\n'
            f'CLAHE grayscale pixel value: mean {self.clahe_mean},'
            f' stdev {self.clahe_sd}'
        )

        # Put text into contoured_txt for printing and saving to file.
        self.settings_txt = the_text

        # Need to set the dimensions of the settings area to fit all text.
        #   Font style parameters are set in constants.py module.
        settings_img = utils.text_array((150, 500), the_text)

        cv2.imshow(self.settings_win, settings_img)


if __name__ == "__main__":
    # Program exits here if system platform or Python version check fails.
    utils.check_platform()
    vcheck.minversion('3.7')

    # All checks are good, so grab as a 'global' the dictionary of
    #   command line argument values.
    arguments = manage.arguments()

    # Need to not set up tk canvas to display Histograms b/c
    #  generates a fatal memory allocation error. It has something
    #  to do with the start_event_loop function.
    if utils.MY_OS in 'dar, win':
        PI = ProcessImage()
        print(f'{Path(__file__).name} is now running...')
        print('Currently, histograms do not plot in Windows or macOS.\n'
              'Working on it though...')

        # Set infinite loop with sigint handler to monitor "quit" keystrokes.
        utils.quit_keys()
    else:  # is Linux
        # Run the Matplotlib histogram plots in a tkinter window.
        canvas_window = tk.Tk()

        PI = ProcessImage()
        print(f'{Path(__file__).name} is now running...')

        # Set infinite loop with sigint handler to monitor "quit" keystrokes.
        quit_thread = threading.Thread(
            target=utils.quit_keys(), daemon=True)
        quit_thread.start()

        canvas_window.mainloop()
