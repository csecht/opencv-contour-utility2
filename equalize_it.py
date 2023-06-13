#!/usr/bin/env python3
"""
Use a tkinter GUI to explore OpenCV's CLAHE image processing parameters.
Parameter values are adjusted with slide bars.

USAGE Example command lines, from within the image-processor-main folder:
python3 -m equalize_it --help
python3 -m equalize_it --about
python3 -m equalize_it --input images/sample2.jpg
python3 -m equalize_it -i images/sample2.jpg -s 0.6

Windows systems may need to substitute 'python3' with 'py' or 'python'.

Quit program with Esc key, Ctrl-Q key, the close window icon of the
settings windows, or from command line with Ctrl-C.
Save settings and the CLAHE adjusted image with Save button.

Requires Python3.7 or later and the packages opencv-python and numpy.
See this distribution's requirements.txt file for details.
Developed in Python 3.8-3.9.
"""

# Copyright (C) 2022-2023 C.S. Echt, under GNU General Public License
# pylint: disable=use-dict-literal, no-member

# Standard library imports.
import sys

from pathlib import Path

# Local application imports.
from contour_modules import (vcheck,
                             utils,
                             manage,
                             constants as const,
                             )

# Third party imports.
# tkinter(Tk/Tcl) is included with most Python3 distributions,
#  but may sometimes need to be regarded as third-party.
try:
    import cv2
    import tkinter as tk
    from matplotlib import pyplot as plt
    from tkinter import ttk

except (ImportError, ModuleNotFoundError) as import_err:
    sys.exit(
        '*** One or more required Python packages were not found'
        ' or need an update:\nOpenCV-Python, NumPy, Matplotlib, tkinter (Tk/Tcl).\n\n'
        'To install: from the current folder, run this command'
        ' for the Python package installer (PIP):\n'
        '   python3 -m pip install -r requirements.txt\n\n'
        'Alternative command formats (system dependent):\n'
        '   py -m pip install -r requirements.txt (Windows)\n'
        '   pip install -r requirements.txt\n\n'
        'You my also install directly using, for example, this command,'
        ' for the Python package installer (PIP):\n'
        '   python3 -m pip install opencv-python\n\n'
        'A package may already be installed, but needs an update;\n'
        '   this may be the case when the error message (below) is a bit cryptic\n'
        '   Example update command:\n'
        '   python3 -m pip install -U numpy\n\n'
        'On Linux, if tkinter is the problem, then you may need:\n'
        '   sudo apt-get install python3-tk\n\n'
        'See also: https://numpy.org/install/\n'
        '  https://tkdocs.com/tutorial/install.html\n'
        '  https://docs.opencv.org/4.6.0/d5/de5/tutorial_py_setup_in_windows.html\n\n'
        f'Error message:\n{import_err}')


class ProcessImage(tk.Tk):
    """
    Matplotlib and tkinter methods for applying cv2.createCLAHE to an
    image from the input file.

    Class methods:
    apply_clahe()
    """

    __slots__ = (
        'ax1', 'ax2', 'clahe_img', 'clahe_mean', 'clahe_sd', 'curr_contrast_std', 'fig',
        'img_label', 'img_window', 'input_contrast_std', 'input_mean', 'input_sd', 'slider_val',
        'tkimg', 'tk', 'plt'
    )

    def __init__(self):
        super().__init__()

        # Matplotlib plotting with live updates.
        plt.ion()
        plt.style.use(('bmh', 'fast'))

        # Note: The matching selector widgets for these control variables
        #  are in ImageViewer __init__. Don't really need a dict for
        #  two var, but it's there to maintain the naming convention in
        #  contour_it().
        self.slider_val = {
            'clip_limit': tk.DoubleVar(),
            'tile_size': tk.IntVar()
        }

        self.input_contrast_std = tk.DoubleVar()
        self.curr_contrast_std = tk.DoubleVar()

        # Arrays of images to be processed. When used within a method,
        #  the purpose of self.tkimg[*] as an instance attribute is to
        #  retain the attribute reference and thus prevent garbage collection.
        #  Dict values will be defined for panels of PIL ImageTk.PhotoImage
        #  with Label images displayed in their respective img_window Toplevel.
        self.tkimg = {
            'input': tk.PhotoImage(),
            'gray': tk.PhotoImage(),
            'clahe': tk.PhotoImage(),
        }

        # Dict values that are defined in ImageViewer.setup_image_windows().
        self.img_window = {}
        self.img_label = {}

        self.input_sd = 0  # int(GRAY_IMG.std())
        self.input_mean = 0  # int(GRAY_IMG.mean())
        self.clahe_sd = 0  # int(self.clahe_img.std())
        self.clahe_mean = 0  # int(self.clahe_img.mean())
        self.clahe_img = None  # a np.ndarray

    def apply_clahe(self) -> None:
        """
        Applies CLAHE adjustments to image and calculates pixel values
        for reporting.

        Returns: None
        """

        clahe = cv2.createCLAHE(clipLimit=self.slider_val['clip_limit'].get(),
                                tileGridSize=(self.slider_val['tile_size'].get(),
                                              self.slider_val['tile_size'].get()),
                                )
        self.clahe_img = clahe.apply(GRAY_IMG)

        self.input_sd = int(GRAY_IMG.std())
        self.input_mean = int(GRAY_IMG.mean())
        self.clahe_sd = int(self.clahe_img.std())
        self.clahe_mean = int(self.clahe_img.mean())

        self.tkimg['clahe'] = manage.tk_image(self.clahe_img)
        self.img_label['clahe'].configure(image=self.tkimg['clahe'])


class ImageViewer(ProcessImage):
    """
    Methods to adjust CLAHE parameters, and display and update the
    resulting images.

    Class Methods:
    setup_image_windows -> no_exit_on_x
    setup_report_window
    config_sliders
    config_buttons -> save_settings
    grid_widgets
    display_images
    set_clahe_defaults
    show_histograms
    report_clahe
    process_all
    """

    __slots__ = (
        'clahe_report_frame', 'clahe_selectors_frame', 'clahe_settings_txt',
        'flat_gray', 'img_label', 'img_window', 'reset_btn', 'save_btn',
        'separator', 'slider'
    )

    def __init__(self):
        super().__init__()

        self.clahe_report_frame = tk.Frame()
        self.clahe_selectors_frame = tk.Frame()
        # self.configure(bg='green')  # for dev.

        # Note: The matching control variable attributes for the
        #   following 14 selector widgets are in ProcessImage __init__.
        self.slider = {
            'clip_limit': tk.Scale(master=self.clahe_selectors_frame),
            'clip_limit_lbl': tk.Label(master=self.clahe_selectors_frame),

            'tile_size': tk.Scale(master=self.clahe_selectors_frame),
            'tile_size_lbl': tk.Label(master=self.clahe_selectors_frame),
        }

        self.reset_btn = ttk.Button(master=self)
        self.save_btn = ttk.Button(master=self)

        # Separator used in settings report window.
        self.separator = ttk.Separator(master=self)

        # Is an instance attribute here only because it is used in call
        #  to utils.save_settings_and_img() from the Save button.
        self.clahe_settings_txt = ''

        # A np.ndarray of the flattened input grayscale image used for histogram.
        self.flat_gray = None

        # Put everything in place, establish initial settings and displays.
        self.setup_image_windows()
        self.setup_report_window()
        self.config_sliders()
        self.config_buttons()
        self.grid_widgets()
        self.display_images()
        self.set_clahe_defaults()
        self.show_histograms()

    def setup_image_windows(self) -> None:
        """
        Create and configure all Toplevel windows and their Labels
        needed to display processed images.

        Returns: None
        """

        def no_exit_on_x():
            """
            Provide a notice in Terminal. Called from .protocol() in loop.
            """
            print('Image windows cannot be closed from the window bar.\n'
                  'They can be minimized to get them out of the way.\n'
                  'You can quit the program from the OpenCV Settings Report'
                  '  window bar or Esc or Ctrl-Q keys.'
                  )

        # NOTE: dict item order affects the order that windows are
        #  drawn, so here use an inverse order of processing steps to
        #  arrange windows overlaid from first to last, e.g.,
        #  input on bottom, sized or clahe layered on top.
        self.img_window = {
            'clahe': tk.Toplevel(),
            'input': tk.Toplevel(),
        }

        # Move the report (app, self) to the right side,
        #  CLAHE window toward the center; stack the input images window
        #  below those. Histogram window by default will be on the left side.
        input_w_offset = int(self.winfo_screenwidth() * 0.25)
        input_h_offset = int(self.winfo_screenheight() * 0.1)
        clahe_w_offset = int(self.winfo_screenwidth() * 0.5)
        clahe_h_offset = int(self.winfo_screenheight() * 0.5)
        self.img_window['input'].lower(belowThis=self)
        self.img_window['input'].geometry(f'+{input_w_offset}+{input_h_offset}')
        self.img_window['clahe'].geometry(f'+{clahe_w_offset}+{clahe_h_offset}')

        # The Labels to display scaled images, which are updated using
        #  .configure() for 'image=' in their respective processing methods.
        self.img_label = {
            'input': tk.Label(self.img_window['input']),
            'gray': tk.Label(self.img_window['input']),
            'clahe': tk.Label(self.img_window['clahe']),
        }

        # Prevent user from inadvertently resizing a window too small to use.
        # Need to disable default window Exit in display windows b/c
        #  subsequent calls to them need a valid path name.
        for _, toplevel in self.img_window.items():
            toplevel.minsize(200, 200)
            toplevel.protocol('WM_DELETE_WINDOW', no_exit_on_x)

        self.img_window['input'].title(const.WIN_NAME['input+gray'])
        self.img_window['clahe'].title(const.WIN_NAME['clahe'])

        # Allow images to maintain borders and relative positions with window resize.
        self.img_window['input'].columnconfigure(0, weight=1)
        self.img_window['input'].columnconfigure(1, weight=1)
        self.img_window['input'].rowconfigure(0, weight=1)
        self.img_window['clahe'].columnconfigure(0, weight=1)
        self.img_window['clahe'].rowconfigure(0, weight=1)

    def setup_report_window(self) -> None:
        """
        Master (main tk window, "app") settings and reporting frames,
        and utility buttons, configurations, keybindings, and grids in the
        main "app" window.
        """

        # Need to provide exit info msg to Terminal.
        self.protocol('WM_DELETE_WINDOW', lambda: utils.quit_gui(mainloop=self,
                                                                 plot=True))

        self.bind_all('<Escape>', lambda _: utils.quit_gui(mainloop=self,
                                                           plot=True))
        self.bind_all('<Control-q>', lambda _: utils.quit_gui(mainloop=self,
                                                              plot=True))
        # ^^ Note: macOS Command-q will quit program without utils.quit_gui info msg.

        # Place settings/report window on upper right of screen. Width factor
        #  of 0.66 was empirically determined from ~width of report (self) window.
        w_offset = int(self.winfo_screenwidth() * 0.66)
        self.geometry(f'+{w_offset}+0')

        # Need to have the report window on top and take focus away from the
        #  Histogram window so that user can immediately quit with keys and not
        #  have to first click on the report window.
        #  Each platform's window manager may draw windows in different order.
        self.tkraise(aboveThis=self.img_window['clahe'])
        self.after(1, lambda: self.focus_force())

        self.config(
            bg=const.MASTER_BG,  # gray80 matches report_clahe() txt fg.
            # bg=const.CBLIND_COLOR_TK['sky blue'],  # for dev.
            highlightthickness=5,
            highlightcolor=const.CBLIND_COLOR_TK['yellow'],
            highlightbackground=const.DRAG_GRAY
        )
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.clahe_report_frame.configure(relief='flat',
                                          bg=const.CBLIND_COLOR_TK['sky blue']
                                          )  # bg won't show with grid sticky EW.
        self.clahe_report_frame.columnconfigure(1, weight=1)
        self.clahe_report_frame.rowconfigure(0, weight=1)

        self.clahe_selectors_frame.configure(relief='raised',
                                             bg=const.DARK_BG,
                                             borderwidth=5, )
        self.clahe_selectors_frame.columnconfigure(0, weight=1)
        self.clahe_selectors_frame.columnconfigure(1, weight=1)

        self.clahe_report_frame.grid(column=0, row=0,
                                     columnspan=2,
                                     padx=5, pady=5,
                                     sticky=tk.EW)
        self.clahe_selectors_frame.grid(column=0, row=1,
                                        columnspan=2,
                                        padx=5, pady=(0, 5),
                                        ipadx=4, ipady=4,
                                        sticky=tk.EW)

    def config_sliders(self) -> None:
        """
        Configure arguments for Scale() sliders.

        Returns: None
        """

        # On the development Linux Ubuntu PC, calling process_all() for
        #  slider command arguments causes flickering of the report
        #  Frame text. This doesn't happen on Windows or macOS. Therefore,
        #  replace continuous slide processing on Linux with bindings
        #  to call process_all() only on left button release -> no flicker.

        if const.MY_OS == 'lin':
            slider_cmd = ''

            # Note that the if '_lbl' condition isn't needed to improve
            #   performance; it's just there for clarity's sake.
            for name, widget in self.slider.items():
                if '_lbl' not in name:
                    widget.bind('<ButtonRelease-1>', self.process_all)
        else:  # is Windows or macOS
            slider_cmd = self.process_all

        self.slider['clip_limit_lbl'].configure(text='Clip limit:',
                                                **const.LABEL_PARAMETERS)
        self.slider['clip_limit'].configure(from_=0.1, to=5,
                                            resolution=0.1,
                                            tickinterval=1,
                                            variable=self.slider_val['clip_limit'],
                                            command=slider_cmd,
                                            **const.SCALE_PARAMETERS)

        self.slider['tile_size_lbl'].configure(text='Tile size (px):',
                                               **const.LABEL_PARAMETERS)
        self.slider['tile_size'].configure(from_=1, to=200,
                                           tickinterval=20,
                                           variable=self.slider_val['tile_size'],
                                           command=slider_cmd,
                                           **const.SCALE_PARAMETERS)

    def config_buttons(self) -> None:
        """
        Configure utility Buttons.
        Called from __init__.

        Returns: None
        """
        manage.ttk_styles(self)

        def save_settings():
            """
            A Button "command" kw caller to avoid messy or lambda
            statements.
            """
            utils.save_settings_and_img(img2save=self.clahe_img,
                                        txt2save=self.clahe_settings_txt,
                                        caller='CLAHE')

        button_params = dict(
            style='My.TButton',
            width=0)

        self.reset_btn.config(text='Reset to defaults',
                              command=self.set_clahe_defaults,
                              **button_params)

        self.save_btn.config(text='Save settings & image',
                             command=save_settings,
                             **button_params)

    def grid_widgets(self) -> None:
        """
        Widgets' grid in the main, "app", window.
        Note: Grid all here to make clear the spatial relationships
        of all elements.
        """

        # Use the dict() function with keyword arguments to mimic the
        #  keyword parameter structure of the grid() function.
        if const.MY_OS in 'lin, win':
            slider_grid_params = dict(
                padx=5,
                pady=(7, 0),
                sticky=tk.W)
            label_grid_params = dict(
                padx=5,
                pady=(5, 0),
                sticky=tk.E)
            grid_params = dict(
                padx=(10, 0),
                pady=(0, 5),
                sticky=tk.W)
        else:  # is macOS
            slider_grid_params = dict(
                padx=5,
                pady=(4, 0))
            label_grid_params = dict(
                padx=5,
                pady=(4, 0),
                sticky=tk.E)
            grid_params = dict(
                padx=(10, 0),
                pady=(0, 5),
                sticky=tk.W)

        # Widgets gridded in the self.clahe_selectors_frame Frame.
        # Sorted by row number:
        self.slider['clip_limit_lbl'].grid(column=0, row=0,
                                           **label_grid_params)
        self.slider['clip_limit'].grid(column=1, row=0,
                                       **slider_grid_params)

        self.slider['tile_size_lbl'].grid(column=0, row=1,
                                          **label_grid_params)
        self.slider['tile_size'].grid(column=1, row=1,
                                      **slider_grid_params)

        self.separator.grid(column=0, row=2,
                            columnspan=2,
                            pady=6,
                            sticky=tk.NSEW)

        self.reset_btn.grid(column=0, row=3,
                            **grid_params)
        self.save_btn.grid(column=0, row=4,
                           **grid_params)

    def display_images(self) -> None:
        """
        Converts input image and its grayscale to tk image format and
        displays them as panels gridded in their toplevel window.
        Displays the CLAHE image from ProcessImage.apply_clahe().
        Calls manage.tkimage(), which applies scaling, cv2 -> tk array
        conversion, and updates the panel Label's image parameter.
        """

        panel_left = dict(
            column=0, row=0,
            padx=5, pady=5,
            sticky=tk.NSEW)
        panel_right = dict(
            column=1, row=0,
            padx=5, pady=5,
            sticky=tk.NSEW)

        # Display the input image and its grayscale; both are static, so
        #  do not need updating, but retain the image display statement
        #  structure of processed images that do need updating.
        # Note: Use 'self' to scope the ImageTk.PhotoImage in the Class,
        #  otherwise it will/may not show b/c of garbage collection.
        self.tkimg['input'] = manage.tk_image(INPUT_IMG)
        self.img_label['input'].configure(image=self.tkimg['input'])
        self.img_label['input'].grid(**panel_left)

        self.tkimg['gray'] = manage.tk_image(GRAY_IMG)
        self.img_label['gray'].configure(image=self.tkimg['gray'])
        self.img_label['gray'].grid(**panel_right)

        # Remember that the CLAHE image is converted and configured in
        #   ProcessImage.apply_clahe().
        self.img_label['clahe'].grid(**panel_left)

        # To avoid numerous calls for ravel(), pre-process the flattened
        #  grayscale here to use for the input histogram in show_histograms().
        self.flat_gray = GRAY_IMG.ravel()

    def set_clahe_defaults(self) -> None:
        """
        Sets slider widgets at startup. Called from "Reset" button.
        """

        # Set/Reset Scale widgets.
        self.slider_val['clip_limit'].set(2.0)
        self.slider_val['tile_size'].set(8)

        # Apply the default settings.
        self.process_all()

    def show_histograms(self) -> None:
        """
        Updates CLAHE adjusted histogram plot with Matplotlib from
        trackbar changes. Called from apply_clahe().

        Args: A flattened (1-D) ndarray of the image so that its
            histogram can be displayed.

        Returns: None
        """
        # Need to clear prior histograms before drawing new ones.
        #  Redrawing both histograms is slower, but it works.
        plt.cla()

        # hist() returns tuple of (counts(n), bins(edges), patches(artists)
        # histtype='step' draws a line, 'stepfilled' fills under the line;
        #   both are patches. Polygon artists provide faster rendering
        #   than the default 'bar', which is a BarContainer object of
        #   Rectangle artists.
        # For input img, use 'step' and pre-flattened ndarray for better
        #   performance when overlaying a static histogram.
        plt.hist(self.flat_gray,
                 bins=255,
                 range=[0, 256],
                 color='black',
                 histtype='step',
                 label='Input, grayscale'
                 )
        plt.hist(self.clahe_img.ravel(),
                 bins=255,
                 range=[0, 256],
                 color='orange',
                 histtype='stepfilled',
                 label='CLAHE adjusted'
                 )

        plt.title('Histograms', fontweight='bold')
        plt.xlabel('Pixel value')
        plt.ylabel('Pixel count')
        plt.legend(fontsize=8)  # Default size is 10

    def report_clahe(self) -> None:
        """
        Write the current settings and cv2 metrics in a Text widget of
        the report_frame. Same text is printed in Terminal from "Save"
        button. Called from __init__ and process_all().
        """

        # Note: recall that *_val dict are inherited from ProcessImage().
        clip_limit = self.slider_val['clip_limit'].get()
        tile_size = (self.slider_val['tile_size'].get(),
                     self.slider_val['tile_size'].get())

        # Text is formatted for clarity in window, terminal, and saved file.
        self.clahe_settings_txt = (
            f'Image: {INPUT_PATH}\n\n'
            f'Input grayscale pixel value: mean {self.input_mean},'
            f' stdev {self.input_sd}\n'
            f'cv2.createCLAHE cliplimit={clip_limit}, tileGridSize{tile_size}\n'
            f'CLAHE grayscale pixel value: mean {self.clahe_mean},'
            f' stdev {self.clahe_sd}\n'
        )

        utils.display_report(frame=self.clahe_report_frame,
                             report=self.clahe_settings_txt)

    def process_all(self, event=None) -> None:
        """
        Runs all image processing methods and the settings report.
        Calls apply_clahe(), report_clahe(), self_histograms().

        Args:
            event: The implicit mouse button event.
        Returns: *event* as a formality; is functionally None.
        """
        self.apply_clahe()  # inherited from ProcessImage
        self.show_histograms()
        self.report_clahe()

        return event


if __name__ == "__main__":
    # Program exits here if the system platform or Python version check fails.
    utils.check_platform()
    vcheck.minversion('3.7')

    INPUT_PATH = manage.arguments()['input']
    # Need file check here instead of in manage.arguments() to avoid
    #   numerous calls to that module.
    if not Path.exists(utils.valid_path_to(INPUT_PATH)):
        sys.exit(f'COULD NOT OPEN the image: {INPUT_PATH}  <-Check spelling.\n'
                 "  If spelled correctly, then try using the file's absolute (full) path.")

    # All checks are good, so define some additional run-specific constants...
    infile_dict = manage.infile()
    INPUT_IMG = infile_dict['input_img']
    GRAY_IMG = infile_dict['gray_img']
    LINE_THICKNESS = infile_dict['line_thickness']

    try:
        app = ImageViewer()
        app.title('CLAHE Settings Report')
        app.resizable(False, False)
        print(f'{Path(__file__).name} is now running...')
        app.mainloop()
    except KeyboardInterrupt:
        print('*** User quit the program from Terminal/Console ***\n')
