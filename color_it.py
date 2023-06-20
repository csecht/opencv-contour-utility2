#!/usr/bin/env python3
"""
Use a tkinter GUI to explore OpenCV's CLAHE image processing parameters.
Parameter values are adjusted with slide bars.

USAGE Example command lines, from within the image-processor-main folder:
python3 -m color_it --help
python3 -m color_it --about
python3 -m color_it --input images/sample4.jpg
python3 -m color_it -i images/sample4.jpg -s 0.6

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
    from tkinter import ttk

except (ImportError, ModuleNotFoundError) as import_err:
    sys.exit(
        '*** One or more required Python packages were not found'
        ' or need an update:\nOpenCV-Python or tkinter (Tk/Tcl).\n\n'
        'To install: from the current folder, run this command'
        ' for the Python package installer (PIP):\n'
        '   python3 -m pip install -r requirements.txt\n\n'
        'Alternative command formats (system dependent):\n'
        '   py -m pip install -r requirements.txt (Windows)\n'
        '   pip install -r requirements.txt\n\n'
        'You may also install directly using, for example, this command,'
        ' for the Python package installer (PIP):\n'
        '   python3 -m pip install opencv-python\n\n'
        'On Linux, if tkinter is the problem, then you may need:\n'
        '   sudo apt-get install python3-tk\n\n'
        'See also: https://numpy.org/install/\n'
        '  https://tkdocs.com/tutorial/install.html\n'
        '  https://docs.opencv.org/4.6.0/d5/de5/tutorial_py_setup_in_windows.html\n\n'
        f'Error message:\n{import_err}')


class ProcessImage(tk.Tk):
    """
    Uses OpenCV methods to apply color-specific masks to the image
    specified in the input file.

    Class methods:
   find_colors()
    """

    __slots__ = (
        'cbox_val', 'hibound', 'hsv_img', 'img_label', 'img_window',
        'lobound', 'radio_val', 'slider_val', 'tkimg', 'tk',
    )

    def __init__(self):
        super().__init__()

        self.slider_val = {
            'H_min': tk.IntVar(),
            'S_min': tk.IntVar(),
            'V_min': tk.IntVar(),
            'H_max': tk.IntVar(),
            'S_max': tk.IntVar(),
            'V_max': tk.IntVar(),
        }

        self.cbox_val = {
            'color_pref': tk.StringVar(),
        }

        self.radio_val = {
            'filter_pref': tk.BooleanVar(),
            'redux_pref': tk.BooleanVar(),
        }

        # Arrays of images to be processed. When used within a method,
        #  the purpose of self.tkimg[*] as an instance attribute is to
        #  retain the attribute reference and thus prevent garbage collection.
        #  Dict values will be defined for panels of PIL ImageTk.PhotoImage
        #  with Label images displayed in their respective img_window Toplevel.
        self.tkimg = {
            'input2color': tk.PhotoImage(),
            'hsv': tk.PhotoImage(),
            'color': tk.PhotoImage(),
        }

        # Tuples used for setting and reporting the HSV ranges for masks.
        self.lobound = 0, 0, 0
        self.hibound = 0, 0, 0

        # Dict items are defined in ImageViewer.setup_image_windows().
        self.img_window = None
        self.img_label = None

        self.hsv_img = None

        # The highlight color used to draw contours and shapes.
        # if arguments['color'] == 'yellow':
        #     self.contour_color = const.CBLIND_COLOR_CV['yellow']
        # else:  # is default cv2 contour color, green, as (B,G,R).
        #     self.contour_color = arguments['color']

    def find_colors(self, color2find=None) -> None:
        """
        Applies a mask to image for the specified color range. Provides
        optional pre-filtering of image and post-noise reduction of mask.

        Args:
            color2find: Uses pre-set color ranges from a dictionary when
                        specified, otherwise uses range values from
                        Scale() variables.
        Returns: None
        """
        # Coding ideas from:
        # https://techvidvan.com/tutorials/detect-objects-of-similar-color-using-opencv-in-python/

        h_min = self.slider_val['H_min'].get()
        s_min = self.slider_val['S_min'].get()
        v_min = self.slider_val['V_min'].get()
        h_max = self.slider_val['H_max'].get()
        s_max = self.slider_val['S_max'].get()
        v_max = self.slider_val['V_max'].get()

        self.lobound = h_min, s_min, v_min
        self.hibound = h_max, s_max, v_max

        # Note: these values are hard-coded in report_color_settings(),
        #   so update report text if these values change.
        # In 'else', need the same number of newlines as in the True condition.
        if self.radio_val['filter_pref'].get():
            filtered = cv2.bilateralFilter(src=INPUT_IMG,
                                           d=0,  # is calculated from sigma.
                                           sigmaColor=9,
                                           sigmaSpace=9,
                                           borderType=cv2.BORDER_REPLICATE)
            # filtered = cv2.GaussianBlur(src=INPUT_IMG,
            #                             ksize=(7, 7),
            #                             sigmaX=0,
            #                             sigmaY=0,
            #                             borderType=cv2.BORDER_REPLICATE)

            self.hsv_img = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
        else:  # User chose not to apply filter.
            self.hsv_img = cv2.cvtColor(INPUT_IMG, cv2.COLOR_BGR2HSV)

        # Dict values are the lower and upper (light & dark)
        #   BGR colorspace range boundaries to use for HSV color discrimination.
        if color2find:
            lower, upper = const.COLOR_BOUNDARIES[color2find]
            mask = cv2.inRange(self.hsv_img, lower, upper)

            # Note that this 'red' selection includes a red-orange also
            #  found by 'orange'.  Red color wraps around the HSV boundary,
            #  so need to merge two range sets to include all likely reds,
            #  as explained in:
            # https://stackoverflow.com/questions/30331944/
            #   finding-red-color-in-image-using-python-opencv
            # and in:
            # https://answers.opencv.org/question/28899/
            #   correct-hsv-inrange-values-for-red-objects/
            # The lower boundary set red is defined, now do the upper set
            #   to straddle over the HSV red break and join the masks.
            if color2find == 'red':
                mask_red = cv2.inRange(self.hsv_img,
                                       const.LOWER_RED, const.UPPER_RED)
                mask = mask + mask_red
        else:  # using sliders
            mask = cv2.inRange(self.hsv_img,
                               self.lobound, self.hibound)

        # Remove unnecessary noise from mask
        element = cv2.getStructuringElement(
            # shape=cv2.MORPH_ELLIPSE,  # cv2.MORPH_RECT, default
            # ksize=(3, 3),  # smaller ksize preserves details of mask.
            shape=cv2.MORPH_CROSS,  # CROSS to best bring out mask detail.
            ksize=(3, 3),  # smaller kernel preserves more details.
            anchor=(-1, -1)  # anchor cross kernel to center with -1.
        )

        # Apply noise reduction to the mask.
        redux_mask = cv2.morphologyEx(
            src=mask,
            op=cv2.MORPH_HITMISS,
            kernel=element,
            iterations=1,
            borderType=cv2.BORDER_DEFAULT
        )

        # Segment only the detected region. cv2.bitwise_and() applies mask on
        #  frame in only that region where the mask is true means white.
        # masked_img = cv2.bitwise_and(INPUT_IMG, INPUT_IMG, mask=filtered_mask)
        if self.radio_val['redux_pref'].get():
            masked_img = cv2.bitwise_and(INPUT_IMG, INPUT_IMG, mask=redux_mask)
        else:
            masked_img = cv2.bitwise_and(INPUT_IMG, INPUT_IMG, mask=mask)

        self.tkimg['color'] = manage.tk_image(masked_img, colorspace='bgr')
        self.img_label['color'].configure(image=self.tkimg['color'])


class ImageViewer(ProcessImage):
    """
    Uses OpenCV and tkinter methods to display color masks and report
    image processing settings and images to file,

    Class Methods:
    setup_image_windows -> no_exit_on_x
    setup_report_window
    config_sliders
    config_cbox
    config_buttons -> save_settings
    set_color_defaults
    grid_widgets
    display_images
    report_color_settings
    process_all
    """

    __slots__ = (
        'color_list', 'color_report_frame', 'color_selectors_frame',
        'color_settings_txt', 'flat_gray', 'reset_btn', 'save_btn',
        'filter_btn', 'redux_btn', 'separator', 'slider',
    )

    def __init__(self):
        super().__init__()

        self.color_report_frame = tk.Frame()
        self.color_selectors_frame = tk.Frame()
        # self.configure(bg='green')  # for development.

        # Note: The matching control variable attributes for the
        #   slider widgets are in ProcessImage __init__.
        self.slider = {
            'H_min': tk.Scale(master=self.color_selectors_frame),
            'H_min_lbl': tk.Label(master=self.color_selectors_frame),

            'S_min': tk.Scale(master=self.color_selectors_frame),
            'S_min_lbl': tk.Label(master=self.color_selectors_frame),

            'V_min': tk.Scale(master=self.color_selectors_frame),
            'V_min_lbl': tk.Label(master=self.color_selectors_frame),

            'H_max': tk.Scale(master=self.color_selectors_frame),
            'H_max_lbl': tk.Label(master=self.color_selectors_frame),

            'S_max': tk.Scale(master=self.color_selectors_frame),
            'S_max_lbl': tk.Label(master=self.color_selectors_frame),

            'V_max': tk.Scale(master=self.color_selectors_frame),
            'V_max_lbl': tk.Label(master=self.color_selectors_frame),
        }

        self.cbox = {
            'choose_color': ttk.Combobox(master=self.color_selectors_frame),
            'choose_color_lbl': tk.Label(master=self.color_selectors_frame),
        }

        self.radio = {
            'filter_lbl': tk.Label(master=self),
            'filter_yes': tk.Radiobutton(master=self),
            'filter_no': tk.Radiobutton(master=self),

            'redux_lbl': tk.Label(master=self),
            'redux_yes': tk.Radiobutton(master=self),
            'redux_no': tk.Radiobutton(master=self),
        }

        self.reset_btn = ttk.Button(master=self)
        self.save_btn = ttk.Button(master=self)

        # Separator used in settings report window.
        self.separator = ttk.Separator(master=self)

        self.color_list = []
        self.color_settings_txt = None

        # A np.ndarray of the flattened input grayscale image used for histogram.
        self.flat_gray = None

        # Put everything in place, establish initial settings and displays.
        self.setup_image_windows()
        self.setup_report_window()
        self.config_sliders()
        self.config_cbox()
        self.config_buttons()
        self.grid_widgets()
        self.set_color_defaults()

        self.find_colors()  # Called with argument from Combobox cmd.
        self.display_images()
        # On Linux Ubuntu, need update to activate self.force_focus().
        self.update_idletasks()

    def setup_image_windows(self) -> None:
        """
        Create and configure Toplevel window and to display input and
        processed images.

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
        #  arrange windows overlaid from first to last, unless .lower() or
        #  tkraise() are used.
        # NOTE: keys here must match corresponding keys in const.WIN_NAME
        self.img_window = {
            'input2color': tk.Toplevel(),
        }

        # Prevent user from inadvertently resizing a window too small to use.
        # Need to disable default window Exit in display windows b/c
        #  subsequent calls to them need a valid path name.
        # Allow image label panels in image windows to resize with window.
        #  Note that images don't proportionally resize, just their boundaries;
        #    images will remain anchored at their top left corners.
        for _name, toplevel in self.img_window.items():
            toplevel.minsize(200, 100)
            toplevel.protocol('WM_DELETE_WINDOW', no_exit_on_x)
            toplevel.columnconfigure(0, weight=1)
            toplevel.columnconfigure(1, weight=1)
            toplevel.rowconfigure(0, weight=1)
            toplevel.title(const.WIN_NAME[_name])

        # Move CLAHE window toward the center; stack the input images window
        #  below those. Histogram window by default will be on the left side.
        input_w_offset = int(self.winfo_screenwidth() * 0.25)
        input_h_offset = int(self.winfo_screenheight() * 0.1)
        self.img_window['input2color'].lower(belowThis=self)
        self.img_window['input2color'].geometry(f'+{input_w_offset}+{input_h_offset}')

        # The Labels to display scaled images, which are updated using
        #  .configure() for 'image=' in their respective processing methods.
        self.img_label = {
            'input2color': tk.Label(self.img_window['input2color']),
            'hsv': tk.Label(self.img_window['input2color']),
            'color': tk.Label(self.img_window['input2color']),
        }

    def setup_report_window(self) -> None:
        """
        Master (main tk window, "app") settings and reporting frames,
        utility buttons, configurations, and keybindings.
        """

        # Need to provide exit info msg to Terminal.
        self.protocol('WM_DELETE_WINDOW', lambda: utils.quit_gui(mainloop=app))

        self.bind_all('<Escape>', lambda _: utils.quit_gui(mainloop=app))
        self.bind_all('<Control-q>', lambda _: utils.quit_gui(mainloop=app))
        # ^^ Note: macOS Command-q will quit program without utils.quit_gui info msg.

        # Place settings/report window at upper right of screen.
        #   Note: the report window (self, app) width is ~ 600 across platforms,
        #   but instead of hard coding that, make geometry offset a function of
        #   screen width. This is needed b/c of differences among platforms'
        #   window managers in how they place windows.
        w_offset = int(self.winfo_screenwidth() * 0.66)
        self.geometry(f'+{w_offset}+0')

        self.after(1, self.focus_force)

        self.config(
            bg=const.MASTER_BG,  # gray80 matches report_color_settings() txt fg.
            # bg=const.CBLIND_COLOR_TK['sky blue'],  # for development.
            highlightthickness=5,
            highlightcolor=const.CBLIND_COLOR_TK['yellow'],
            highlightbackground=const.DRAG_GRAY
        )
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.color_report_frame.configure(relief='flat',
                                          bg=const.CBLIND_COLOR_TK['sky blue']
                                          )  # bg won't show with grid sticky EW.
        self.color_report_frame.columnconfigure(1, weight=1)
        self.color_report_frame.rowconfigure(0, weight=1)

        self.color_selectors_frame.configure(relief='raised',
                                             bg=const.DARK_BG,
                                             borderwidth=5, )
        self.color_selectors_frame.columnconfigure(0, weight=1)
        self.color_selectors_frame.columnconfigure(1, weight=1)

        self.color_report_frame.grid(column=0, row=0,
                                     columnspan=2,
                                     padx=5, pady=5,
                                     sticky=tk.EW)
        self.color_selectors_frame.grid(column=0, row=1,
                                        columnspan=2,
                                        padx=5, pady=(0, 5),
                                        ipadx=4, ipady=4,
                                        sticky=tk.EW)

    def config_sliders(self) -> None:
        """
        Configure arguments for Scale() sliders. Called from __init__.

        Returns: None
        """

        # On the development Linux Ubuntu PC, calling process_all() for
        #  slider command arguments causes flickering of the report
        #  Frame text. This doesn't happen on Windows or macOS. Therefore,
        #  replace continuous slide processing on Linux with bindings
        #  to call process_all() only on left button release so no flicker.

        if const.MY_OS == 'lin':
            slider_cmd = ''

            # Note that the if '_lbl' condition isn't needed to improve
            #   performance; it's just there for clarity's sake.
            for name, widget in self.slider.items():
                if '_lbl' not in name:
                    widget.bind('<ButtonRelease-1>', self.process_all)
        else:  # is Windows or macOS
            slider_cmd = self.process_all

        self.slider['H_min_lbl'].configure(text='H minimum:',
                                           **const.LABEL_PARAMETERS)
        self.slider['H_min'].configure(from_=1, to=255,
                                       resolution=1,
                                       tickinterval=20,
                                       variable=self.slider_val['H_min'],
                                       command=slider_cmd,
                                       **const.SCALE_PARAMETERS)

        self.slider['S_min_lbl'].configure(text='S minimum:',
                                           **const.LABEL_PARAMETERS)
        self.slider['S_min'].configure(from_=1, to=255,
                                       resolution=1,
                                       tickinterval=20,
                                       variable=self.slider_val['S_min'],
                                       command=slider_cmd,
                                       **const.SCALE_PARAMETERS)

        self.slider['V_min_lbl'].configure(text='V minimum:',
                                           **const.LABEL_PARAMETERS)
        self.slider['V_min'].configure(from_=1, to=255,
                                       resolution=1,
                                       tickinterval=20,
                                       variable=self.slider_val['V_min'],
                                       command=slider_cmd,
                                       **const.SCALE_PARAMETERS)

        self.slider['H_max_lbl'].configure(text='H maximum:',
                                           **const.LABEL_PARAMETERS)
        self.slider['H_max'].configure(from_=1, to=255,
                                       resolution=1,
                                       tickinterval=20,
                                       variable=self.slider_val['H_max'],
                                       command=slider_cmd,
                                       **const.SCALE_PARAMETERS)

        self.slider['S_max_lbl'].configure(text='S maximum:',
                                           **const.LABEL_PARAMETERS)
        self.slider['S_max'].configure(from_=1, to=255,
                                       resolution=1,
                                       tickinterval=20,
                                       variable=self.slider_val['S_max'],
                                       command=slider_cmd,
                                       **const.SCALE_PARAMETERS)

        self.slider['V_max_lbl'].configure(text='V maximum:',
                                           **const.LABEL_PARAMETERS)
        self.slider['V_max'].configure(from_=1, to=255,
                                       resolution=1,
                                       tickinterval=20,
                                       variable=self.slider_val['V_max'],
                                       command=slider_cmd,
                                       **const.SCALE_PARAMETERS)

    def config_cbox(self) -> None:
        """
        Configure arguments for the color selection Combobox().
        Called from __init__.

        Returns: None
        """

        # Different Combobox widths are needed to account for font widths
        #  and padding in different systems.
        width_correction = 2 if const.MY_OS == 'win' else 0  # is Linux or macOS

        self.color_list = list(const.COLOR_BOUNDARIES.keys())
        self.color_list.insert(0, 'Use sliders')
        self.cbox['choose_color_lbl'].config(text='Use sliders or pick colors:',
                                             **const.LABEL_PARAMETERS)
        self.cbox['choose_color'].config(textvariable=self.cbox_val['color_pref'],
                                         width=14 + width_correction,
                                         values=self.color_list,
                                         **const.COMBO_PARAMETERS)
        # Set to green at startup, or 1st color in list.
        self.cbox['choose_color'].current(1)  # Set to green at startup.

        # Now bind functions to all Comboboxes.
        # Note that the if condition doesn't seem to be needed to
        #   improve performance or affect bindings;
        #   it just clarifies the intention.
        for name, widget in self.cbox.items():
            if '_lbl' not in name:
                widget.bind('<<ComboboxSelected>>', func=self.process_all)

    def config_buttons(self) -> None:
        """
        Configure utility Buttons and Radiobuttons. Called from __init__.

        Returns: None
        """
        manage.ttk_styles(self)

        def save_settings():
            """
            A Button "command" kw caller to avoid messy or lambda
            statements.
            """
            utils.save_settings_and_img(img2save=self.tkimg['color'],
                                        txt2save=self.color_settings_txt,
                                        caller='COLOR')

        button_params = dict(
            style='My.TButton',
            width=0)

        self.reset_btn.config(text='Reset to defaults',
                              command=self.set_color_defaults,
                              **button_params)

        self.save_btn.config(text='Save settings & image',
                             command=save_settings,
                             **button_params)

        self.radio['filter_lbl'].config(text='Apply blur filter to image?',
                                        **const.LABEL_PARAMETERS)
        self.radio['filter_yes'].config(text='Yes',
                                        variable=self.radio_val['filter_pref'],
                                        value=True,
                                        command=self.process_all,
                                        **const.RADIO_PARAMETERS)
        self.radio['filter_no'].config(text='No',
                                       variable=self.radio_val['filter_pref'],
                                       value=False,
                                       command=self.process_all,
                                       **const.RADIO_PARAMETERS)

        self.radio['redux_lbl'].config(text='Apply noise reduction to mask?',
                                       **const.LABEL_PARAMETERS)
        self.radio['redux_yes'].config(text='Yes',
                                       variable=self.radio_val['redux_pref'],
                                       value=True,
                                       command=self.process_all,
                                       **const.RADIO_PARAMETERS)
        self.radio['redux_no'].config(text='No',
                                      variable=self.radio_val['redux_pref'],
                                      value=False,
                                      command=self.process_all,
                                      **const.RADIO_PARAMETERS)

    def set_color_defaults(self) -> None:
        """
        Sets selector widgets values.
        Called from "Reset" button and __init__. Calls process_all().

        Returns: None
        """

        # Set/Reset Scale widgets.
        #  Default color range is green.
        self.slider_val['H_min'].set(36)
        self.slider_val['S_min'].set(25)
        self.slider_val['V_min'].set(25)
        self.slider_val['H_max'].set(70)
        self.slider_val['S_max'].set(255)
        self.slider_val['V_max'].set(255)

        self.cbox_val['color_pref'].set('green')

        self.radio['filter_yes'].select()
        self.radio['redux_yes'].select()

        # Apply the default settings.
        self.process_all()

    def grid_widgets(self) -> None:
        """
        Widgets' grids in the main, "app", window.

        Returns: None
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
                padx=(0, 0),
                pady=(0, 5),
                sticky=tk.W)

        # Widgets gridded in the self.color_selectors_frame Frame.
        # Sorted by row number:
        self.slider['H_min_lbl'].grid(column=0, row=0,
                                      **label_grid_params)
        self.slider['H_min'].grid(column=1, row=0,
                                  **slider_grid_params)

        self.slider['S_min_lbl'].grid(column=0, row=1,
                                      **label_grid_params)
        self.slider['S_min'].grid(column=1, row=1,
                                  **slider_grid_params)

        self.slider['V_min_lbl'].grid(column=0, row=2,
                                      **label_grid_params)
        self.slider['V_min'].grid(column=1, row=2,
                                  **slider_grid_params)

        self.slider['H_max_lbl'].grid(column=0, row=3,
                                      **label_grid_params)
        self.slider['H_max'].grid(column=1, row=3,
                                  **slider_grid_params)

        self.slider['S_max_lbl'].grid(column=0, row=4,
                                      **label_grid_params)
        self.slider['S_max'].grid(column=1, row=4,
                                  **slider_grid_params)

        self.slider['V_max_lbl'].grid(column=0, row=5,
                                      **label_grid_params)
        self.slider['V_max'].grid(column=1, row=5,
                                  **slider_grid_params)

        self.cbox['choose_color_lbl'].grid(column=1, row=9,
                                           padx=(0, 0),
                                           pady=(5, 5),
                                           sticky=tk.W)
        self.cbox['choose_color'].grid(column=1, row=9,
                                       padx=(150, 0),
                                       pady=(5, 5),
                                       sticky=tk.W)

        self.separator.grid(column=0, row=6,
                            columnspan=2,
                            pady=6,
                            sticky=tk.NSEW)

        self.reset_btn.grid(column=0, row=7,
                            **grid_params)
        self.save_btn.grid(column=0, row=8,
                           **grid_params)

        self.radio['filter_lbl'].grid(column=1, row=7,
                                      **grid_params)
        self.radio['filter_yes'].grid(column=1, row=7,
                                      padx=(0, 70),
                                      pady=(0, 5),
                                      sticky=tk.E)
        self.radio['filter_no'].grid(column=1, row=7,
                                     padx=(0, 100),
                                     pady=(0, 5),
                                     sticky=tk.E)

        self.radio['redux_lbl'].grid(column=1, row=8,
                                     **grid_params)
        self.radio['redux_yes'].grid(column=1, row=8,
                                     padx=(0, 45),
                                     pady=(0, 5),
                                     sticky=tk.E)
        self.radio['redux_no'].grid(column=1, row=8,
                                    padx=(0, 75),
                                    pady=(0, 5),
                                    sticky=tk.E)

    def display_images(self) -> None:
        """
        Displays input and processed cv2 images in a tk image format
        in a Toplevel window.
        Displays the color-masked image from find_colors() (inherited).
        Calls manage.tkimage(), which applies scaling, cv2 -> tk array
        conversion, and updates the panel Label's image parameter.

        Returns: None
        """

        # Display the input image and its grayscale; both are static, so
        #  do not need updating, but retain the image display statement
        #  structure of processed images that do need updating.
        # Note: Use 'self' to scope the ImageTk.PhotoImage in the Class,
        #  otherwise it will/may not show b/c of garbage collection.
        self.tkimg['input2color'] = manage.tk_image(INPUT_IMG, colorspace='bgr')
        self.img_label['input2color'].configure(image=self.tkimg['input2color'])
        self.img_label['input2color'].grid(**const.PANEL_LEFT)

        # Remember that the colored image is converted and configured in
        #   ProcessImage.find_colors().
        self.img_label['color'].grid(**const.PANEL_RIGHT)

    def report_color_settings(self) -> None:
        """
        Write the current settings and cv2 metrics in a Text widget of
        the report_frame. Same text is printed in Terminal from "Save"
        button. Called from __init__ and process_all().

        Returns: None
        """

        indent = ' '.ljust(18)
        bigindent = ' '.ljust(26)
        selected_color = self.cbox['choose_color'].get()

        if selected_color == self.color_list[0]:  # is 'Use sliders'
            range_txt = (f'Selected values, lower HSV range: {self.lobound}\n'
                         f'Selected values, upper HSV range: {self.hibound}\n')
        else:  # a color is selected
            lobound, hibound = const.COLOR_BOUNDARIES[selected_color]
            range_txt = (f'Pre-set values, lower HSV range: {lobound}\n'
                         f'Pre-set values, upper HSV range: {hibound}\n')

        # Note: these values need to be updated with those is used in find_colors().
        # In 'else' statement, keep the same number of newlines as in True statement.
        if self.radio_val['filter_pref'].get():
            filter_txt = (
                'cv2.bilateralFilter(src=INPUT_IMG, d=0,\n'
                f'{indent}sigmaColor=9, sigmaSpace=9,\n'
                f'{indent}borderType=cv2.BORDER_REPLICATE)')
        else:  # is False, 'No' selected
            filter_txt = 'None selected.\n\n'

        if self.radio_val['redux_pref'].get():
            redux_txt = (
                'cv2.morphologyEx(src=mask,\n'
                f'{bigindent}op=cv2.MORPH_HITMISS,\n'
                f'{bigindent}kernel=element, iterations=1,\n'
                f'{bigindent}borderType=cv2.BORDER_DEFAULT)\n'
                ' ...with cv2.getStructuringElement(shape=cv2.MORPH_CROSS,\n'
                f'{bigindent}ksize=(3, 3), anchor=(-1, -1))')
        else:  # is False, 'No' selected
            redux_txt = 'None selected.\n\n\n\n\n'

        # Text is formatted for clarity in window, terminal, and saved file.
        self.color_settings_txt = (
            f'Image: {INPUT_PATH}\n\n'
            f'Pre-set color: {selected_color}\n\n'
            f'{range_txt}\n'
            'Displayed min and max values are the BGR values used\n'
            '  for HSV color discrimination.\n'
            'Note that when "red" is selected, the cv2.inRange mask\n'
            '  used to find HSV colors is a composite of displayed\n'
            '  values plus hard-coded lower and upper ranges:\n'
            f'  {const.LOWER_RED} and {const.UPPER_RED}\n'
            f'Image filter: {filter_txt}\n'
            f'Mask noise reduction: {redux_txt}\n'
        )

        utils.display_report(frame=self.color_report_frame,
                             report=self.color_settings_txt)

    def process_all(self, event=None) -> None:
        """
        Runs all image processing methods and the settings report.
        Called from every selector widget or its mouse binding.
        Calls find_colors() and report_color_settings().

        Args:
            event: The implicit mouse button event.

        Returns: *event* as a formality; is functionally None.
        """

        selected_color = self.cbox_val['color_pref'].get()

        # For user convenience, set sliders to match selected color,
        #   but don't allow changing sliders until 'Use sliders' is selected.
        if selected_color == self.color_list[0]:  # 'Use sliders'
            for _s in self.slider_val:
                self.slider[_s].config(state=tk.NORMAL)

            self.find_colors()
        else:  # A color has been chosen from the combobox.
            lobound, hibound = const.COLOR_BOUNDARIES[selected_color]
            lo_h, lo_s, lo_v = lobound
            hi_h, hi_s, hi_v = hibound
            self.slider_val['H_min'].set(lo_h)
            self.slider_val['S_min'].set(lo_s)
            self.slider_val['V_min'].set(lo_v)
            self.slider_val['H_max'].set(hi_h)
            self.slider_val['S_max'].set(hi_s)
            self.slider_val['V_max'].set(hi_v)

            for _s in self.slider_val:
                self.slider[_s].config(state=tk.DISABLED)

            self.find_colors(color2find=selected_color)

        self.report_color_settings()

        return event


if __name__ == "__main__":
    # Program exits here if the system platform or Python version check fails.
    utils.check_platform()
    vcheck.minversion('3.7')
    arguments = manage.arguments()

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
        app.title('Color Settings Report')
        app.resizable(False, False)
        print(f'{Path(__file__).name} is now running...')
        app.mainloop()
    except KeyboardInterrupt:
        print('*** User quit the program from Terminal/Console ***\n')
