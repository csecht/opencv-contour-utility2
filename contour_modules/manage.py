"""
General housekeeping utilities.
Functions:
arguments: handles command line arguments
infile: reads specified input image and derives associated metrics.
scale: manages the specified scale factor for display of images.
tk_image: converts scaled cv2 image to a compatible tk.TK image format.
"""
# Copyright (C) 2022 C.S. Echt, under GNU General Public License'

# Standard library imports.
import argparse
import math
import sys
import tkinter

# noinspection PyCompatibility
from __main__ import __doc__

# Third party imports.
import cv2
import numpy as np
from PIL import Image, ImageTk
from PIL.ImageTk import PhotoImage
from tkinter import ttk

# Local application imports.
import contour_modules
from contour_modules import constants as const


def arguments() -> dict:
    """
    Handle command line arguments.

    Returns: Dictionary of argument values.
    """

    parser = argparse.ArgumentParser(description='Explore Image Processing Parameters.')
    parser.add_argument('--about',
                        help='Provide description, version, GNU license',
                        action='store_true',
                        default=False)
    parser.add_argument('--input', '-i',
                        help='Path to input image.',
                        default='images/sample1.jpg',
                        metavar='PATH/FILE')
    parser.add_argument('--scale', '-s',
                        help='Factor, X, to change displayed image size (default: 1.0).',
                        default=1.0,
                        type=float,
                        required=False,
                        metavar='X')
    parser.add_argument('--color', '-c',
                        help='CV contour color, C. (default: green; option: yellow).',
                        default='green',
                        choices=('green', 'yellow'),
                        metavar='C')

    args = parser.parse_args()

    about_text = (f'{__doc__}\n'
                  f'{"Author:".ljust(13)}{contour_modules.__author__}\n'
                  f'{"Version:".ljust(13)}{contour_modules.__version__}\n'
                  f'{"Status:".ljust(13)}{contour_modules.__status__}\n'
                  f'{"URL:".ljust(13)}{contour_modules.URL}\n'
                  f'{contour_modules.__copyright__}'
                  f'{contour_modules.__license__}\n'
                  )

    if args.about:
        print('====================== ABOUT START ====================')
        print(about_text)
        print('====================== ABOUT END ====================')

        sys.exit(0)

    if args.scale <= 0:
        args.scale = 1
        print('--scale X: X must be greater than zero. Resetting to 1.')

    if args.color == 'green':
        args.color = (0, 255, 0)
    elif args.color == 'yellow':
        args.color = const.CBLIND_COLOR_CV['yellow']

    arg_dict = {
        'about': args.about,
        'input': args.input,
        'scale': args.scale,
        'color': args.color,
    }

    return arg_dict


def infile() -> dict:
    """
    Read the image file specified in the --input command line option,
    then calculates and assign to a dictionary values that can be used
    in the main script as constants for that input image.

    Returns: Dictionary of various image metrics.
    """

    # manage.arguments() has verified the image path, so read from it.
    input_img = cv2.imread(arguments()['input'])
    gray_img = cv2.imread(arguments()['input'], cv2.IMREAD_GRAYSCALE)

    # Ideas for scaling: https://stackoverflow.com/questions/52846474/
    #   how-to-resize-text-for-cv2-puttext-according-to-the-image-size-in-opencv-python
    size2scale = min(input_img.shape[0], input_img.shape[1])

    # used only in PI.circle_the_contour() putText
    font_scale = max(size2scale * const.FONT_SCALE, 0.5)
    center_xoffset = math.ceil(size2scale * const.CENTER_XSCALE * arguments()['scale'])

    # used in PI.circle_the_contour() putText and PI.contour_threshold, etc
    line_thickness = math.ceil(size2scale * const.LINE_SCALE)  # * arguments()['scale'])

    managed_outputs = {
        'input_img': input_img,
        'gray_img': gray_img,
        'font_scale': font_scale,
        'center_xoffset': center_xoffset,
        'line_thickness': line_thickness,
        'size2scale': size2scale,
    }
    return managed_outputs


def scale(img: np.ndarray, scalefactor: float) -> np.ndarray:
    """
    Change size of displayed images from original (input) size.
    Intended mainly for when input image is too large to fit on screen.

    Args:
        img: A numpy.ndarray of image to be scaled.
        scalefactor: The multiplication factor to grow or shrink the
                displayed image. Defined from cmd line arg '--scale'.
                Default from argparse is 1.0.

    Returns: A scaled np.ndarray object; if *scale* is 1, then no change.
    """

    # Is redundant with check of --scale value in args_handler().
    scalefactor = 1 if scalefactor == 0 else scalefactor

    # Provide the best interpolation method for slight improvement of
    #  resized image depending on whether it is down- or up-scaled.
    interpolate = cv2.INTER_AREA if scalefactor < 0 else cv2.INTER_CUBIC

    scaled_image = cv2.resize(src=img,
                              dsize=None,
                              fx=scalefactor, fy=scalefactor,
                              interpolation=interpolate)
    return scaled_image


# def tkimage(input_tkimg: tuple, gray_tkimg: tuple = None):
def tk_image(image: np.ndarray) -> PhotoImage:
    """
    Scales and converts cv2 images to a compatible format for display
    in tk window. Be sure that the returned image is properly scoped in
    the Class where it is called; e.g., use as self.tk_image attribute.

    Args:
        image: A cv2 numpy array of the image to scale and convert to
               a PIL ImageTk.PhotoImage.

    Returns: Scaled PIL ImageTk.PhotoImage to display in tk.Label, etc..
    """

    # Need to scale images for display; images for processing are left raw.
    #   Default --scale arg is 1.0, so no scaling when option in not used.
    scaled_img = scale(image, arguments()['scale'])

    # based on tutorial: https://pyimagesearch.com/2016/05/23/opencv-with-tkinter/
    scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
    scaled_img = Image.fromarray(scaled_img)
    tk_img = ImageTk.PhotoImage(scaled_img)
    # Need to prevent garbage collection to show image in tk.Label, etc.
    tk_img.image = tk_img

    return tk_img


def ttk_styles(mainloop: tkinter.Tk) -> None:
    """
    Configure platform-specific ttk.Style for Buttons and Comboboxes.
    Font and color values need to be edited as appropriate for the
    application (to avoid lengthy parameter arguments).

    Args:
         mainloop: The tk.Toplevel running as the mainloop.

    Returns: None
    """

    ttk.Style().theme_use('alt')

    # Use fancy buttons for Linux;
    #   standard theme for Windows and macOS, but with custom font.
    bstyle = ttk.Style()
    combo_style = ttk.Style()

    if const.MY_OS == 'lin':
        font_size = 8
    elif const.MY_OS == 'win':
        font_size = 7
    else:  # is macOS
        font_size = 11

    bstyle.configure("My.TButton", font=('TkTooltipFont', font_size))
    mainloop.option_add("*TCombobox*Font", ('TkTooltipFont', font_size))

    if const.MY_OS == 'lin':
        bstyle.map("My.TButton",
                   foreground=[('active', const.CBLIND_COLOR_TK['yellow'])],
                   background=[('pressed', 'gray30'),
                               ('active', const.CBLIND_COLOR_TK['vermilion'])],
                   )
        combo_style.map('TCombobox',
                        fieldbackground=[('readonly',
                                          const.CBLIND_COLOR_TK['dark blue'])],
                        selectbackground=[('readonly',
                                           const.CBLIND_COLOR_TK['dark blue'])],
                        selectforeround=[('readonly',
                                          const.CBLIND_COLOR_TK['yellow'])],
                        )
    elif const.MY_OS == 'win':
        bstyle.map("My.TButton",
                   foreground=[('active', const.CBLIND_COLOR_TK['yellow'])],
                   background=[('pressed', 'gray30'),
                               ('active', const.CBLIND_COLOR_TK['vermilion'])],
                   )
