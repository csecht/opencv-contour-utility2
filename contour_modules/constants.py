"""
Constants used throughout main script and modules:
ALPHA_MAX
BETA_MAX
CBLIND_COLOR_CV
CBLIND_COLOR_TK
CENTER_XSCALE
COLOR_BOUNDARIES
CONTOUR_METHOD
CONTOUR_MODE
CV_BORDER
CV_FILTER
CV_MORPH_SHAPE
CV_MORPHOP
DARK_BG
DRAG_GRAY
FONT_SCALE
FONT_TYPE
LABEL_PARAMETERS
LINE_SCALE
MASTER_BG
MY_OS
PANEL_LEFT
PANEL_RIGHT
RADIO_PARAMETERS
SCALE_PARAMETERS
SHAPE_SCALE_PARAMETERS
SHAPE_VERTICES
STD_CONTOUR_COLOR
STUB_ARRAY
TEXT_COLOR
TEXT_THICKNESS
THRESH_TYPE
WIDGET_FG
WIN_NAME
"""
# Copyright (C) 2023 C.S. Echt, under GNU General Public License'

# Standard library import
import sys
import numpy as np

# Third party import
import cv2

MY_OS = sys.platform[:3]

STUB_ARRAY = np.ones((5, 5), 'uint8')

# NOTE: keys here must match corresponding keys in contour_it.py
#   img_window dict.
WIN_NAME = {
    'input': 'Input <- | -> Grayscale for processing',
    'contrasted': 'Adjusted contrast <- | -> Reduced noise',
    'filtered': 'Filtered image',
    'thresholded': 'Threshold <- | -> Selected Threshold contours',
    'th+cnt&hull': 'Threshold <- | -> Contours (yellow), hulls (blue)',
    'thresh sized': 'Threshold objects, with relative px sizes',
    'canny sized': 'Canny edged objects, with relative px sizes',
    'canny': 'Edges <- | -> Selected Canny contours',
    'shaped': 'Shapes found',
    'shape_report': 'Shape Settings Report',
    'thresh shaped': 'Shapes found in threshold contours',
    'canny shaped': 'Shapes found in Canny edge contours',
    'circle in filtered': 'Circles in the filtered image',
    'circle in thresh': 'Circles in an Otsu threshold image',
    'clahe': 'CLAHE adjusted',
    'histo': 'Histograms',
    'input2color': 'Input <- | -> Elements in selected color range',
}

# Set ranges for trackbars used to adjust contrast and brightness for
#  the cv2.convertScaleAbs method.
ALPHA_MAX = 400
BETA_MAX = 254  # Provides a range of [-127 -- 127].

# CV dict values are cv2 constants' (key) returned integers.
# Some of these dictionaries are used only to populate Combobox lists.
CV_BORDER = {
    'cv2.BORDER_REFLECT_101': 4,  # is same as cv2.BORDER_DEFAULT.
    'cv2.BORDER_REFLECT': 2,
    'cv2.BORDER_REPLICATE': 1,
    'cv2.BORDER_ISOLATED': 16,
}

THRESH_TYPE = {
    # Note: Can mimic inverse types by adjusting alpha and beta channels.
    # Note: THRESH_BINARY* is used with cv2.adaptiveThreshold, which is
    #  not implemented here.
    'cv2.THRESH_BINARY': 0,
    'cv2.THRESH_BINARY_INVERSE': 1,
    'cv2.THRESH_OTSU': 8,
    'cv2.THRESH_OTSU_INVERSE': 9,
    'cv2.THRESH_TRIANGLE': 16,
    'cv2.THRESH_TRIANGLE_INVERSE': 17,
}

CV_MORPHOP = {
    'cv2.MORPH_OPEN': 2,
    'cv2.MORPH_CLOSE': 3,
    'cv2.MORPH_GRADIENT': 4,
    'cv2.MORPH_BLACKHAT': 6,
    'cv2.MORPH_HITMISS': 7,
}

CV_MORPH_SHAPE = {
    'cv2.MORPH_RECT': 0,  # cv2 default
    'cv2.MORPH_CROSS': 1,
    'cv2.MORPH_ELLIPSE': 2,
}

CV_FILTER = {
    'cv2.blur': 0,  # cv2 default
    'cv2.bilateralFilter': 1,
    'cv2.GaussianBlur': 2,
    'cv2.medianBlur': 3,
    'Convolve laplace': None,
    'Convolve outline': None,
    'Convolve sharpen': None,
}

CONTOUR_MODE = {
    'cv2.RETR_EXTERNAL': 0,  # cv2 default
    'cv2.RETR_LIST': 1,
    'cv2.RETR_CCOMP': 2,
    'cv2.RETR_TREE': 3,
    'cv2.RETR_FLOODFILL': 4,
}

# from: https://docs.opencv.org/4.4.0/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff
# CHAIN_APPROX_NONE
# stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.
# CHAIN_APPROX_SIMPLE
# compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.
# CHAIN_APPROX_TC89_L1
# applies one of the flavors of the Teh-Chin chain approximation algorithm [229]
# CHAIN_APPROX_TC89_KCOS
# applies one of the flavors of the Teh-Chin chain approximation algorithm [229]
CONTOUR_METHOD = {
    'cv2.CHAIN_APPROX_NONE': 1,
    'cv2.CHAIN_APPROX_SIMPLE': 2,
    'cv2.CHAIN_APPROX_TC89_L1': 3,
    'cv2.CHAIN_APPROX_TC89_KCOS': 4
}

SHAPE_VERTICES = {
    'Triangle': 3,
    'Rectangle': 4,
    'Pentagon': 5,
    'Hexagon': 6,
    'Heptagon': 7,
    'Octagon': 8,
    '5-pointed Star': 10,
    'Circle': 0,
}

"""
Colorblind color pallet source:
  Wong, B. Points of view: Color blindness. Nat Methods 8, 441 (2011).
  https://doi.org/10.1038/nmeth.1618
Hex values source: https://www.rgbtohex.net/
See also: https://matplotlib.org/stable/tutorials/colors/colormaps.html
"""
# OpenCV uses a BGR (B, G, R) color convention, instead of RGB.
CBLIND_COLOR_CV = {
    'blue': (178, 114, 0),
    'orange': (0, 159, 230),
    'sky blue': (233, 180, 86),
    'blueish green': (115, 158, 0),
    'vermilion': (0, 94, 213),
    'reddish purple': (167, 121, 204),
    'yellow': (66, 228, 240),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}

CBLIND_COLOR_TK = {
    'blue': '#0072B2',
    'dark blue': 'MidnightBlue',
    'orange': '#E69F00',
    'sky blue': '#56B4E9',
    'blueish green': '#009E73',
    'vermilion': '#D55E00',
    'reddish purple': '#CC79A7',
    'yellow': '#F0E442',
    'black': 'black',
    'white': 'white',
    'tk_white': '',  # system's default, conditional on MY_OS
}

# Need tk to match system's default white shade.
if MY_OS == 'dar':  # macOS
    CBLIND_COLOR_TK['tk_white'] = 'white'
elif MY_OS == 'lin':  # Linux (Ubuntu)
    CBLIND_COLOR_TK['tk_white'] = 'grey85'
else:  # platform is 'win'  # Windows (10)
    CBLIND_COLOR_TK['tk_white'] = 'grey95'

STD_CONTOUR_COLOR = {'green': (0, 255, 0)}

# 	cv::HersheyFonts {
#   cv::FONT_HERSHEY_SIMPLEX = 0, # cv2 default
#   cv::FONT_HERSHEY_PLAIN = 1,
#   cv::FONT_HERSHEY_DUPLEX = 2,
#   cv::FONT_HERSHEY_COMPLEX = 3,
#   cv::FONT_HERSHEY_TRIPLEX = 4,
#   cv::FONT_HERSHEY_COMPLEX_SMALL = 5,
#   cv::FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
#   cv::FONT_HERSHEY_SCRIPT_COMPLEX = 7,
#   cv::FONT_ITALIC = 16
# }

# Need to adjust text length across platform's for the cv2.getTextSize()
# function used in utils.text_array() module.
# https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_HersheyFonts.html
FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX

TEXT_THICKNESS = 1
TEXT_COLOR = 180, 180, 180  # light gray for a dark gray background

# Scaling factors for contours, circles and text, empirically determined;
#  used in manage_input().
LINE_SCALE = 5e-04  # 1e-03
FONT_SCALE = 5.5e-04  # 7.7e-04

# Used in manage.input_metrics() module.
CENTER_XSCALE = 0.035

if MY_OS == 'lin':
    WIDGET_FONT = 'TkTooltipFont', 8
    radio_params = dict(
        fg=CBLIND_COLOR_TK['yellow'],
        activebackground='gray50',  # Default is 'white'.
        activeforeground=CBLIND_COLOR_TK['sky blue'],  # Default is 'black'.
        selectcolor=CBLIND_COLOR_TK['dark blue'])
elif MY_OS == 'win':
    WIDGET_FONT = 'TkTooltipFont', 8
    radio_params = dict(fg='black')
else:  # is macOS
    WIDGET_FONT = 'TkTooltipFont', 9
    radio_params = dict(fg='black')

MASTER_BG = 'gray80'
DARK_BG = 'gray20'
DRAG_GRAY = 'gray65'
WIDGET_FG = CBLIND_COLOR_TK['yellow']

LABEL_PARAMETERS = dict(
    font=WIDGET_FONT,
    bg=DARK_BG,
    fg=WIDGET_FG,
)

SCALE_PARAMETERS = dict(
    width=10,
    orient='horizontal',
    showvalue=False,
    sliderlength=20,
    font=WIDGET_FONT,
    bg=CBLIND_COLOR_TK['dark blue'],
    fg=WIDGET_FG,
    troughcolor=MASTER_BG,
)

RADIO_PARAMETERS = dict(
    font=WIDGET_FONT,
    bg='gray50',
    bd=2,
    indicatoron=False,
    **radio_params  # are OS-specific.
)

# Here 'font' sets the shown value; font in the pull-down values
#   is set by option_add in ContourViewer.setup_styles()
if MY_OS == 'lin':
    COMBO_PARAMETERS = dict(
        font=WIDGET_FONT,
        foreground=CBLIND_COLOR_TK['yellow'],
        takefocus=False,
        state='readonly')
elif MY_OS == 'win':  # is Windows
    COMBO_PARAMETERS = dict(
        font=('TkTooltipFont', 7),
        takefocus=False,
        state='readonly')
else:  # is macOS
    COMBO_PARAMETERS = dict(
        font=('TkTooltipFont', 9),
        takefocus=False,
        state='readonly')

# Grid arguments to place Label images in image windows.
PANEL_LEFT = dict(
            column=0, row=0,
            padx=5, pady=5,
            sticky='nsew')
PANEL_RIGHT = dict(
            column=1, row=0,
            padx=5, pady=5,
            sticky='nsew')

# Sets of BGR color values for converting to HSV color ranges.
COLOR_BOUNDARIES = {
    'red & brown': ((0, 100, 100), (6, 255, 255)),  # Also used for HSV red><red.
    'red><red': ((0, 0, 0), (0, 0, 0)),  # Stub values: see find_colors().
    'red & hot pink': ((165, 100, 100), (180, 255, 255)),  # Also used for HSV red><red.
    'pink': ((155, 50, 120), (180, 255, 255)),
    'deep pink': ((155, 200, 0), (180, 255, 255)),
    'purple': ((130, 120, 160), (160, 240, 240)),
    'orange': ((5, 190, 200), (18, 255, 255)),
    'orange2': ((3, 102, 102), (20, 255, 255)),
    'yellow': ((20, 70, 80), (30, 255, 255)),
    'green': ((36, 25, 25), (70, 255, 255)),
    'green & cyan': ((50, 20, 20), (100, 255, 255)),
    'blue': ((110, 150, 50), (120, 255, 255)),  # Deep true blues.
    'blue & cyan': ((80, 50, 45), (120, 255, 255)),
    'royal blue': ((105, 150, 0), (130, 200, 255)),
    'royal & slate': ((105, 50, 50), (130, 200, 255)),
    'vivid': ((0, 220, 90), (150, 255, 255)),
    'vivid2': ((0, 175, 150), (180, 255, 255)),
    'earthtones': ((0, 14, 80), (120, 120, 225)),
    'whites': ((0, 0, 200), (125, 60, 255)),  # Includes some red & blue whites.
    'mid grays': ((0, 0, 55), (0, 0, 200)),
    'lt grays & white': ((0, 0, 200), (0, 0, 255)),
}
