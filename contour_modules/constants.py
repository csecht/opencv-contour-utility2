
import cv2
import sys

MY_OS = sys.platform[:3]

SHAPE_NAME = {
    3: 'triangle',
    4: 'rectangle',
    5: 'pentagon',
    6: 'hexagon',
    7: 'heptagon',
    8: 'octagon',
    9: 'nonagon',
    10: 'star',
    11: 'circle',
}

WIN_NAME = {
    'input+gray': 'Input <- | -> Grayscale for processing',
    'contrast+redux': 'Adjusted contrast <- | -> Reduced noise',
    'filtered': 'Filtered image',
    'th+contours': 'Threshold <- | -> Selected Threshold contours',
    'th+cnt&hull': 'Threshold <- | -> Contours (yellow), hulls (blue)',
    'thresh sized': 'Threshold objects, with relative px sizes',
    'canny sized': 'Canny edged objects, with relative px sizes',
    'canny+contours': 'Edges <- | -> Selected Canny contours',
    'shapes': 'Shaped found',
    'shape_report': 'Shape Report Settings',
    'shape report_th': 'Shape Report, using threshold contours',
    'shape report_can': 'Shape Report, using Canny contours',
    'thresh': 'Shapes found in threshold contours',
    'canny': 'Shapes found in Canny edge contours',
    'circle in filtered': 'Circles in the filtered image',
    'circle in thresh': 'Circles in an Otsu threshold image',
    'clahe': 'CLAHE adjusted'
}

# Set ranges for trackbars used to adjust contrast and brightness for
#  the cv2.convertScaleAbs method.
ALPHA_MAX = 400
BETA_MAX = 254  # Provides a range of [-127 -- 127].

CV_BORDER = {
    # Value of cv2.BORDER_* returns an integer.
    'cv2.BORDER_REFLECT_101': 4,  # is same as cv2.BORDER_DEFAULT.
    'cv2.BORDER_REFLECT': 2,
    'cv2.BORDER_REPLICATE': 1,
    'cv2.BORDER_ISOLATED': 16,
}

TH_TYPE = {
    # Note: Can mimic inverse types by adjusting alpha and beta channels.
    # Value is the cv2 constant's returned integer.
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
    # Value is the cv2 constant's returned integer.
    'cv2.MORPH_OPEN': 2,
    'cv2.MORPH_CLOSE': 3,
    'cv2.MORPH_GRADIENT': 4,
    'cv2.MORPH_BLACKHAT': 6,
    'cv2.MORPH_HITMISS': 7,
}

CV_MORPH_SHAPE = {
    # Value is the cv2 constant's returned integer.
    'cv2.MORPH_RECT': 0,  # (default)
    'cv2.MORPH_CROSS': 1,
    'cv2.MORPH_ELLIPSE': 2,
}

CONTOUR_MODE = {
    # Value is the cv2 constant's returned integer.
    'cv2.RETR_EXTERNAL': 0,
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
    # Value is the cv2 constant's returned integer.
    'cv2.CHAIN_APPROX_NONE': 1,
    'cv2.CHAIN_APPROX_SIMPLE': 2,
    'cv2.CHAIN_APPROX_TC89_L1': 3,
    'cv2.CHAIN_APPROX_TC89_KCOS': 4
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
#   cv::FONT_HERSHEY_SIMPLEX = 0,
#   cv::FONT_HERSHEY_PLAIN = 1,
#   cv::FONT_HERSHEY_DUPLEX = 2,
#   cv::FONT_HERSHEY_COMPLEX = 3,
#   cv::FONT_HERSHEY_TRIPLEX = 4,
#   cv::FONT_HERSHEY_COMPLEX_SMALL = 5,
#   cv::FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
#   cv::FONT_HERSHEY_SCRIPT_COMPLEX = 7,
#   cv::FONT_ITALIC = 16
# }
# https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_HersheyFonts.html
FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX

if MY_OS == 'lin':
    TEXT_SCALER = 0.5
elif MY_OS == 'dar':
    TEXT_SCALER = 0.4
else:  # is Windows
    TEXT_SCALER = 0.6

TEXT_THICKNESS = 1
TEXT_COLOR = 180, 180, 180  # light gray for a dark gray background

# Scaling factors for contours, circles and text, empirically determined;
#  used in manage_input().
LINE_SCALE = 1e-03
FONT_SCALE = 7.7e-04
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
widget_fg = CBLIND_COLOR_TK['yellow']

LABEL_PARAMETERS = dict(
    font=WIDGET_FONT,
    bg=DARK_BG,
    fg=widget_fg,
)

if MY_OS == 'lin':
    scale_len = 400
    shape_scale_len = 400
elif MY_OS == 'dar':
    scale_len = 400
    shape_scale_len = 320
else:  # is Windows
    scale_len = 450
    shape_scale_len = 450

SCALE_PARAMETERS = dict(
    length=scale_len,
    width=10,
    orient='horizontal',
    showvalue=False,
    sliderlength=20,
    font=WIDGET_FONT,
    bg=CBLIND_COLOR_TK['dark blue'],
    fg=widget_fg,
    troughcolor=MASTER_BG,
)

SHAPE_SCALE_PARAMS = dict(
    length=shape_scale_len,
    width=10,
    orient='horizontal',
    showvalue=False,
    sliderlength=20,
    font=WIDGET_FONT,
    bg=CBLIND_COLOR_TK['dark blue'],
    fg=widget_fg,
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
        # takefocus=False,
        state='readonly')
else:  # is macOS
    COMBO_PARAMETERS = dict(
        font=('TkTooltipFont', 9),
        takefocus=False,
        state='readonly')
