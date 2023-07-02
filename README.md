# Project: opencv-contour-util2

| ![canny_crane.png](images/canny_crane.png) | ![canny_crane_contours.png](images/canny_crane_contours.png) |
| ------------- | ------------- |

A set of Python programs that use a tkinter GUI to explore many OpenCV parameters to draw image contours and identify objects, find specific shapes, or mask colors. Parameter values can be selected using slider bars, drop down menus, and button toggles (see screenshots, below). Multiple windows are opened that display live updates as individual parameters change for each image processing step involved in object or shape detection.

The intention of this utility is to help OpenCV users understand the relevant parameters and their value ranges that may be needed to find contours, and identify objects, shapes, and colors.

Programs that run from the command line: 
* `contour_it.py` draws contours based on edges, thresholds, or shapes.
* `equalize_it.py` does CLAHE histogram equalization.
* `color_it.py` finds colors.
* 
All modules can be executed on Linux, Windows, and macOS platforms. `contour_it.py` is an upgrade of the original module found in the opencv-contour-utils repository that uses the native OpenCV GUI, but only runs on Linux systems.

All contour processing steps are conducted on grayscale conversions of the specified input file. A text file of chosen settings and the resulting image file of drawn contours, overlaid on the original image, can be saved. Image file samples, listed below, are provided in the `images` folder.

The default `contour_it.py` contour outline color is green, but can be changed to yellow with the `--color yellow` command line argument. This may be useful for certain images or users with a green color vision impairment.

<sub>Project inspired by code from Adrian Rosebrock:
https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
</sub>

Development environment was Linux Ubuntu 20.04, macOS 13.2, and Windows11.

## contour_it.py
The module `contour_it.py` uses cv2.threshold and cv2.Canny for contour detection.
It uses cv2.approxPolyDP and cv2.HoughCircles for shape identification. See the startup window screenshot, below, for all cv2 functions and parameters used.

## equalize_it.py
The module `equalize_it.py` does not involve contours, but explores parameters for automatic histogram equalization. It can be used as an optional pre-processing step for contour detection. Equalization is by CLAHE, a contrast-limited adaptive histogram equalization method. Live updates of the CLAHE histogram are controlled by slide bars for the clipLimit and tileGridSize parameters of cv2.createCLAHE. All processing is carried out on a grayscale of the input file. The grayscale equalized image can be saved to use as input for `equalize_it.py`. For most contour needs, however, the contrast and brightness controls in `equalize_it.py` should be sufficient.

Be aware that CLAHE works best on images that have a more-or-less continuous range of pixel values across the image, as in sample2.jpg (shells). Hence, it does not work well with images that have large areas of similar colors, as in sample1.jpg (pills), sample3.jpg (rummikub), or sample4.jgp (shapes). Nonetheless, this module can be fun to play with. Live histogram updating may be less responsive with larger image files.

Note that the `--color` command line argument does nothing with `equalize_it.py`.

## color_it.py
The module `color_it.py` does not involve contours, but explores HSV colorspace using: `cv2.cvtColor(src, cv2.COLOR_BGR2HSV)` for colorspace conversion, `cv2.inRange` for HSV thresholding, and `cv2.bitwise_and` for masking. Colors can be adjusted using slide bars for the upper and lower bounds to set the HSV color threshold. Sliders use a 0-255 scale of BGR values (screenshot below), which are converted to HSV values. A pull-down menu of pre-selected colors is available. These pre-set colors and color groups can be used as a starting point for subsequent optimization with the sliders.

There are button toggles to apply or remove a blurring filter for the input image and noise reduction for the color mask. These operations, while adjustable in `contour_it.py`, are hard-coded here, but generally work well for color masking. The applied filter and noise reduction methods and parameters are posted in the report window. 

Selected color values and the resulting color-masked image can be saved to files.

Note that the `--color` command line argument does nothing with `color_it.py`. Strange, but true.

### Usage examples:
From within the program's folder, example command line invocations:

       python3 -m contour_it  (uses a default input image)

       python3 -m contour_it --input images/sample2.jpg

       python3 -m contour_it -i /home/me/Desktop/myphoto.jpg --scale 0.2

       python3 -m equalize_it --i images/sample2.jpg -s 0.5

Note that with no input argument, as in the first example, the default sample1.jpg from the `images` folder is used for input. Additional sample input files are provided in the `images` folder.

On Windows systems, you may need to replace 'python3' with 'python' or 'py'.

Be aware that very large image file may require a few seconds to display the program widows, depending on your system performance. Be patient.

To list command line options: `python3 -m contour_it --help`
       
       Explore Image Processing Parameters.
       optional arguments:
         -h, --help            show this help message and exit
         --about               Provide description, version, GNU license.
         --input PATH/FILE, -i PATH/FILE
                               Path to input image (PNG or JPG file).
         --scale X, -s X       Factor, X, to change displayed image size (default: 1.0).
         --color C, -c C       cv2 contour color, C. (default: green; option: yellow).


To view basic information, author, version, license, etc.: `python3 -m contour_it --about`
 
The Esc or Q key will quit any running module.

The displayed image size can be adjusted with the `--scale` command line argument. All image processing, however, is performed at the original image resolution.

Image file examples provided in the `images` folder:
* sample1.jpg (pills, 800x600 692 kB),
* sample2.jpg (shells, 1050x750, 438 kB),
* sample3.jpg (rummikub, 4032x3024, 2.94 MB)
* sample4.jpg (shapes, 1245x1532, 137 kB)
* sample5.png (X11 RGB color table, 1210x667, 497 kB)

### Requirements:
Python 3.7 or later, plus the packages OpenCV, Numpy and tkinter.

For quick installation of the required Python PIP packages:
from the downloaded GitHub repository folder, run this command

    pip install -r requirements.txt

Alternative commands (system dependent):

    python3 -m pip install -r requirements.txt (Linux and macOS)
    py -m pip install -r requirements.txt (Windows)
    python -m pip install -r requirements.txt (Windows)

As with all repository downloads, it is a good idea to install the requirements in a Python virtual environment to avoid undesired changes in your system's Python library.

### Screenshots:
All screenshots are from an Ubuntu Linux platform. For Windows and macOS platforms, window and widget look or layout may be slightly different.

![contour_settings_window](images/contour_report_window.png)

Default startup window for contour parameter settings and reporting. Command line: `python3 -m contour_it`.

![ID_image_windows](images/all_image_windows.png)

All image windows that open to display each image processing step. Initial window layout will depend on your system. Here, the windows were dragged into position to show them all without overlap. Command line: `python3 -m contour_it  -i images/sample2.jpg -s 0.3`.

![shapes_settings_and_report](images/shape_report_window.png)
![hexagon_shape_found](images/found_hexagon_screenshot.png)

The Shape windows appear when the "Show Shapes windows" is clicked. In this example, settings selected to find hexagon shapes. Command line: `python3 -m contour_it  -i images/sample4.jpg -s 0.3 --color yellow`.

![CLAHE_windows](images/CLAHE_screenshot.png)

The windows, manually rearranged, showing default settings for the two CLAHE parameters. Command line: `python3 -m equalize_it -i images/sample2.jpg -s 0.4`

![color_settings](images/color_settings_screenshot.png)

Above: The report and settings window for `color_it.py` at default settings. 
Command line: `python3 -m color_it -i images/sample5.png -s 0.5`. 

Below: The input and processed images.
![colors_scrnshot.png](images/colors_scrnshot.png)

Pre-set BGR values, to be converted to HSV color bounds, for color selections in the "Select a color..." pull-down menu. (Values are from the COLOR_BOUNDARIES dictionary in contour_modules/constants.py.)

| color choice     | lower bound     | upper bound     | Notes                          |
|------------------|-----------------|-----------------|--------------------------------|
| red & brown      | (0, 100, 100)   | (6, 255, 255)   | Used for HSV red><red.         |
| red><red         | (0, 0, 0)       | (0, 0, 0)       | Stub values; masks are joined. |
| red & deep pink  | (170, 100, 100) | (180, 255, 255) | Used for HSV red><red.         |
| pink             | (155, 50, 120)  | (180, 255, 255) |
| deep pink        | (155, 200, 0)   | (180, 255, 255) |
| purple           | (130, 120, 160) | (160, 240, 240) |
| orange           | (5, 190, 200)   | (18, 255, 255)  |
| orange2          | (3, 102, 102)   | (20, 255, 255)  |
| yellow           | (20, 70, 80)    | (30, 255, 255)  |
| green            | (36, 25, 25)    | (70, 255, 255)  |
| green & cyan     | (50, 20, 20)    | (100, 255, 255) |
| blue             | (110, 150, 50)  | (120, 255, 255) | Deep true blues.               |
| blue & cyan      | (80, 50, 45)    | (120, 255, 255) |
| royal blue       | (105, 150, 0)   | (130, 200, 255) |
| royal & slate    | (105, 50, 50)   | (130, 200, 255) |
| vivid            | (0, 220, 90)    | (150, 255, 255) |
| vivid2           | (0, 175, 150)   | (180, 255, 255) |
| earthtones       | (0, 14, 80)     | (120, 120, 225) |
| whites           | (0, 0, 200)     | (125, 60, 255)  | Includes pale red & blue       |
| mid grays        | (0, 0, 55)      | (0, 0, 200)     |
| lt grays & white | (0, 0, 200)     | (0, 0, 255)     |

### Known Issues:

While not a program issue, there is a potential source of confusion when using the example image, sample4.jpg (shapes). With the default settings, the white border around the black background will display a hexagon-shaped contour, which may be difficult to see, especially when using yellow --color option. Consequently, it will be counted as a hexagon shape unless, in main settings, it is not displayed as a contour by clicking the button for cv2.arcLength button instead of cv2.contourArea.

Widths and grid spacing were generalized across platforms for settings/report windows, but may not be optimal for your particular setup. If a settings window seems too wide or too narrow, you can horizontally resize it for a better layout.

### Attributions

Source of sample1.jpg image file:
Adrian Rosebrock at https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/

Source of sample2.jpg image file:
http://sunrisekauai.blogspot.com/2012/06/new-group-of-sunrise-shells.html

All other image files are from the author, C.S. Echt. The sample5.png color table was generated by tk_color_helper.py from https://github.com/csecht/tkinter_utilities
