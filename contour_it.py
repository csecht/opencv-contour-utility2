#!/usr/bin/env python3
"""
Use a tkinter GUI to explore OpenCV image processing parameters that
are involved in identifying objects and drawing contours.
Parameter values are adjusted with slide bars, pull-down menus, and
button toggles.

USAGE Example command lines, from within the image-processor-main folder:
python3 -m contour_it --help
python3 -m contour_it --about
python3 -m contour_it --input images/sample1.jpg
python3 -m contour_it -i images/sample2.jpg

Windows systems may need to substitute 'python3' with 'py' or 'python'.

Quit program with Esc key, Ctrl-Q key, the close window icon of the
settings windows, or from command line with Ctrl-C.
Save settings and the contoured image with Save button.

Requires Python3.7 or later and the packages opencv-python and numpy.
See this distribution's requirements.txt file for details.
Developed in Python 3.8-3.9.
"""

# Copyright (C) 2022-2023 C.S. Echt, under GNU General Public License

# Standard library imports
import sys
from pathlib import Path

# Local application imports
from contour_modules import (vcheck,
                             utils,
                             manage,
                             constants as const,
                             )

# Third party imports.
# tkinter(Tk / Tcl) is included with most Python3 distributions,
#  but may sometimes need to be regarded as third-party.
try:
    import cv2
    import numpy as np
    import tkinter as tk
    from tkinter import ttk
except (ImportError, ModuleNotFoundError) as import_err:
    sys.exit(
        '*** One or more required Python packages were not found'
        ' or need an update:\nOpenCV-Python, NumPy, tkinter (Tk/Tcl).\n\n'
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
    A suite of OpenCV methods for applying various image processing
    functions involved in identifying objects from an image file.
    OpenCV's methods used: cv2.convertScaleAbs, cv2.getStructuringElement,
    cv2.morphologyEx, cv2 filters, cv2.threshold, cv2.Canny,
    cv2.findContours, cv2.contourArea,cv2.arcLength, cv2.drawContours,
    cv2.minEnclosingCircle.
    Class methods:
        adjust_contrast
        reduce_noise
        filter_image
        contour_threshold
        contour_canny
        size_the_contours
    """

    __slots__ = ('tk',
                 'canny_lbl', 'cbox_val', 'circled_can_lbl',
                 'circled_th_lbl', 'computed_threshold',
                 'contour_color', 'contour_limit', 'contours',
                 'contrast_lbl', 'curr_contrast_std',
                 'edge_contour_lbl', 'filter_lbl', 'filtered_img',
                 'gray_lbl', 'img_window', 'input_contrast_std',
                 'input_lbl', 'num_contours', 'radio_val',
                 'reduced_noise_img', 'redux_lbl', 'sigma_color',
                 'sigma_space', 'sigma_x', 'sigma_y', 'slider_val',
                 'th_contour_lbl', 'thresh_lbl', 'tkimg',
                 )

    def __init__(self):
        super().__init__()

        # Note: The matching selector widgets for the following 15
        #  control variables are in ContourViewer __init__.
        self.slider_val = {
            'alpha': tk.DoubleVar(),
            'beta': tk.IntVar(),
            'noise_k': tk.IntVar(),
            'noise_iter': tk.IntVar(),
            'filter_k': tk.IntVar(),
            'canny_th_ratio': tk.DoubleVar(),
            'canny_th_min': tk.IntVar(),
            'c_limit': tk.IntVar(),
        }
        self.cbox_val = {
            'morphop_pref': tk.StringVar(),
            'morphshape_pref': tk.StringVar(),
            'border_pref': tk.StringVar(),
            'filter_pref': tk.StringVar(),
            'th_type_pref': tk.StringVar(),
            'c_method_pref': tk.StringVar(),
        }
        self.radio_val = {
            'c_mode_pref': tk.StringVar(),
            'c_type_pref': tk.StringVar(),
            'hull_pref': tk.BooleanVar(),
        }

        self.input_contrast_std = tk.DoubleVar()
        self.curr_contrast_std = tk.DoubleVar()

        # Arrays of images to be processed. When used within a method,
        #  the purpose of self.tkimg* is to prevent losing the image var
        #  through garbage collection. These are for panels of PIL
        #  ImageTk.PhotoImage used for Label image display in windows.
        self.tkimg = {
            'input': None,
            'gray': None,
            'contrast': None,
            'redux': None,
            'filter': None,
            'thresh': None,
            'canny': None,
            'drawn_th': None,
            'drawn_can': None,
            'circled_th': None,
            'circled_can': None
        }

        # Contour lists populated with cv2.findContours point sets.
        stub_array = np.ones((5, 5), 'uint8')
        self.contours = {
            'drawn_thresh': stub_array,
            'drawn_canny': stub_array,
            'selected_thresh': [],
            'selected_canny': [],
        }

        self.num_contours = {
            'th_all': tk.IntVar(),
            'th_select': tk.IntVar(),
            'canny_all': tk.IntVar(),
            'canny_select': tk.IntVar(),
        }

        self.filtered_img = None
        self.reduced_noise_img = None

        # Image processing parameters.
        self.sigma_color = 1
        self.sigma_space = 1
        self.sigma_x = 1
        self.sigma_y = 1
        self.computed_threshold = 0
        self.contour_limit = 0

        # Tk windows are assigned here, but are titled in
        #  CoutourViewer.setup_image_windows().
        self.img_window = {
            'input': tk.Toplevel(),
            'contrasted': tk.Toplevel(),
            'filtered': tk.Toplevel(),
            'thresholded': tk.Toplevel(),
            'canny': tk.Toplevel(),
            'thresh sized': tk.Toplevel(),
            'canny sized': tk.Toplevel(),
        }
        # # Sized windows are deiconified in size_the_contours().
        # self.img_window['thresh sized'].withdraw()
        # self.img_window['canny sized'].withdraw()

        # The Labels to display scaled images, which are updated using
        #  .configure() for 'image=' in their respective processing methods.
        self.input_lbl = tk.Label(self.img_window['input'])
        self.gray_lbl = tk.Label(self.img_window['input'])
        self.contrast_lbl = tk.Label(self.img_window['contrasted'])
        self.redux_lbl = tk.Label(self.img_window['contrasted'])
        self.filter_lbl = tk.Label(self.img_window['filtered'])
        self.thresh_lbl = tk.Label(self.img_window['thresholded'])
        self.th_contour_lbl = tk.Label(self.img_window['thresholded'])
        self.canny_lbl = tk.Label(self.img_window['canny'])
        self.edge_contour_lbl = tk.Label(self.img_window['canny'])
        self.circled_th_lbl = tk.Label(self.img_window['thresh sized'])
        self.circled_can_lbl = tk.Label(self.img_window['canny sized'])

        # (Statement is duplicated in ShapeViewer.__init__)
        if arguments['color'] == 'yellow':
            self.contour_color = const.CBLIND_COLOR_CV['yellow']
        else:  # is default CV2 contour color option, green, as (BGR).
            self.contour_color = arguments['color']

    def adjust_contrast(self) -> None:
        """
        Adjust contrast of the input GRAY_IMG image.
        Updates contrast and brightness via alpha and beta sliders.
        Calls reduce_noise. Displays contrasted and redux noise images.

        Returns: None
        """
        # Source concepts:
        # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
        # https://stackoverflow.com/questions/39308030/
        #   how-do-i-increase-the-contrast-of-an-image-in-python-opencv

        contrasted = (
            cv2.convertScaleAbs(
                src=GRAY_IMG,
                alpha=self.slider_val['alpha'].get(),
                beta=self.slider_val['beta'].get())
        )

        self.input_contrast_std.set(int(np.std(GRAY_IMG)))
        self.curr_contrast_std.set(int(np.std(contrasted)))
        # Using .configure to update image avoids the white flash each time an
        #  image is updated were a Label() to be re-made here each call.
        self.tkimg['contrast'] = manage.tkimage(contrasted)
        self.contrast_lbl.configure(image=self.tkimg['contrast'])
        # Place the label panel to the left of the contrast label gridded
        #  in reduce_noise().
        self.contrast_lbl.grid(column=0, row=0,
                               padx=5, pady=5,
                               sticky=tk.NSEW)

        return contrasted

    def reduce_noise(self) -> np.ndarray:
        """
        Reduce noise in grayscale image with erode and dilate actions of
        cv2.morphologyEx.
        Uses cv2.getStructuringElement params shape=self.morphshape_val.
        Uses cv2.morphologyEx params op=self.morph_op,
        kernel=<local structuring element>, iterations=self.noise_iter,
        borderType=self.border_val.
        Called only by adjust_contrast().

        Returns:
             The array defined in adjust_contrast() with noise reduction.
        """

        # Need integers for the cv2 function parameters.
        morph_shape = const.CV_MORPH_SHAPE[self.cbox_val['morphshape_pref'].get()]

        # ksize value needs to be a tuple.
        kernel = (self.slider_val['noise_k'].get(),
                  self.slider_val['noise_k'].get())

        morph_op = const.CV_MORPHOP[self.cbox_val['morphop_pref'].get()]
        border_type = const.CV_BORDER[self.cbox_val['border_pref'].get()]
        iteration = self.slider_val['noise_iter'].get()

        # See: https://docs.opencv2.org/3.0-beta/modules/imgproc/doc/filtering.html
        #  on page, see: cv2.getStructuringElement(shape, ksize[, anchor])
        # see: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        element = cv2.getStructuringElement(
            shape=morph_shape,
            ksize=kernel)

        # Use morphologyEx as a shortcut for erosion followed by dilation.
        #   MORPH_OPEN is useful to remove noise and small features.
        #   MORPH_HITMISS helps to separate close objects by shrinking them.
        # Read https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
        # https://theailearner.com/tag/cv2-morphologyex/
        self.reduced_noise_img = cv2.morphologyEx(
            src=self.adjust_contrast(),
            op=morph_op,
            kernel=element,
            iterations=iteration,
            borderType=border_type
        )

        self.tkimg['redux'] = manage.tkimage(self.reduced_noise_img)
        self.redux_lbl.configure(image=self.tkimg['redux'])
        # Place the label panel to the right of the contrast label gridded
        #  in adjust_contrast().
        self.redux_lbl.grid(column=1, row=0,
                            padx=5, pady=5,
                            sticky=tk.NSEW)

        return self.reduced_noise_img

    def filter_image(self) -> np.ndarray:
        """
        Applies a filter selection to blur the image for Canny edge
        detection or threshold contouring.
        Called from contour_threshold().

        Returns: The filtered (blurred) image array processed by
                 reduce_noise().

        """

        filter_selected = self.cbox_val['filter_pref'].get()
        redux_img = self.reduce_noise()

        # Need to translate the string border type to that constant's integer.
        border_type = const.CV_BORDER[self.cbox_val['border_pref'].get()]

        # cv2.GaussianBlur and cv2.medianBlur need to have odd kernels,
        #   but cv2.blur and cv2.bilateralFilter will shift image between
        #   even and odd kernels so just make everything odd.
        got_k = self.slider_val['filter_k'].get()
        filter_k = got_k + 1 if got_k % 2 == 0 else got_k

        # Bilateral parameters:
        # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html
        # from doc: Sigma values: For simplicity, you can set the 2 sigma
        #  values to be the same. If they are small (< 10), the filter
        #  will not have much effect, whereas if they are large (> 150),
        #  they will have a very strong effect, making the image look "cartoonish".
        # NOTE: The larger the sigma the greater the effect of kernel size d.
        if self.reduced_noise_img is not None:
            self.sigma_color = int(np.std(self.reduced_noise_img))
        else:
            self.sigma_color = 3

        self.sigma_space = self.sigma_color

        # Gaussian parameters:
        # see: https://theailearner.com/tag/cv2-gaussianblur/
        self.sigma_x = self.sigma_color
        # NOTE: The larger the sigma the greater the effect of kernel size d.
        # sigmaY=0 also uses sigmaX. Matches Space to d if d>0.
        self.sigma_y = self.sigma_x

        # Apply a filter to blur edges:
        if filter_selected == 'cv2.bilateralFilter':
            self.filtered_img = cv2.bilateralFilter(src=self.reduced_noise_img,
                                                    # d=-1 or 0, is very CPU intensive.
                                                    d=filter_k,
                                                    sigmaColor=self.sigma_color,
                                                    sigmaSpace=self.sigma_space,
                                                    borderType=border_type)
        elif filter_selected == 'cv2.GaussianBlur':
            # see: https://dsp.stackexchange.com/questions/32273/
            #  how-to-get-rid-of-ripples-from-a-gradient-image-of-a-smoothed-image
            self.filtered_img = cv2.GaussianBlur(src=self.reduced_noise_img,
                                                 ksize=(filter_k, filter_k),
                                                 sigmaX=self.sigma_x,
                                                 sigmaY=self.sigma_y,
                                                 borderType=border_type)
        elif filter_selected == 'cv2.medianBlur':
            self.filtered_img = cv2.medianBlur(src=self.reduced_noise_img,
                                               ksize=filter_k)
        elif filter_selected == 'cv2.blur':
            self.filtered_img = cv2.blur(src=redux_img,
                                         ksize=(filter_k, filter_k),
                                         borderType=border_type)
        else:
            self.filtered_img = cv2.blur(src=self.reduced_noise_img,
                                         ksize=(filter_k, filter_k),
                                         borderType=border_type)

        self.tkimg['filter'] = manage.tkimage(self.filtered_img)
        self.filter_lbl.configure(image=self.tkimg['filter'])
        self.filter_lbl.grid(column=1, row=0,
                             padx=5, pady=5,
                             sticky=tk.NSEW)

        return self.filtered_img

    def contour_threshold(self, event=None) -> int:
        """
        Identify object contours with cv2.threshold() and
        cv2.drawContours(). Threshold types limited to Otsu and Triangle.
        Args:
            event: An implicit mouse button event.

        Returns: None
        """
        # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        # https://towardsdatascience.com/clahe-and-thresholding-in-python-3bf690303e40

        # Thresholding with OTSU works best with a blurring filter applied to
        #   image, like Gaussian or Bilateral
        # see: https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
        # https://theailearner.com/tag/cv2-thresh_otsu/
        th_type = const.TH_TYPE[self.cbox_val['th_type_pref'].get()]
        c_mode = const.CONTOUR_MODE[self.radio_val['c_mode_pref'].get()]
        c_method = const.CONTOUR_METHOD[self.cbox_val['c_method_pref'].get()]
        c_type = self.radio_val['c_type_pref'].get()
        c_limit = self.slider_val['c_limit'].get()
        line_thickness = infile_dict['line_thickness']

        # Set values to exclude threshold contours that may include
        #  contrasting borders on the image; an arbitrary 90% length
        #  limit, 81% area limit.
        # Note: On sample4, the 3-sided border is included in all settings, except
        #  when perimeter length is selected. IT IS NOT included in original thresh_it(?).
        #  It shouldn't work in thresh_it b/c the border contour is not the image area.
        #  So, it's Okay to use area, b/c is difficult to ID contours that are
        #    at or near an image border.
        max_area = GRAY_IMG.shape[0] * GRAY_IMG.shape[1] * 0.81
        max_length = max(GRAY_IMG.shape[0], GRAY_IMG.shape[1]) * 0.9

        # Note from doc: Currently, the Otsu's and Triangle methods
        #   are implemented only for 8-bit single-channel images.
        # OTSU & TRIANGLE compute thresh value, hence thresh=0 is replaced
        #   with the self.computed_threshold;
        #   for other cv2.THRESH_*, thresh needs to be manually provided.
        # Convert values above thresh to 255, white.
        self.computed_threshold, thresh_img = cv2.threshold(
            src=self.filter_image(),
            thresh=0,
            maxval=255,
            type=th_type)

        found_contours, _h = cv2.findContours(image=thresh_img,
                                              mode=c_mode,
                                              method=c_method)
        if c_type == 'cv2.contourArea':
            self.contours['selected_thresh'] = [
                _c for _c in found_contours
                if max_area > cv2.contourArea(_c) >= c_limit]
        else:  # c_type is cv2.arcLength; aka "perimeter"
            self.contours['selected_thresh'] = [
                _c for _c in found_contours
                if max_length > cv2.arcLength(_c, closed=False) >= c_limit]

        # Used only for reporting.
        self.num_contours['th_all'].set(len(found_contours))
        self.num_contours['th_select'].set(len(self.contours['selected_thresh']))

        contoured_img = INPUT_IMG.copy()

        # Draw hulls around selected contours when hull area is more than
        #   10% of contour area. This prevents obfuscation of drawn lines
        #   when hulls and contours are similar. 10% limit is arbitrary.
        if self.radio_val['hull_pref'].get():
            hull_list = []
            for i, _ in enumerate(self.contours['selected_thresh']):
                hull = cv2.convexHull(self.contours['selected_thresh'][i])
                if cv2.contourArea(hull) >= cv2.contourArea(
                        self.contours['selected_thresh'][i]) * 1.1:
                    hull_list.append(hull)

            cv2.drawContours(contoured_img,
                             contours=hull_list,
                             contourIdx=-1,  # all hulls.
                             color=const.CBLIND_COLOR_CV['sky blue'],
                             thickness=line_thickness * 3,
                             lineType=cv2.LINE_AA)

        # NOTE: drawn_thresh is what is saved with the 'Save' button.
        self.contours['drawn_thresh'] = cv2.drawContours(
            contoured_img,
            contours=self.contours['selected_thresh'],
            contourIdx=-1,  # all contours.
            color=self.contour_color,
            thickness=line_thickness * 2,
            lineType=cv2.LINE_AA)

        # Need to use self.*_img to keep attribute reference and thus
        #   prevent garbage collection.
        self.tkimg['thresh'] = manage.tkimage(thresh_img)
        self.thresh_lbl.configure(image=self.tkimg['thresh'])
        self.thresh_lbl.grid(column=0, row=0,
                             padx=5, pady=5,
                             sticky=tk.NSEW)

        self.tkimg['drawn_th'] = manage.tkimage(self.contours['drawn_thresh'])
        self.th_contour_lbl.configure(image=self.tkimg['drawn_th'])
        self.th_contour_lbl.grid(column=1, row=0,
                                 padx=5, pady=5,
                                 sticky=tk.NSEW)

        return th_type

    def contour_canny(self, event=None) -> None:
        """
        Identify objects with cv2.Canny() edges and cv2.drawContours().
        Args:
            event: An implicit mouse button event.

        Returns: None
        """

        # Source of coding ideas:
        # https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
        # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

        # Note: Much of this method is duplicated in contour_threshold();
        #   consider consolidating the two methods.

        # Canny recommended an upper:lower ratio between 2:1 and 3:1.
        canny_th_ratio = self.slider_val['canny_th_ratio'].get()
        canny_th_min = self.slider_val['canny_th_min'].get()
        canny_th_max = int(canny_th_min * canny_th_ratio)
        c_mode = const.CONTOUR_MODE[self.radio_val['c_mode_pref'].get()]
        c_method = const.CONTOUR_METHOD[self.cbox_val['c_method_pref'].get()]
        c_type = self.radio_val['c_type_pref'].get()
        c_limit = self.slider_val['c_limit'].get()
        line_thickness = infile_dict['line_thickness']

        # Set values to exclude threshold contours that may include
        #  contrasting borders on the image; an arbitrary 90% length
        #  limit, 81% area limit.
        # Note: On sample4, the 3-sided border is included in all settings, except
        #  when perimeter length is selected. IT IS NOT included in original thresh_it(?).
        #  It shouldn't work in thresh_it b/c the border contour is not the image area.
        #  So, it's Okay to use area, b/c is difficult to ID contours that are
        #    at or near an image border.
        max_area = GRAY_IMG.shape[0] * GRAY_IMG.shape[1] * 0.81
        max_length = max(GRAY_IMG.shape[0], GRAY_IMG.shape[1]) * 0.9

        # Note: using aperatureSize decreases effects of other parameters.
        found_edges = cv2.Canny(image=self.filter_image(),
                                threshold1=canny_th_min,
                                threshold2=canny_th_max,
                                # apertureSize=3,  # Must be 3, 5, or 7.
                                L2gradient=True)

        mask = found_edges != 0
        canny_img = GRAY_IMG * (mask[:, :].astype(GRAY_IMG.dtype))
        # canny_img dtype: unit8

        found_contours, _h = cv2.findContours(image=canny_img,
                                              mode=c_mode,
                                              method=c_method)
        if c_type == 'cv2.contourArea':
            self.contours['selected_canny'] = [
                _c for _c in found_contours
                if max_area > cv2.contourArea(_c) >= c_limit]
        else:  # type is cv2.arcLength; aka "perimeter"
            self.contours['selected_canny'] = [
                _c for _c in found_contours
                if max_length > cv2.arcLength(_c, closed=False) >= self.contour_limit]

        # Used only for reporting.
        self.num_contours['canny_all'].set(len(found_contours))
        self.num_contours['canny_select'].set(len(self.contours['selected_canny']))

        contoured_img = INPUT_IMG.copy()

        # Draw hulls around selected contours when hull area is more than
        #   10% of contour area. This prevents obfuscation of drawn lines
        #   when hulls and contours are similar. 10% limit is arbitrary.
        if self.radio_val['hull_pref'].get():
            hull_list = []
            for i, _ in enumerate(self.contours['selected_canny']):
                hull = cv2.convexHull(self.contours['selected_canny'][i])
                if cv2.contourArea(hull) >= cv2.contourArea(
                        self.contours['selected_canny'][i]) * 1.1:
                    hull_list.append(hull)

            cv2.drawContours(contoured_img,
                             contours=hull_list,
                             contourIdx=-1,  # all hulls.
                             color=const.CBLIND_COLOR_CV['sky blue'],
                             thickness=line_thickness * 3,
                             lineType=cv2.LINE_AA)

        # NOTE: drawn_canny is what is saved with the 'Save' button.
        self.contours['drawn_canny'] = cv2.drawContours(contoured_img,
                                                        contours=self.contours['selected_canny'],
                                                        contourIdx=-1,  # all contours.
                                                        color=self.contour_color,
                                                        thickness=line_thickness * 2,
                                                        lineType=cv2.LINE_AA)

        self.tkimg['canny'] = manage.tkimage(canny_img)
        self.canny_lbl.configure(image=self.tkimg['canny'])
        self.canny_lbl.grid(column=0, row=0,
                            padx=5, pady=5,
                            sticky=tk.NSEW)

        self.tkimg['drawn_can'] = manage.tkimage(self.contours['drawn_canny'])
        self.edge_contour_lbl.configure(image=self.tkimg['drawn_can'])
        self.edge_contour_lbl.grid(column=1, row=0,
                                   padx=5, pady=5,
                                   sticky=tk.NSEW)

    def size_the_contours(self,
                          contour_list: list,
                          called_by: str) -> None:
        """
        Draws a circles around contoured objects. Objects are expected
        to be oblong so that circle diameter can represent object length.
        Args:
            contour_list: List of selected contours from cv2.findContours.
            called_by: Descriptive name of calling function;
            e.g. 'thresh sized' or 'canny sized'. Needs to match string
            used for dict keys in const.WIN_NAME for the sized windows.

        Returns: None
        """

        # Need to show the size window that was hidden in __init__.
        # self.img_window[called_by].deiconify()

        circled_contours = INPUT_IMG.copy()
        line_thickness = infile_dict['line_thickness']
        font_scale = infile_dict['font_scale']
        center_xoffset = infile_dict['center_xoffset']

        for _c in contour_list:
            (_x, _y), radius = cv2.minEnclosingCircle(_c)
            center = (int(_x), int(_y))
            radius = int(radius)
            cv2.circle(circled_contours,
                       center=center,
                       radius=radius,
                       color=self.contour_color,
                       thickness=line_thickness * 2,
                       lineType=cv2.LINE_AA)

            # Display pixel diameter of each circled contour.
            #  Draw a filled black circle to use for text background.
            cv2.circle(circled_contours,
                       center=center,
                       radius=int(radius * 0.7),
                       color=(0, 0, 0),
                       thickness=-1,
                       lineType=cv2.LINE_AA)

            cv2.putText(img=circled_contours,
                        text=f'{radius * 2}px',
                        # Center text in the enclosing circle, scaled by px size.
                        org=(center[0] - center_xoffset, center[1] + 5),
                        fontFace=const.FONT_TYPE,
                        fontScale=font_scale,
                        color=self.contour_color,
                        thickness=line_thickness,
                        lineType=cv2.LINE_AA)  # LINE_AA is anti-aliased

        # cv2.mEC returns circled radius of contour as last element.
        # dia_list = [cv2.minEnclosingCircle(_c)[-1] * 2 for _c in selected_contour_list]
        # mean_size = round(mean(dia_list), 1) if dia_list else 0
        # print('mean threshold dia', mean_size)

        # Note: this string needs to match that used as the key in
        #   const.WIN_NAME dictionary, in the img_window dict, and in
        #   the respective size Button 'command' kw call in
        #   ContourViewer.setup_buttons().  Ugh, messy hard coding.
        if called_by == 'thresh sized':
            self.tkimg['circled_th'] = manage.tkimage(circled_contours)
            self.circled_th_lbl.configure(image=self.tkimg['circled_th'])
            self.circled_th_lbl.grid(column=0, row=0,
                                     padx=5, pady=5,
                                     sticky=tk.NSEW)
        else:  # called by 'canny sized'
            self.tkimg['circled_can'] = manage.tkimage(circled_contours)
            self.circled_can_lbl.configure(image=self.tkimg['circled_can'])
            self.circled_can_lbl.grid(column=0, row=0,
                                      padx=5, pady=5,
                                      sticky=tk.NSEW)


class ContourViewer(ProcessImage):
    """
    A suite of methods to display cv2 contours based on chosen settings
    and parameters as applied in ProcessImage().
    Methods:
        master_layout
        setup_styles
        setup_buttons
        setup_image_windows
        config_sliders
        config_comboboxes
        config_radiobuttons
        grid_selector_widgets
        default_settings
        report
        process_all
        process_contours
    """

    __slots__ = ('cbox', 'radio',
                 'report_frame', 'selector_frame',
                 'slider', 'contour_settings_txt',
                 )

    def __init__(self):
        super().__init__()
        # self.configure(bg='green')
        self.frame_report = tk.Frame()
        self.frame_selectors = tk.Frame()

        # Note: The matching control variable attributes for the
        #   following 15 selector widgets are in ProcessImage __init__.
        self.slider = {
            'alpha': tk.Scale(master=self.frame_selectors),
            'alpha_lbl': tk.Label(master=self.frame_selectors),
            'beta': tk.Scale(master=self.frame_selectors),
            'beta_lbl': tk.Label(master=self.frame_selectors),
            'noise_k': tk.Scale(master=self.frame_selectors),
            'noise_k_lbl': tk.Label(master=self.frame_selectors),
            'noise_iter': tk.Scale(master=self.frame_selectors),
            'noise_iter_lbl': tk.Label(master=self.frame_selectors),
            'filter_k': tk.Scale(master=self.frame_selectors),
            'filter_k_lbl': tk.Label(master=self.frame_selectors),
            'canny_th_ratio': tk.Scale(master=self.frame_selectors),
            'canny_th_ratio_lbl': tk.Label(master=self.frame_selectors),
            'canny_th_min': tk.Scale(master=self.frame_selectors),
            'canny_min_lbl': tk.Label(master=self.frame_selectors),
            'c_limit': tk.Scale(master=self.frame_selectors),
            'c_limit_lbl': tk.Label(master=self.frame_selectors),
        }

        self.cbox = {
            'choose_morphop': ttk.Combobox(master=self.frame_selectors),
            'choose_morphop_lbl': tk.Label(master=self.frame_selectors),

            'choose_morphshape': ttk.Combobox(master=self.frame_selectors),
            'choose_morphshape_lbl': tk.Label(master=self.frame_selectors),

            'choose_border': ttk.Combobox(master=self.frame_selectors),
            'choose_border_lbl': tk.Label(master=self.frame_selectors),

            'choose_filter': ttk.Combobox(master=self.frame_selectors),
            'choose_filter_lbl': tk.Label(master=self.frame_selectors),

            'choose_th_type': ttk.Combobox(master=self.frame_selectors),
            'choose_th_type_lbl': tk.Label(master=self.frame_selectors),

            'choose_c_method': ttk.Combobox(master=self.frame_selectors),
            'choose_c_method_lbl': tk.Label(master=self.frame_selectors),
        }

        # Note: c_ is for contour, th_ is for threshold.
        self.radio = {
            'c_mode_lbl': tk.Label(master=self.frame_selectors),
            'c_mode_external': tk.Radiobutton(master=self.frame_selectors),
            'c_mode_list': tk.Radiobutton(master=self.frame_selectors),

            'c_type_lbl': tk.Label(master=self.frame_selectors),
            'c_type_area': tk.Radiobutton(master=self.frame_selectors),
            'c_type_length': tk.Radiobutton(master=self.frame_selectors),

            'hull_lbl': tk.Label(master=self.frame_selectors),
            'hull_yes': tk.Radiobutton(master=self.frame_selectors),
            'hull_no': tk.Radiobutton(master=self.frame_selectors),
        }

        # Is an instance attribute here only because it is used in call
        #  to utils.save_settings_and_img() from the Save button.
        self.contour_settings_txt = ''

        self.master_layout()
        self.setup_styles()
        self.setup_buttons()
        self.config_sliders()
        self.config_comboboxes()
        self.config_radiobuttons()
        self.grid_selector_widgets()
        self.default_settings()
        self.report()
        self.setup_image_windows()

    def master_layout(self) -> None:
        """
        Master (main tk window) keybindings, configurations, and grids
        for settings and reporting frames, and utility buttons.
        """

        # The expected width of the settings report window (app Toplevel)
        #  is 729. Need to set this window near the top right corner
        #  of the screen so that it doesn't cover up the img windows; also
        #  so that the bottom of it is, hopefully, not below the bottom
        #  of the screen.
        self.geometry(f'+{self.winfo_screenwidth()-750}+0')

        # OS-specific window size ranges set in Controller __init__
        # Need to color in all the master Frame and use near-white border;
        #   bd changes to darker shade for click-drag and loss of focus.
        self.config(
            bg=const.MASTER_BG,  # gray80 matches report() txt fg.
            # bg=const.CBLIND_COLOR_TK['sky blue'],
            highlightthickness=5,
            highlightcolor='grey95',
            highlightbackground='grey65'
        )
        # Need to provide exit info msg to Terminal.
        self.protocol('WM_DELETE_WINDOW', lambda: utils.quit_gui(app))

        self.bind_all('<Escape>', lambda _: utils.quit_gui(app))
        self.bind('<Control-q>', lambda _: utils.quit_gui(app))
        # ^^ Note: macOS Command-q will quit program without utils.quit_gui info msg.

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.frame_report.configure(relief='flat',
                                    bg=const.CBLIND_COLOR_TK['sky blue']
                                    )  # bg doesn't show with grid sticky EW.
        self.frame_report.columnconfigure(0, weight=1)
        self.frame_report.columnconfigure(1, weight=1)

        self.frame_selectors.configure(relief='raised',
                                       bg=const.DARK_BG,
                                       borderwidth=5, )
        self.frame_selectors.columnconfigure(0, weight=1)
        self.frame_selectors.columnconfigure(1, weight=1)

        self.frame_report.grid(row=0, column=0,
                               columnspan=2,
                               padx=(5, 5), pady=(5, 5),
                               sticky=tk.EW)
        self.frame_selectors.grid(row=1, column=0,
                                  columnspan=2,
                                  padx=5, pady=(0, 5),
                                  ipadx=4, ipady=4,
                                  sticky=tk.EW)

    def setup_styles(self):
        """
        Configure ttk.Style for Buttons and Comboboxes.
        Called by __init__ and ShapeViewer.shape_settings_layout().

        Returns: None
        """

        # There are problems of tk.Button text showing up on macOS, so use ttk.
        # Explicit styles are needed for buttons to show properly on MacOS.
        #  ... even then, background and pressed colors won't be recognized.
        ttk.Style().theme_use('alt')

        # Use fancy buttons and comboboxes for Linux;
        #   standard theme for Windows and macOS, but with custom font.
        if const.MY_OS == 'lin':
            # This font setting is for the pull-down values.
            self.option_add("*TCombobox*Font", ('TkTooltipFont', 8))
            bstyle = ttk.Style()
            bstyle.configure("My.TButton", font=('TkTooltipFont', 8))
            bstyle.map("My.TButton",
                       foreground=[('active', const.CBLIND_COLOR_TK['yellow'])],
                       background=[('pressed', 'gray30'),
                                   ('active', const.CBLIND_COLOR_TK['vermilion'])],
                       )

            combo_style = ttk.Style()
            combo_style.map('TCombobox',
                            fieldbackground=[('readonly',
                                              const.CBLIND_COLOR_TK['dark blue'])],
                            selectbackground=[('readonly',
                                               const.CBLIND_COLOR_TK['dark blue'])],
                            selectforeround=[('readonly',
                                              const.CBLIND_COLOR_TK['yellow'])],
                            )
        elif const.MY_OS == 'win':
            self.option_add("*TCombobox*Font", ('TkTooltipFont', 7))
        else:  # is macOS
            self.option_add("*TCombobox*Font", ('TkTooltipFont', 10))
            bstyle = ttk.Style()
            bstyle.configure("My.TButton", font=('TkTooltipFont', 11))

    def setup_buttons(self):
        """
        Assign and grid Buttons in the main (app) window.
        Called from __init__.

        Returns: None
        """

        def save_th_settings():
            """
            A Button "command" kw caller to avoid messy or lambda
            statements.
            """
            utils.save_settings_and_img(img2save=self.contours['drawn_thresh'],
                                        txt2save=self.contour_settings_txt,
                                        caller='Threshold')

        def save_can_settings():
            """
            A Button "command" kw caller to avoid messy or lambda
            statements.
            """
            utils.save_settings_and_img(img2save=self.contours['drawn_canny'],
                                        txt2save=self.contour_settings_txt,
                                        caller='Canny')

        # Need to remove the buttons that call ShapeViewer() once used.
        #  Once both buttons have been used (not visible), also remove
        #  their label.
        # This is awkward. The whole ShapeViewer calling or display
        #  logic needs improvement.
        # TODO: FIX problem that shapeimg does not update with changes
        #   in main settings window. Explain in README the update issue.
        def find_shape_from_thresh():
            ShapeViewer(
                self.contours['selected_thresh'],
                self.filtered_img,
                find_in='thresh')
            id_th_shapes_btn.grid_remove()
            if not id_canny_shapes_btn.winfo_ismapped():
                id_shapes_label.grid_remove()

        def find_shape_from_canny():
            ShapeViewer(
                self.contours['selected_canny'],
                self.filtered_img,
                find_in='canny')
            id_canny_shapes_btn.grid_remove()
            if not id_th_shapes_btn.winfo_ismapped():
                id_shapes_label.grid_remove()

        if const.MY_OS in 'lin, win':
            label_font = const.WIDGET_FONT
        else:  # is macOS
            label_font = 'TkTooltipFont', 11

        reset_btn = ttk.Button(text='Reset settings',
                               style='My.TButton',
                               width=0,
                               command=self.default_settings)

        id_shapes_label = tk.Label(text='ID shapes from:',
                                   font=label_font,
                                   bg=const.MASTER_BG)
        id_th_shapes_btn = ttk.Button(text='Threshold',
                                      style='My.TButton',
                                      width=0,
                                      command=find_shape_from_thresh)

        id_canny_shapes_btn = ttk.Button(text='Canny',
                                         style='My.TButton',
                                         width=0,
                                         command=find_shape_from_canny)

        save_btn_label = tk.Label(text='Save settings & contoured image for:',
                                  font=label_font,
                                  bg=const.MASTER_BG,
                                  )
        save_th_btn = ttk.Button(text='Threshold',
                                 style='My.TButton',
                                 width=0,
                                 command=save_th_settings)
        save_canny_btn = ttk.Button(text='Canny',
                                    style='My.TButton',
                                    width=0,
                                    command=save_can_settings)

        # Widget grid for the main window.
        # Note: these grid assignments are a bit shambolic; needs improvement.
        reset_btn.grid(row=2, column=0,
                       padx=(70, 0),
                       pady=(0, 5),
                       sticky=tk.W)

        id_shapes_label.grid(row=3, column=1,
                             padx=(0, 170),
                             pady=(0, 5),
                             sticky=tk.E)
        id_th_shapes_btn.grid(row=3, column=1,
                              padx=(0, 90),
                              pady=(0, 5),
                              sticky=tk.E)
        id_canny_shapes_btn.grid(row=3, column=1,
                                 padx=(0, 35),
                                 pady=(0, 5),
                                 sticky=tk.E)

        save_btn_label.grid(row=3, column=0,
                            padx=(10, 0),
                            pady=(0, 5),
                            sticky=tk.W)
        save_th_btn.grid(row=3, column=0,
                         padx=(230, 0),
                         pady=(0, 5),
                         sticky=tk.W)
        save_canny_btn.grid(row=3, column=0,
                            padx=(310, 0),
                            pady=(0, 5),
                            sticky=tk.W)

    def setup_image_windows(self):
        """
        Reads and displays input image and its grayscale as panels packed
        in their parent toplevel window. Provides initial read of input
        image file.
        Calls manage.tkimage(), which applies scaling, cv2 -> tk array
        conversion, and updates the panel Label's image parameter.
        """

        # These windows are attributed in ProcessImage() __init__.
        self.img_window['input'].title(const.WIN_NAME['input+gray'])
        self.img_window['contrasted'].title(const.WIN_NAME['contrast+redux'])
        self.img_window['filtered'].title(const.WIN_NAME['filtered'])
        self.img_window['thresholded'].title(const.WIN_NAME['th+contours'])
        self.img_window['canny'].title(const.WIN_NAME['canny+contours'])
        self.img_window['thresh sized'].title(const.WIN_NAME['thresh sized'])
        self.img_window['canny sized'].title(const.WIN_NAME['canny sized'])

        def no_exit_on_x():
            """
            Provide a notice in Terminal. Called from .protocol() in loop.
            """
            print('Image windows cannot be closed from the window bar.\n'
                  'They can be minimized to get them out of the way.\n'
                  'You can quit the program from the OpenCV Settings Report'
                  '  window bar or Esc or Ctrl-Q keys.'
                  )

        # Prevent user from inadvertently resizing a window too small to use.
        # Need to disable default window Exit in display windows b/c
        #  subsequent calls to them need a valid path name.
        for _, value in self.img_window.items():
            value.minsize(200, 200)
            value.protocol('WM_DELETE_WINDOW', no_exit_on_x)

        # Display the input image and its grayscale; both are static, so
        #  do not need updating, but retain the image display statement
        #  structure of processed images that need updating.
        # Note: Use 'self' to scope the ImageTk.PhotoImage in the Class,
        #  otherwise it will/may not show b/c of garbage collection.
        self.tkimg['input'] = manage.tkimage(INPUT_IMG)
        self.input_lbl.configure(image=self.tkimg['input'])
        self.input_lbl.grid(column=0, row=0,
                            padx=5, pady=5)

        self.tkimg['gray'] = manage.tkimage(GRAY_IMG)
        self.gray_lbl.configure(image=self.tkimg['gray'])
        self.gray_lbl.grid(column=1, row=0,
                           padx=5, pady=5)

    def config_sliders(self):
        """

        Returns: None

        """
        self.slider['alpha_lbl'].configure(text='Contrast/gain/alpha:',
                                           **const.LABEL_PARAMETERS)
        self.slider['alpha'].configure(from_=0.0, to=4.0,
                                       resolution=0.1,
                                       tickinterval=0.5,
                                       variable=self.slider_val['alpha'],
                                       **const.SCALE_PARAMETERS,
                                       )

        self.slider['beta_lbl'].configure(text='Brightness/bias/beta:',
                                          **const.LABEL_PARAMETERS)
        self.slider['beta'].configure(from_=-127, to=127,
                                      tickinterval=25,
                                      variable=self.slider_val['beta'],
                                      **const.SCALE_PARAMETERS,
                                      )

        self.slider['noise_k_lbl'].configure(text='Reduce noise\nkernel size:',
                                             **const.LABEL_PARAMETERS)
        self.slider['noise_k'].configure(from_=1, to=20,
                                         tickinterval=3,
                                         variable=self.slider_val['noise_k'],
                                         **const.SCALE_PARAMETERS,
                                         )

        self.slider['noise_iter_lbl'].configure(text='Reduce noise\niterations:',
                                                **const.LABEL_PARAMETERS)
        self.slider['noise_iter'].configure(from_=1, to=5,
                                            tickinterval=1,
                                            variable=self.slider_val['noise_iter'],
                                            command=self.process_all,
                                            **const.SCALE_PARAMETERS,
                                            )

        self.slider['filter_k_lbl'].configure(text='Filter kernel size\n'
                                                   '(only odd integers used):',
                                              **const.LABEL_PARAMETERS)
        self.slider['filter_k'].configure(from_=3, to=50,
                                          tickinterval=5,
                                          variable=self.slider_val['filter_k'],
                                          **const.SCALE_PARAMETERS,
                                          )
        self.slider['canny_th_ratio_lbl'].configure(text='Canny threshold ratio:',
                                                    **const.LABEL_PARAMETERS)
        self.slider['canny_th_ratio'].configure(from_=1, to=5,
                                                resolution=0.1,
                                                tickinterval=0.5,
                                                variable=self.slider_val['canny_th_ratio'],
                                                **const.SCALE_PARAMETERS,
                                                )
        self.slider['canny_min_lbl'].configure(text='Canny threshold minimum:',
                                               **const.LABEL_PARAMETERS)
        self.slider['canny_th_min'].configure(from_=1, to=256,
                                              tickinterval=20,
                                              variable=self.slider_val['canny_th_min'],
                                              **const.SCALE_PARAMETERS,
                                              )

        self.slider['c_limit_lbl'].configure(text='Contour chain size\nminimum (px):',
                                             **const.LABEL_PARAMETERS)

        # Need to allow a higher limit for large images; 2000 is arbitrary.
        slide_max = 2000 if manage.infile()['size2scale'] > 2000 else 1000
        self.slider['c_limit'].configure(from_=1, to=slide_max,
                                         tickinterval=slide_max / 10,
                                         variable=self.slider_val['c_limit'],
                                         **const.SCALE_PARAMETERS,
                                         )

        # Need to avoid grabbing all the intermediate values between click
        #   and release; only use value on bind with mouse left button release.
        #   No need to included 'noise_iter' b/c it only has five values.
        sliders = ('alpha', 'beta', 'noise_k', 'filter_k',
                   'canny_th_ratio', 'canny_th_min',)
        for _s in sliders:
            self.slider[_s].bind("<ButtonRelease-1>", self.process_all)

        self.slider['c_limit'].bind("<ButtonRelease-1>", self.process_contours)

    def config_comboboxes(self):

        if const.MY_OS == 'win':
            width_correction = 2
        else:  # is Linux or macOS
            width_correction = 0

        self.cbox['choose_morphop_lbl'].config(text='Reduce noise, morphology operator:',
                                               **const.LABEL_PARAMETERS)
        self.cbox['choose_morphop'].config(textvariable=self.cbox_val['morphop_pref'],
                                           width=18 + width_correction,
                                           values=('cv2.MORPH_OPEN',  # cv2 returns 2
                                                   'cv2.MORPH_CLOSE',  # cv2 returns 3
                                                   'cv2.MORPH_GRADIENT',  # cv2 returns 4
                                                   'cv2.MORPH_BLACKHAT',  # cv2 returns 6
                                                   'cv2.MORPH_HITMISS'),  # cv2 returns 7
                                           **const.COMBO_PARAMETERS
                                           )
        self.cbox['choose_morphop'].bind('<<ComboboxSelected>>',
                                         func=self.process_all)

        self.cbox['choose_morphshape_lbl'].configure(text='morphology shape:',
                                                     **const.LABEL_PARAMETERS)
        self.cbox['choose_morphshape'].config(textvariable=self.cbox_val['morphshape_pref'],
                                              width=16 + width_correction,
                                              values=('cv2.MORPH_RECT',  # cv2 returns 0
                                                      'cv2.MORPH_CROSS',  # cv2 returns 1
                                                      'cv2.MORPH_ELLIPSE'),  # cv2 returns 2
                                              **const.COMBO_PARAMETERS
                                              )
        self.cbox['choose_morphshape'].bind('<<ComboboxSelected>>',
                                            func=self.process_all)

        self.cbox['choose_border_lbl'].configure(text='Border type:',
                                                 **const.LABEL_PARAMETERS)
        self.cbox['choose_border'].config(textvariable=self.cbox_val['border_pref'],
                                          width=22 + width_correction,
                                          values=(
                                              'cv2.BORDER_REFLECT_101',  # cv2 returns 4, default
                                              'cv2.BORDER_REFLECT',  # cv2 returns 2
                                              'cv2.BORDER_REPLICATE',  # cv2 returns 1
                                              'cv2.BORDER_ISOLATED'),  # cv2 returns 16
                                          **const.COMBO_PARAMETERS
                                          )
        self.cbox['choose_border'].bind(
            '<<ComboboxSelected>>', lambda _: self.process_all())

        self.cbox['choose_filter_lbl'].configure(text='Filter type:',
                                                 **const.LABEL_PARAMETERS)
        self.cbox['choose_filter'].config(textvariable=self.cbox_val['filter_pref'],
                                          width=14 + width_correction,
                                          values=(
                                              'cv2.blur',  # is default, 0, a box filter.
                                              'cv2.bilateralFilter',  # cv2 returns 1
                                              'cv2.GaussianBlur',  # cv2 returns 2
                                              'cv2.medianBlur'),  # cv2 returns 3
                                          **const.COMBO_PARAMETERS
                                          )
        self.cbox['choose_filter'].bind(
            '<<ComboboxSelected>>', lambda _: self.process_all())

        self.cbox['choose_th_type_lbl'].configure(text='Threshold type:',
                                                  **const.LABEL_PARAMETERS)
        self.cbox['choose_th_type'].config(textvariable=self.cbox_val['th_type_pref'],
                                           width=26 + width_correction,
                                           values=('cv2.THRESH_BINARY',  # cv2 returns 0
                                                   'cv2.THRESH_BINARY_INVERSE',  # cv2 returns 1
                                                   'cv2.THRESH_OTSU',  # cv2 returns 8
                                                   'cv2.THRESH_OTSU_INVERSE',  # cv2 returns 9
                                                   'cv2.THRESH_TRIANGLE',  # cv2 returns 16
                                                   'cv2.THRESH_TRIANGLE_INVERSE'),  # returns 17
                                           **const.COMBO_PARAMETERS
                                           )
        self.cbox['choose_th_type'].bind(
            '<<ComboboxSelected>>', lambda _: self.process_contours())

        self.cbox['choose_c_method_lbl'].configure(text='... method:',
                                                   **const.LABEL_PARAMETERS)
        self.cbox['choose_c_method'].config(textvariable=self.cbox_val['c_method_pref'],
                                            width=26 + width_correction,
                                            values=('cv2.CHAIN_APPROX_NONE',  # cv2 returns 1
                                                    'cv2.CHAIN_APPROX_SIMPLE',  # cv2 returns 2
                                                    'cv2.CHAIN_APPROX_TC89_L1',  # cv2 returns 3
                                                    'cv2.CHAIN_APPROX_TC89_KCOS'),  # cv2 returns 4
                                            **const.COMBO_PARAMETERS
                                            )
        self.cbox['choose_c_method'].bind(
            '<<ComboboxSelected>>', lambda _: self.process_contours())

    def config_radiobuttons(self):

        self.radio['c_mode_lbl'].config(text='cv2.findContours mode:',
                                        **const.LABEL_PARAMETERS)
        self.radio['c_mode_external'].config(
            text='External',
            variable=self.radio_val['c_mode_pref'],
            value='cv2.RETR_EXTERNAL',
            command=self.process_contours,
            **const.RADIO_PARAMETERS,
        )
        self.radio['c_mode_list'].config(
            text='List',
            variable=self.radio_val['c_mode_pref'],
            value='cv2.RETR_LIST',
            command=self.process_contours,
            **const.RADIO_PARAMETERS,
        )

        self.radio['c_type_lbl'].config(text='Contour chain size, type:',
                                        **const.LABEL_PARAMETERS)
        self.radio['c_type_area'].config(
            text='cv2.contourArea',
            variable=self.radio_val['c_type_pref'],
            value='cv2.contourArea',
            command=self.process_contours,
            **const.RADIO_PARAMETERS,
        )
        self.radio['c_type_length'].config(
            text='cv2.arcLength',
            variable=self.radio_val['c_type_pref'],
            value='cv2.arcLength',
            command=self.process_contours,
            **const.RADIO_PARAMETERS,
        )

        self.radio['hull_lbl'].config(text='Show hulls (in blue)?',
                                      **const.LABEL_PARAMETERS)
        self.radio['hull_yes'].config(
            text='Yes',
            variable=self.radio_val['hull_pref'],
            value=True,
            command=self.process_contours,
            **const.RADIO_PARAMETERS,
        )
        self.radio['hull_no'].config(
            text='No',
            variable=self.radio_val['hull_pref'],
            value=False,
            command=self.process_contours,
            **const.RADIO_PARAMETERS,
        )

    def grid_selector_widgets(self):
        """
        Developer: Grid as a group to make clear spatial relationships.
        """

        if const.MY_OS in 'lin, win':
            slider_grid_params = dict(
                padx=5,
                pady=(7, 0))

            label_grid_params = dict(
                padx=5,
                pady=(5, 0),
                sticky=tk.E)

            # Used for some Combobox and Radiobutton widgets.
            grid_params = dict(
                padx=(8, 0),
                pady=(5, 0),
                sticky=tk.W)
        else:  # is macOS
            slider_grid_params = dict(
                padx=5,
                pady=(4, 0))

            label_grid_params = dict(
                padx=5,
                pady=(4, 0),
                sticky=tk.E)

            # Used for some Combobox and Radiobutton widgets.
            grid_params = dict(
                padx=(8, 0),
                pady=(4, 0),
                sticky=tk.W)

        # Widgets gridded in the self.frame_selectors Frame.
        # Sorted by row number:
        self.slider['alpha_lbl'].grid(column=0, row=0,
                                      **label_grid_params)
        self.slider['alpha'].grid(column=1, row=0,
                                  **slider_grid_params)

        self.slider['beta_lbl'].grid(column=0, row=1,
                                     **label_grid_params)
        self.slider['beta'].grid(column=1, row=1,
                                 **slider_grid_params)

        self.cbox['choose_morphop_lbl'].grid(column=0, row=2,
                                             **label_grid_params)
        self.cbox['choose_morphop'].grid(column=1, row=2,
                                         **grid_params)

        # Note: Put morph shape on same row as morph op.
        self.cbox['choose_morphshape_lbl'].grid(column=1, row=2,
                                                padx=(200, 0),
                                                pady=(5, 0),
                                                sticky=tk.W)

        self.cbox['choose_morphshape'].grid(column=1, row=2,
                                            padx=(0, 5),
                                            pady=(5, 0),
                                            sticky=tk.E)

        self.slider['noise_k_lbl'].grid(column=0, row=4,
                                        **label_grid_params)
        self.slider['noise_k'].grid(column=1, row=4,
                                    **slider_grid_params)

        self.slider['noise_iter_lbl'].grid(column=0, row=5,
                                           **label_grid_params)
        self.slider['noise_iter'].grid(column=1, row=5,
                                       **slider_grid_params)

        self.cbox['choose_border_lbl'].grid(column=0, row=6,
                                            **label_grid_params)
        self.cbox['choose_border'].grid(column=1, row=6,
                                        **grid_params)

        self.cbox['choose_filter_lbl'].grid(column=1, row=6,
                                            padx=(0, 150),
                                            pady=(5, 0),
                                            sticky=tk.E)
        self.cbox['choose_filter'].grid(column=1, row=6,
                                        padx=(0, 5),
                                        pady=(5, 0),
                                        sticky=tk.E)

        self.slider['filter_k_lbl'].grid(column=0, row=8,
                                         **label_grid_params)
        self.slider['filter_k'].grid(column=1, row=8,
                                     **slider_grid_params)

        self.slider['canny_th_ratio_lbl'].grid(column=0, row=9,
                                               **label_grid_params)
        self.slider['canny_th_ratio'].grid(column=1, row=9,
                                           **slider_grid_params)

        self.slider['canny_min_lbl'].grid(column=0, row=10,
                                          **label_grid_params)
        self.slider['canny_th_min'].grid(column=1, row=10,
                                         **slider_grid_params)

        self.cbox['choose_th_type_lbl'].grid(column=0, row=11,
                                             **label_grid_params)
        self.cbox['choose_th_type'].grid(column=1, row=11,
                                         **grid_params)

        self.radio['hull_lbl'].grid(column=1, row=11,
                                    padx=(0, 10),
                                    pady=(5, 0),
                                    sticky=tk.E)
        self.radio['hull_no'].grid(column=1, row=12,
                                   padx=(0, 80),
                                   pady=(0, 0),
                                   sticky=tk.E)
        self.radio['hull_yes'].grid(column=1, row=12,
                                    padx=(0, 30),
                                    pady=(0, 0),
                                    sticky=tk.E)

        self.radio['c_type_lbl'].grid(column=0, row=12,
                                      **label_grid_params)
        self.radio['c_type_area'].grid(column=1, row=12,
                                       **grid_params)
        self.radio['c_type_length'].grid(column=1, row=12,
                                         padx=(120, 0),
                                         pady=(5, 0),
                                         sticky=tk.W)

        self.radio['c_mode_lbl'].grid(column=0, row=13,
                                      **label_grid_params)
        self.radio['c_mode_external'].grid(column=1, row=13,
                                           **grid_params)
        self.radio['c_mode_list'].grid(column=1, row=13,
                                       padx=(84, 0),
                                       pady=(5, 0),
                                       sticky=tk.W)

        if const.MY_OS == 'lin':
            c_method_param = dict(
                padx=(185, 0),
                pady=(5, 0),
                sticky=tk.W)
        elif const.MY_OS == 'win':
            c_method_param = dict(
                padx=(170, 0),
                pady=(5, 0),
                sticky=tk.W)
        else:  # is macOS
            c_method_param = dict(
                padx=(220, 0),
                pady=(4, 0),
                sticky=tk.W)

        self.cbox['choose_c_method_lbl'].grid(column=1, row=13,
                                              **c_method_param)
        self.cbox['choose_c_method'].grid(column=1, row=13,
                                          padx=(0, 10),
                                          pady=(5, 0),
                                          sticky=tk.E)

        self.slider['c_limit_lbl'].grid(column=0, row=15,
                                        **label_grid_params)
        self.slider['c_limit'].grid(column=1, row=15,
                                    **slider_grid_params)

    def default_settings(self) -> None:
        """
        Sets controller widgets at startup. Called from "Reset" button.
        """
        # Note: These 3 statement are duplicated in adjust_contrast().
        contrasted = (
            cv2.convertScaleAbs(
                src=GRAY_IMG,
                alpha=self.slider_val['alpha'].get(),
                beta=self.slider_val['beta'].get())
        )
        self.input_contrast_std.set(int(np.std(GRAY_IMG)))
        self.curr_contrast_std.set(int(np.std(contrasted)))

        # Set/Reset Scale widgets.
        self.slider_val['alpha'].set(1.0)
        self.slider_val['beta'].set(0)
        self.slider_val['noise_k'].set(3)
        self.slider_val['noise_iter'].set(1)
        self.slider_val['filter_k'].set(3)
        self.slider_val['canny_th_ratio'].set(2.5)
        self.slider_val['canny_th_min'].set(50)
        self.slider_val['c_limit'].set(100)

        # Set/Reset Combobox widgets.
        self.cbox['choose_morphop'].current(0)
        self.cbox['choose_morphshape'].current(0)
        self.cbox['choose_border'].current(0)
        self.cbox['choose_filter'].current(0)
        self.cbox['choose_th_type'].current(2)  # cv2.THRESH_OTSU

        # Set/Reset Radiobutton widgets:
        self.radio['hull_no'].select()
        self.radio['c_type_area'].select()
        self.cbox['choose_c_method'].current(1)  # cv2.CHAIN_APPROX_SIMPLE
        self.radio['c_mode_external'].select()

        # Apply the default settings.
        self.process_all()

    def report(self) -> None:
        """
        Write the current settings and cv2 metrics in a Text widget of
        the report_frame. Same text is printed in Terminal from "Save"
        button. Called at from __init__ and process_*() methods.
        """

        # Note: recall that *_val dict are inherited from ProcessImage().
        image_file = manage.arguments()['input']
        start_std = self.input_contrast_std.get()
        new_std = self.curr_contrast_std.get()
        alpha = self.slider_val['alpha'].get()
        beta = self.slider_val['beta'].get()
        noise_k = self.slider_val['noise_k'].get()
        noise_iter = self.slider_val['noise_iter'].get()
        morph_op = self.cbox_val['morphop_pref'].get()
        morph_shape = self.cbox_val['morphshape_pref'].get()
        border = self.cbox_val['border_pref'].get()
        filter_selected = self.cbox_val['filter_pref'].get()
        canny_th_ratio = self.slider_val['canny_th_ratio'].get()
        canny_th_min = self.slider_val['canny_th_min'].get()
        canny_th_max = int(canny_th_min * canny_th_ratio)

        # Need to use only odd kernel integers.
        _k = self.slider_val['filter_k'].get()
        filter_k = _k + 1 if _k % 2 == 0 else _k

        th_type = self.cbox_val['th_type_pref'].get()
        c_limit = self.slider_val['c_limit'].get()
        c_mode = self.radio_val['c_mode_pref'].get()
        c_method = self.cbox_val['c_method_pref'].get()
        c_type = self.radio_val['c_type_pref'].get()

        if c_type == 'cv2.arcLength':
            c_type = f'{c_type} (closed=False)'

        num_th_c_select = self.num_contours['th_select'].get()
        num_th_c_all = self.num_contours['th_all'].get()
        num_canny_c_select = self.num_contours['canny_select'].get()
        num_canny_c_all = self.num_contours['canny_all'].get()

        if filter_selected == 'cv2.bilateralFilter':
            filter_sigmas = (f'd=({filter_k},{filter_k}),'
                             f' sigmaColor={self.sigma_color},'
                             f' sigmaSpace={self.sigma_space}')
        elif filter_selected == 'cv2.GaussianBlur':
            filter_sigmas = (f'sigmaX={self.sigma_x},'
                             f' sigmaY={self.sigma_y}')
        else:
            filter_sigmas = ''

        # Text is formatted for clarity in window, terminal, and saved file.
        tab = " ".ljust(21)
        self.contour_settings_txt = (
            f'Image: {image_file} (alpha SD: {start_std})\n\n'
            f'{"Contrast:".ljust(21)}convertScaleAbs alpha={alpha},'
            f' beta={beta} (adjusted alpha SD {new_std})\n'
            f'{"Noise reduction:".ljust(21)}cv2.getStructuringElement ksize={noise_k},\n'
            f'{tab}cv2.getStructuringElement shape={morph_shape}\n'
            f'{tab}cv2.morphologyEx iterations={noise_iter}\n'
            f'{tab}cv2.morphologyEx op={morph_op},\n'
            f'{tab}cv2.morphologyEx borderType={border}\n'
            f'{"Filter:".ljust(21)}{filter_selected} ksize=({filter_k},{filter_k})\n'
            f'{tab}borderType={border}\n'
            f'{tab}{filter_sigmas}\n'  # is blank line for box and median.
            f'{"cv2.threshold:".ljust(21)}type={th_type},'
            f' thresh={int(self.computed_threshold)}, maxval=255\n'
            f'{"cv2.Canny:".ljust(21)}threshold1={canny_th_min}, threshold2={canny_th_max}\n'
            f'{tab}(1:{canny_th_ratio} threshold ratio), L2gradient=True\n'
            f'{"cv2.findContours:".ljust(21)}mode={c_mode}\n'
            f'{tab}method={c_method}\n'
            f'{"Contour chain size:".ljust(21)}type is {c_type}, minimum is {c_limit} pixels\n\n'
            f'{"# contours selected:".ljust(21)}Threshold {num_th_c_select}'
            f' (from {num_th_c_all} total)\n'
            f'{tab}Canny {num_canny_c_select} (from {num_canny_c_all} total)\n'
        )

        max_line = len(max(self.contour_settings_txt.splitlines(), key=len))

        # Note: TkFixedFont only works when not in a tuple, so no font size.
        #  The goal is to get a suitable platform-independent mono font.
        #  font=('Courier', 10) should also work, if you need font size.
        #  A smaller font is needed to shorten the window as lines & rows are added.
        #  With smaller font, need better fg font contrast, e.g. yellow, not MASTER_BG.
        if const.MY_OS == 'lin':
            txt_font = ('Courier', 9)
        elif const.MY_OS == 'win':
            txt_font = ('Courier', 8)
        else:  # is macOS
            txt_font = ('Courier', 10)

        reporttxt = tk.Text(self.frame_report,
                            # font='TkFixedFont',
                            font=txt_font,
                            bg=const.DARK_BG,
                            # fg=const.MASTER_BG,  # gray80 matches master self bg.
                            fg=const.CBLIND_COLOR_TK['yellow'],  # Matches slider labels.
                            width=max_line,
                            height=self.contour_settings_txt.count('\n'),
                            relief='flat',
                            padx=8, pady=8
                            )
        # Replace prior Text with current text;
        #   hide cursor in Text; (re-)grid in-place.
        reporttxt.delete('1.0', tk.END)
        reporttxt.insert(tk.INSERT, self.contour_settings_txt)
        # Indent helps center text in the Frame.
        reporttxt.tag_config('leftmargin', lmargin1=25)
        reporttxt.tag_add('leftmargin', '1.0', tk.END)
        reporttxt.configure(state=tk.DISABLED)

        reporttxt.grid(column=0, row=0,
                       columnspan=2,
                       sticky=tk.EW)

    def process_all(self, event=None) -> None:
        """
        Runs all image processing methods from ProcessImage() and the
        settings report.
        Calls adjust_contrast(), reduce_noise(), filter_image(), and
        contour_threshold() from ProcessImage.
        Calls report() from ContourViewer.
        Args:
            event: The implicit mouse button event.

        Returns: *event* as a formality; is functionally None.

        """
        self.adjust_contrast()
        self.reduce_noise()
        self.filter_image()
        self.contour_threshold(event)
        self.contour_canny(event)
        self.size_the_contours(self.contours['selected_thresh'], 'thresh sized')
        self.size_the_contours(self.contours['selected_canny'], 'canny sized')
        self.report()

        return event

    def process_contours(self, event=None) -> None:
        """
        Calls contour_threshold() from ProcessImage.
        Calls report() from ContourViewer.
        Args:
            event: The implicit mouse button event.

        Returns: *event* as a formality; is functionally None.

        """
        self.contour_threshold(event)
        self.contour_canny(event)
        self.size_the_contours(self.contours['selected_thresh'], 'thresh sized')
        self.size_the_contours(self.contours['selected_canny'], 'canny sized')
        self.report()

        return event


class ShapeViewer(tk.Canvas):  # or tk.Frame
    """
    A suite of methods to identify selected basic shapes: Triangle,
    Rectangle, Pentagon, Hexagon, Heptagon, Octagon, 5-pointed Star,
    Circle.  Identification is based on chosen settings and parameters
    for cv2.approxPolyDP and cv2.HoughCircles.
    Methods:
        shape_settings_layout
        config_selector_widgets
        set_defaults
        grid_shape_widgets
        process_shapes
        select_shape
        draw_shapes
        find_circles
        report_shape
    """

    # Note: Need to inherit tk.Frame to gain access to tk attributes.
    # Include 'tk' in __slots__ because the tk.Tk inherited in ProcessImage()
    #   is a different inheritance tree than tk.Frame inherited here?? IDK.

    __slots__ = ('tk',
                 'choose_shape', 'choose_shape_lbl', 'close_button',
                 'contours4shape', 'filtered4shape', 'frame_shape_report',
                 'frame_shape_selectors', 'line_thickness', 'num_shapes',
                 'radiobtn', 'saveshape_button', 'select_val',
                 'separator', 'shape_settings_txt', 'shape_settings_win',
                 'shape_tkimg', 'shaped_img', 'shaped_img_window',
                 'shapeimg_lbl', 'slider',
                 )

    def __init__(self, contours, filtered, find_in=None):
        super().__init__()

        self.contours4shape = contours
        self.filtered4shape = filtered
        self.find_in = find_in  # Should be either 'thresh' or 'canny'.

        self.shape_settings_win = None
        self.frame_shape_report = None
        self.frame_shape_selectors = None
        self.separator = None
        self.saveshape_button = None
        self.close_button = None

        self.shaped_img_window = None
        self.shaped_img = None
        self.shape_tkimg = None
        self.shapeimg_lbl = None

        self.shape_settings_txt = ''

        self.choose_shape_lbl = tk.Label()
        self.choose_shape = ttk.Combobox()

        self.select_val = {
            'polygon': tk.StringVar(),
            'hull': tk.StringVar(),
            'find_circle_in': tk.StringVar(),
            'epsilon': tk.DoubleVar(),
            'circle_mindist': tk.IntVar(),
            'circle_param1': tk.IntVar(),
            'circle_param2': tk.IntVar(),
            'circle_minradius': tk.IntVar(),
            'circle_maxradius': tk.IntVar(),
        }

        self.radiobtn = {
            'hull_yes': tk.Radiobutton(),
            'hull_no': tk.Radiobutton(),
            'find_circle_in_th': tk.Radiobutton(),
            'find_circle_in_filtered': tk.Radiobutton(),
            'hull_lbl': tk.Label(),
            'find_circle_in_lbl': tk.Label(),
        }

        self.slider = {
            'epsilon': tk.Scale(),
            'epsilon_lbl': tk.Label(),
            'circle_mindist': tk.Scale(),
            'circle_mindist_lbl': tk.Label(),
            'circle_param1': tk.Scale(),
            'circle_param1_lbl': tk.Label(),
            'circle_param2': tk.Scale(),
            'circle_param2_lbl': tk.Label(),
            'circle_minradius': tk.Scale(),
            'circle_minradius_lbl': tk.Label(),
            'circle_maxradius': tk.Scale(),
            'circle_maxradius_lbl': tk.Label(),
        }

        self.line_thickness = 0
        self.num_shapes = 0

        # (Statement is duplicated in ProcessImage.__init__)
        if arguments['color'] == 'yellow':
            self.contour_color = const.CBLIND_COLOR_CV['yellow']
        else:  # is default CV2 contour color option, green, as (BGR).
            self.contour_color = arguments['color']

        self.shape_settings_layout()
        self.config_selector_widgets()
        self.set_defaults()
        self.grid_shape_widgets()
        self.process_shapes()

    def shape_settings_layout(self) -> None:
        """
        Shape settings and reporting frame configuration, keybindings,
        and grids.
        """

        # Note that ttk.Styles are already set by ContourViewer.setup_styles().

        self.shape_settings_win = tk.Toplevel()
        self.shape_settings_win.title(const.WIN_NAME['shape report'])
        #  ^^title('Shape Approximation, Settings Report')
        self.shape_settings_win.minsize(200, 200)

        self.shaped_img_window = tk.Toplevel()
        self.shaped_img_window.title(const.WIN_NAME[self.find_in])
        self.shaped_img_window.minsize(200, 200)

        self.frame_shape_report = tk.Frame(master=self.shape_settings_win)
        self.frame_shape_selectors = tk.Frame(master=self.shape_settings_win)
        self.shapeimg_lbl = tk.Label(master=self.shaped_img_window)

        self.close_button = ttk.Button(master=self.shape_settings_win)
        self.saveshape_button = ttk.Button(master=self.shape_settings_win)

        def no_exit_on_x():
            """Notice in Terminal. Called from .protocol() in loop."""
            print('The Shape window cannot be closed from the window bar.\n'
                  'It can closed from the "Close" button.\n'
                  'You may quit the program from the OpenCV Settings Report window bar'
                  ' or Esc or Ctrl-Q key.'
                  )

        self.shape_settings_win.protocol('WM_DELETE_WINDOW', no_exit_on_x)

        # OS-specific window size ranges set in Controller __init__
        # Need to color in all the master Frame and use near-white border;
        #   bd changes to darker shade for click-drag and loss of focus.
        self.shape_settings_win.config(
            bg='gray80',  # gray80 matches report() txt fg.
            # bg=const.CBLIND_COLOR_TK['sky blue'],
            highlightthickness=5,
            highlightcolor='grey95',
            highlightbackground='grey65'
        )

        self.shape_settings_win.bind_all('<Escape>', lambda _: utils.quit_gui(app))
        self.shape_settings_win.bind('<Control-q>', lambda _: utils.quit_gui(app))
        # ^^ Note: macOS Command-q will quit program without utils.quit_gui info msg.

        self.shape_settings_win.columnconfigure(0, weight=1)
        self.shape_settings_win.columnconfigure(1, weight=1)

        self.frame_shape_report.configure(relief='flat',
                                          bg=const.CBLIND_COLOR_TK['sky blue']
                                          )  # bg won't show with grid sticky EW.
        self.frame_shape_report.columnconfigure(1, weight=1)
        self.frame_shape_report.rowconfigure(0, weight=1)

        self.frame_shape_selectors.configure(relief='raised',
                                             bg=const.DARK_BG,
                                             borderwidth=5, )
        self.frame_shape_selectors.columnconfigure(0, weight=1)
        self.frame_shape_selectors.columnconfigure(1, weight=1)

        def save_shape_cmd():
            if self.find_in == 'thresh':
                utils.save_settings_and_img(
                    img2save=self.shape_tkimg,
                    txt2save=self.shape_settings_txt,
                    caller='thresh_shape')
            else:  # is for canny
                utils.save_settings_and_img(
                    img2save=self.shape_tkimg,
                    txt2save=self.shape_settings_txt,
                    caller='canny_shape')

        self.saveshape_button.configure(text='Save shape settings and image',
                                        style='My.TButton',
                                        width=0,
                                        command=save_shape_cmd)

        self.close_button.configure(text='Close Shapes window',
                                    style='My.TButton',
                                    width=0,
                                    command=self.shape_settings_win.destroy)

        self.frame_shape_report.grid(row=0, column=0,
                                     columnspan=2,
                                     padx=(5, 5), pady=(5, 5),
                                     sticky=tk.EW)
        self.frame_shape_selectors.grid(row=1, column=0,
                                        columnspan=2,
                                        padx=5, pady=(0, 5),
                                        ipadx=4, ipady=4,
                                        sticky=tk.EW)
        self.saveshape_button.grid(column=0, row=3,
                                   padx=(5, 0),
                                   pady=(0, 5),
                                   sticky=tk.W)
        self.close_button.grid(column=1, row=3,
                               padx=(5, 5),
                               pady=(0, 5),
                               sticky=tk.E)

    def config_selector_widgets(self) -> None:
        """
        Configure all selector widgets and their Labels in the
        frame_shape_selectors Frame.

        Returns: None
        """

        self.line_thickness = infile_dict['line_thickness']

        self.separator = ttk.Separator(master=self.frame_shape_selectors,
                                       orient='horizontal')

        self.circle_msg_lbl = tk.Label(master=self.frame_shape_selectors)

        # For reasons not fully understood, need to assign master= here
        #   and not in __init__.
        self.choose_shape_lbl = tk.Label(master=self.frame_shape_selectors)
        self.choose_shape = ttk.Combobox(master=self.frame_shape_selectors)

        self.radiobtn['hull_lbl'] = tk.Label(master=self.frame_shape_selectors)
        self.radiobtn['hull_yes'] = tk.Radiobutton(master=self.frame_shape_selectors)
        self.radiobtn['hull_no'] = tk.Radiobutton(master=self.frame_shape_selectors)

        self.radiobtn['find_circle_in_lbl'] = tk.Label(master=self.frame_shape_selectors)
        self.radiobtn['find_circle_in_th'] = tk.Radiobutton(master=self.frame_shape_selectors)
        self.radiobtn['find_circle_in_filtered'] = tk.Radiobutton(
            master=self.frame_shape_selectors)

        self.slider['epsilon'] = tk.Scale(master=self.frame_shape_selectors)
        self.slider['epsilon_lbl'] = tk.Label(master=self.frame_shape_selectors)

        self.slider['circle_mindist'] = tk.Scale(master=self.frame_shape_selectors)
        self.slider['circle_mindist_lbl'] = tk.Label(master=self.frame_shape_selectors)

        self.slider['circle_param1'] = tk.Scale(master=self.frame_shape_selectors)
        self.slider['circle_param1_lbl'] = tk.Label(master=self.frame_shape_selectors)

        self.slider['circle_param2'] = tk.Scale(master=self.frame_shape_selectors)
        self.slider['circle_param2_lbl'] = tk.Label(master=self.frame_shape_selectors)

        self.slider['circle_minradius'] = tk.Scale(master=self.frame_shape_selectors)
        self.slider['circle_minradius_lbl'] = tk.Label(master=self.frame_shape_selectors)

        self.slider['circle_maxradius'] = tk.Scale(master=self.frame_shape_selectors)
        self.slider['circle_maxradius_lbl'] = tk.Label(master=self.frame_shape_selectors)

        # Comboboxes:
        self.choose_shape_lbl.configure(text='Select shape to find:',
                                        **const.LABEL_PARAMETERS)
        self.choose_shape.configure(
            textvariable=self.select_val['polygon'],
            width=12,
            values=('Triangle',
                    'Rectangle',
                    'Pentagon',
                    'Hexagon',
                    'Heptagon',
                    'Octagon',
                    '5-pointed Star',
                    'Circle'),
            **const.COMBO_PARAMETERS
        )
        self.choose_shape.bind('<<ComboboxSelected>>',
                               func=self.process_shapes)

        self.circle_msg_lbl.configure(text=
                                      ('Note: Circles are found in the filtered image or'
                                       ' an Otsu threshold of it, not from previously found contours.'),
                                      **const.LABEL_PARAMETERS)
        # Scale sliders:
        self.slider['epsilon_lbl'].configure(text='% polygon contour length (epsilon):',
                                             **const.LABEL_PARAMETERS)
        self.slider['epsilon'].configure(from_=0.001, to=0.06,
                                         resolution=0.001,
                                         tickinterval=0.01,
                                         variable=self.select_val['epsilon'],
                                         **const.SCALE_PARAMETERS)

        self.slider['circle_mindist_lbl'].configure(text='Minimum px dist between circles:',
                                                    **const.LABEL_PARAMETERS)
        self.slider['circle_mindist'].configure(from_=10, to=200,
                                                resolution=1,
                                                tickinterval=20,
                                                variable=self.select_val['circle_mindist'],
                                                **const.SCALE_PARAMETERS)

        self.slider['circle_param1_lbl'].configure(text='cv2.HoughCircles, param1:',
                                                   **const.LABEL_PARAMETERS)
        self.slider['circle_param1'].configure(from_=100, to=500,
                                               resolution=100,
                                               tickinterval=100,
                                               variable=self.select_val['circle_param1'],
                                               **const.SCALE_PARAMETERS)

        self.slider['circle_param2_lbl'].configure(text='cv2.HoughCircles, param2:',
                                                   **const.LABEL_PARAMETERS)
        self.slider['circle_param2'].configure(from_=0.1, to=0.98,
                                               resolution=0.1,
                                               tickinterval=0.1,
                                               variable=self.select_val['circle_param2'],
                                               **const.SCALE_PARAMETERS)

        self.slider['circle_minradius_lbl'].configure(text='cv2.HoughCircles, min px radius):',
                                                      **const.LABEL_PARAMETERS)
        self.slider['circle_minradius'].configure(from_=10, to=200,
                                                  resolution=10,
                                                  tickinterval=20,
                                                  variable=self.select_val['circle_minradius'],
                                                  **const.SCALE_PARAMETERS)

        self.slider['circle_maxradius_lbl'].configure(text='cv2.HoughCircles, max px radius:',
                                                      **const.LABEL_PARAMETERS)
        self.slider['circle_maxradius'].configure(from_=10, to=1000,
                                                  resolution=10,
                                                  tickinterval=100,
                                                  variable=self.select_val['circle_maxradius'],
                                                  **const.SCALE_PARAMETERS)

        # Bind sliders to call processing and reporting on button release.
        sliders = ('epsilon', 'circle_mindist', 'circle_param1', 'circle_param2',
                   'circle_minradius', 'circle_maxradius')
        for _s in sliders:
            self.slider[_s].bind('<ButtonRelease-1>', self.process_shapes)

        # Radiobuttons:
        self.radiobtn['hull_lbl'].config(text='Find shapes as hull?',
                                         **const.LABEL_PARAMETERS)
        self.radiobtn['hull_yes'].configure(
            text='Yes',
            variable=self.select_val['hull'],
            value='yes',
            command=self.process_shapes,
            **const.RADIO_PARAMETERS
        )
        self.radiobtn['hull_no'].configure(
            text='No',
            variable=self.select_val['hull'],
            value='no',
            command=self.process_shapes,
            **const.RADIO_PARAMETERS,
        )

        self.radiobtn['find_circle_in_lbl'].config(text='Find Hough circles in:',
                                                   **const.LABEL_PARAMETERS)
        self.radiobtn['find_circle_in_th'].configure(
            text='Threshold img',  # Need to say 'Edged' if/when use cv2.Canny.
            variable=self.select_val['find_circle_in'],
            value='thresholded',
            command=self.process_shapes,
            **const.RADIO_PARAMETERS
        )
        self.radiobtn['find_circle_in_filtered'].configure(
            text='Filtered img',
            variable=self.select_val['find_circle_in'],
            value='filtered',
            command=self.process_shapes,
            **const.RADIO_PARAMETERS
        )

    def set_defaults(self):
        """
        Set default values for selectors. Called at startup.
        Returns: None
        """
        # Set Combobox starting value.
        self.choose_shape.current(0)

        # Set slider starting positions.
        self.slider['epsilon'].set(0.01)
        self.slider['circle_mindist'].set(100)
        self.slider['circle_param1'].set(300)
        self.slider['circle_param2'].set(0.9)
        self.slider['circle_minradius'].set(20)
        self.slider['circle_maxradius'].set(500)

        # Set Radiobutton starting values.
        self.select_val['hull'].set('no')
        self.select_val['find_circle_in'].set('filtered')

    def grid_shape_widgets(self):
        """
        Grid all selector widgets in the frame_shape_selectors Frame.

        Returns: None
        """

        label_grid_params = dict(
            padx=5,
            pady=(7, 0),
            sticky=tk.E)

        selector_grid_params = dict(
            padx=5,
            pady=(7, 0),
            sticky=tk.W)

        self.choose_shape_lbl.grid(column=0, row=0,
                                   **label_grid_params)
        self.choose_shape.grid(column=1, row=0,
                               **selector_grid_params)

        self.radiobtn['hull_lbl'].grid(column=1, row=0,
                                       padx=(230, 0),
                                       sticky=tk.W)
        self.radiobtn['hull_yes'].grid(column=1, row=0,
                                       padx=(0, 75),
                                       pady=(5, 0),
                                       sticky=tk.E)
        self.radiobtn['hull_no'].grid(column=1, row=0,
                                      padx=(0, 35),
                                      pady=(7, 0),
                                      sticky=tk.E)

        self.slider['epsilon_lbl'].grid(column=0, row=1,
                                        **label_grid_params)
        self.slider['epsilon'].grid(column=1, row=1,
                                    **selector_grid_params)

        self.separator.grid(column=0, row=2,
                            columnspan=2,
                            padx=10,
                            pady=(8, 5),
                            sticky=tk.EW)

        self.circle_msg_lbl.grid(column=0, row=3,
                                 columnspan=2,
                                 padx=5,
                                 pady=(7, 0),
                                 sticky=tk.EW)

        self.radiobtn['find_circle_in_lbl'].grid(column=0, row=4,
                                                 **label_grid_params)
        self.radiobtn['find_circle_in_th'].grid(column=1, row=4,
                                                **selector_grid_params)
        self.radiobtn['find_circle_in_filtered'].grid(column=1, row=4,
                                                      padx=100,
                                                      pady=(7, 0),
                                                      sticky=tk.W)

        self.slider['circle_mindist_lbl'].grid(column=0, row=5,
                                               **label_grid_params)
        self.slider['circle_mindist'].grid(column=1, row=5,
                                           **selector_grid_params)

        self.slider['circle_param1_lbl'].grid(column=0, row=6,
                                              **label_grid_params)
        self.slider['circle_param1'].grid(column=1, row=6,
                                          **selector_grid_params)

        self.slider['circle_param2_lbl'].grid(column=0, row=7,
                                              **label_grid_params)
        self.slider['circle_param2'].grid(column=1, row=7,
                                          **selector_grid_params)

        self.slider['circle_minradius_lbl'].grid(column=0, row=8,
                                                 **label_grid_params)
        self.slider['circle_minradius'].grid(column=1, row=8,
                                             **selector_grid_params)

        self.slider['circle_maxradius_lbl'].grid(column=0, row=9,
                                                 **label_grid_params)
        self.slider['circle_maxradius'].grid(column=1, row=9,
                                             **selector_grid_params)

    def process_shapes(self, event=None):
        """
        A handler for the command kw and button binding for the settings
        control widgets to call multiple methods.
        Args:
            event: An implicit mouse button event.

        Returns: *event* as a formality; is functionally None.

        """
        self.select_shape(self.contours4shape)
        self.report_shape()

        return event

    def select_shape(self, contour_list: list) -> None:
        """
        Filter contoured objects of a specific approximated shape.
        Called from contour_threshold().
        Calls draw_shapes() with selected polygon contours.

        Args:
            contour_list: List of selected contours from cv2.findContours.

        Returns: None
        """

        # Inspiration from Adrian Rosebrock's
        #  https://pyimagesearch.com/2016/02/08/opencv-shape-detection/

        poly_choice = self.select_val['polygon'].get()

        # Finding circles is a special condition that uses Hough Transform
        #   on either the filtered or an Ostu threshold image and thus
        #   sidesteps cv2.findContours and cv2.drawContours.
        num_vertices = {
            'Triangle': 3,
            'Rectangle': 4,
            'Pentagon': 5,
            'Hexagon': 6,
            'Heptagon': 7,
            'Octagon': 8,
            '5-pointed Star': 10,
            'Circle': 0,
        }

        if poly_choice == 'Circle':
            self.find_circles()
            return
        else:
            # Need to change title back to default after title change in find_circles().
            # self.shaped_img_window.title(const.WIN_NAME['shape img'])
            self.shaped_img_window.title(const.WIN_NAME[self.find_in])

        # Draw hulls around selected contours when hull area is 10% or
        #   more than contour area. This prevents obfuscation of outlines
        #   when hulls and contours are similar. 10% limit is arbitrary.
        hull_list = []
        for i, _ in enumerate(contour_list):
            hull = cv2.convexHull(contour_list[i])
            if cv2.contourArea(hull) >= cv2.contourArea(contour_list[i]) * 1.1:
                hull_list.append(hull)

        selected_polygon_contours = []
        self.num_shapes = len(selected_polygon_contours)

        # Need to remove prior contours before finding new selected polygon.
        self.draw_shapes(selected_polygon_contours)

        # NOTE: When using the sample4.jpg (shapes) image, the white border
        #  around the black background has a hexagon-shaped contour, but is
        #  difficult to see with the yellow contour lines. It will be counted
        #  as a hexagon shape unless, in main settings, it is not selected as
        #  a contour by setting cv2.arcLength instead of cv2.contourArea.
        def find_poly(point_set):
            len_arc = cv2.arcLength(point_set, True)
            epsilon = self.select_val['epsilon'].get() * len_arc
            approx_poly = cv2.approxPolyDP(curve=point_set,
                                           epsilon=epsilon,
                                           closed=True)

            # Need to cover shapes with 3 to 8 vertices (sides).
            for _v in range(2, 8):
                if len(approx_poly) == num_vertices[poly_choice] == _v + 1:
                    selected_polygon_contours.append(point_set)

            # Special case for a star:
            if len(approx_poly) == (num_vertices[poly_choice] == 10
                                    and not cv2.isContourConvex(point_set)):
                selected_polygon_contours.append(point_set)

        if self.select_val['hull'].get() == 'yes' and hull_list:
            for _h in hull_list:
                find_poly(_h)
        else:
            for _c in contour_list:
                find_poly(_c)

        self.num_shapes = len(selected_polygon_contours)
        self.draw_shapes(selected_polygon_contours)

    def draw_shapes(self, contours: list) -> None:
        """
        Draw *contours* around detected polygons, hulls, or circles.
        Calls show_settings(). Called from select_shape()

        Args:
            contours: Contour list of polygons or circles.

        Returns: None
        """

        shaped_img = INPUT_IMG.copy()
        use_hull = self.select_val['hull'].get()

        if use_hull == 'yes':
            cnt_color = const.CBLIND_COLOR_CV['sky blue']
        else:
            cnt_color = self.contour_color

        thick_x = 3 if use_hull == 'yes' else 2
        if contours:
            for _c in contours:
                cv2.drawContours(shaped_img,
                                 contours=[_c],
                                 contourIdx=-1,
                                 color=cnt_color,
                                 thickness=self.line_thickness * thick_x,
                                 lineType=cv2.LINE_AA
                                 )

        self.shape_tkimg = manage.tkimage(shaped_img)
        self.shapeimg_lbl.configure(image=self.shape_tkimg)
        self.shapeimg_lbl.grid(column=0, row=0,
                               padx=5, pady=5,
                               sticky=tk.NSEW)

        # Now update the settings text with current values.
        self.report_shape()

    def find_circles(self):
        """
        Implements the cv2.HOUGH_GRADIENT_ALT method of cv2.HoughCircles()
        to approximate circles in a filtered/blured threshold image, then
        displays them on the input image.
        Called from select_shape(). Calls utils.text_array().

        Returns: An array of HoughCircles contours.
        """
        print('now in find_circles()')
        shaped_img = INPUT_IMG.copy()

        mindist = self.select_val['circle_mindist'].get()
        param1 = self.select_val['circle_param1'].get()
        param2 = self.select_val['circle_param2'].get()
        min_radius = self.select_val['circle_minradius'].get()
        max_radius = self.select_val['circle_maxradius'].get()

        # Note: 'thresholded' needs to match the "value" kw value as configured for
        #  self.radiobtn['find_circle_in_th'] and self.radiobtn['find_circle_in_filtered'].
        if self.select_val['find_circle_in'].get() == 'thresholded':
            _, circle_this_img = cv2.threshold(
                self.filtered4shape,
                thresh=0,
                maxval=255,
                type=8  # 8 == cv2.THRESH_OTSU, 16 == cv2.THRESH_TRIANGLE
            )

            # Here HoughCircles works on the threshold image, not found
            #  contours.
            # Note: the printed 'type' needs to agree with the above type= value.
            # print('Circles are being found using a threshold image'
            #       ' with type=cv2.THRESH_OTSU; contours are not used.')
            self.shaped_img_window.title(const.WIN_NAME['circle in thresh'])

            self.shape_tkimg = manage.tkimage(circle_this_img)
            self.shapeimg_lbl.configure(image=self.shape_tkimg)
            self.shapeimg_lbl.grid(column=0, row=0,
                                   padx=5, pady=5,
                                   sticky=tk.NSEW)

        else:  # is 'filtered', the default value.
            circle_this_img = self.filtered4shape
            # Here HoughCircles works on the filtered image, not threshold or contours.
            # print("Circles are being found using the filtered image;"
            #       " contours are not used.")
            self.shaped_img_window.title(const.WIN_NAME['circle in filtered'])

        # source: https://www.geeksforgeeks.org/circle-detection-using-opencv-python/
        # https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
        # Docs general recommendations for HOUGH_GRADIENT_ALT with good image contrast:
        #    param1=300, param2=0.9, minRadius=20, maxRadius=400
        found_circles = cv2.HoughCircles(image=circle_this_img,
                                         method=cv2.HOUGH_GRADIENT_ALT,
                                         dp=1.5,
                                         minDist=mindist,
                                         param1=param1,
                                         param2=param2,
                                         minRadius=min_radius,
                                         maxRadius=max_radius)

        if found_circles is not None:
            # Convert the circle parameters to integers to get the right data type.
            found_circles = np.uint16(np.round(found_circles))

            self.num_shapes = len(found_circles[0, :])

            for _circle in found_circles[0, :]:
                _x, _y, _r = _circle
                # Draw the circumference of the found circle.
                cv2.circle(shaped_img,
                           center=(_x, _y),
                           radius=_r,
                           color=self.contour_color,
                           thickness=self.line_thickness * 2,
                           lineType=cv2.LINE_AA
                           )
                # Draw its center.
                cv2.circle(shaped_img,
                           center=(_x, _y),
                           radius=4,
                           color=self.contour_color,
                           thickness=self.line_thickness * 2,
                           lineType=cv2.LINE_AA
                           )

                # Show found circles marked on the input image.
                self.shape_tkimg = manage.tkimage(shaped_img)
                self.shapeimg_lbl.configure(image=self.shape_tkimg)
                self.shapeimg_lbl.grid(column=0, row=0,
                                       padx=5, pady=5,
                                       sticky=tk.NSEW)

        else:
            self.shape_tkimg = manage.tkimage(shaped_img)
            self.shapeimg_lbl.configure(image=self.shape_tkimg)
            self.shapeimg_lbl.grid(column=0, row=0,
                                   padx=5, pady=5,
                                   sticky=tk.NSEW)

        # Now update the settings text with current values.
        self.report_shape()

    def report_shape(self):

        epsilon = self.select_val['epsilon'].get()
        epsilon_pct = round(self.select_val['epsilon'].get() * 100, 2)
        use_image = self.select_val['find_circle_in'].get()
        mindist = self.select_val['circle_mindist'].get()
        param1 = self.select_val['circle_param1'].get()
        param2 = self.select_val['circle_param2'].get()
        min_radius = self.select_val['circle_minradius'].get()
        max_radius = self.select_val['circle_maxradius'].get()

        shape_type = 'Hull shape:' if self.select_val['hull'].get() == 'yes' else 'Contour shape:'
        # poly_choice = self.choose_shape_pref.get()
        poly_choice = self.select_val['polygon'].get()

        # Text is formatted for clarity in window, terminal, and saved file.
        indent = " ".ljust(18)

        self.shape_settings_txt = (
            f'{"cv2.approxPolyDP".ljust(18)}epsilon={epsilon} ({epsilon_pct}% contour length)\n'
            f'{indent}closed=True\n'
            f'{"cv2.HoughCircles".ljust(18)}image={use_image}\n'
            f'{indent}method=cv2.HOUGH_GRADIENT_ALT\n'
            f'{indent}dp=1.5\n'
            f'{indent}minDist={mindist}\n'
            f'{indent}param1={param1}\n'
            f'{indent}param2={param2}\n'
            f'{indent}minRadius={min_radius}\n'
            f'{indent}maxRadius={max_radius}\n'
            f'{shape_type.ljust(18)}{poly_choice}, found: {self.num_shapes}\n'
        )

        max_line = len(max(self.shape_settings_txt.splitlines(), key=len))

        # Note: TkFixedFont only works when not in a tuple, so no font size.
        #  The goal is to get a suitable platform-independent mono font.
        #  font=('Courier', 10) should also work, if you need font size.
        reporttxt = tk.Text(self.frame_shape_report,
                            font='TkFixedFont',
                            bg=const.DARK_BG,
                            fg='gray80',  # gray80 matches master self bg.
                            width=max_line,
                            height=self.shape_settings_txt.count('\n'),
                            relief='flat',
                            padx=8, pady=8
                            )
        # Replace prior Text with current text;
        #   hide cursor in Text; (re-)grid in-place.
        reporttxt.delete('1.0', tk.END)
        reporttxt.insert(tk.INSERT, self.shape_settings_txt)
        # Indent helps center text in the Frame.
        reporttxt.tag_config('leftmargin', lmargin1=25)
        reporttxt.tag_add('leftmargin', '1.0', tk.END)
        reporttxt.configure(state=tk.DISABLED)

        reporttxt.grid(column=0, row=0,
                       columnspan=2,
                       sticky=tk.EW)


if __name__ == "__main__":
    # Program exits here if the system platform or Python version check fails.
    utils.check_platform()
    vcheck.minversion('3.7')
    arguments = manage.arguments()

    # All checks are good, so grab as a 'global' the dictionary of
    #   command line argument values, and reference often used values...
    infile_dict = manage.infile()
    INPUT_IMG = infile_dict['input_img']
    GRAY_IMG = infile_dict['gray_img']

    try:
        app = ContourViewer()
        app.title('OpenCV Settings Report')
        # Need to prevent errant window resize becoming too small to see.
        app.resizable(False, False)
        print(f'{Path(__file__).name} is now running...')
        app.mainloop()
    except KeyboardInterrupt:
        _msg = '*** User quit the program from Terminal/Console ***\n'
        print(_msg)
