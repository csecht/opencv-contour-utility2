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
# tkinter(Tk/Tcl) is included with most Python3 distributions,
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


# pylint: disable=use-dict-literal, no-member

class ProcessImage(tk.Tk):
    """
    A suite of OpenCV methods for applying various image processing
    functions involved in identifying objects from an image file.

    OpenCV's methods used: cv2.convertScaleAbs, cv2.getStructuringElement,
    cv2.morphologyEx, cv2 filters, cv2.threshold, cv2.Canny,
    cv2.findContours, cv2.contourArea,cv2.arcLength, cv2.drawContours,
    cv2.minEnclosingCircle.

    Class methods and internal functions:
    setup_image_windows > no_exit_on_x
    adjust_contrast
    reduce_noise
    filter_image
    contour_threshold
    contour_canny
    size_the_contours
    select_shape > find_poly
    draw_shapes
    find_circles
    """

    def __init__(self):
        super().__init__()

        # Note: The matching selector widgets for the following 15
        #  control variables are in ContourViewer __init__.
        self.slider_val = {
            # Used for contours.
            'alpha': tk.DoubleVar(),
            'beta': tk.IntVar(),
            'noise_k': tk.IntVar(),
            'noise_iter': tk.IntVar(),
            'filter_k': tk.IntVar(),
            'canny_th_ratio': tk.DoubleVar(),
            'canny_th_min': tk.IntVar(),
            'c_limit': tk.IntVar(),
            # Used for shapes.
            'epsilon': tk.DoubleVar(),
            'circle_mindist': tk.IntVar(),
            'circle_param1': tk.IntVar(),
            'circle_param2': tk.IntVar(),
            'circle_minradius': tk.IntVar(),
            'circle_maxradius': tk.IntVar(),
        }
        self.cbox_val = {
            # Used for contours.
            'morphop_pref': tk.StringVar(),
            'morphshape_pref': tk.StringVar(),
            'border_pref': tk.StringVar(),
            'filter_pref': tk.StringVar(),
            'th_type_pref': tk.StringVar(),
            'c_method_pref': tk.StringVar(),
            # Used for shapes.
            'polygon': tk.StringVar(),
        }
        self.radio_val = {
            # Used for contours.
            'c_mode_pref': tk.StringVar(),
            'c_type_pref': tk.StringVar(),
            'hull_pref': tk.BooleanVar(),
            # Used for shapes.
            'hull_shape': tk.StringVar(),
            'find_circle_in': tk.StringVar(),
            'find_shape_in': tk.StringVar(),
        }

        self.input_contrast_std = tk.DoubleVar()
        self.curr_contrast_std = tk.DoubleVar()

        # Arrays of images to be processed. When used within a method,
        #  the purpose of self.tkimg* is to prevent losing the image var
        #  through garbage collection. Dict values are for panels of PIL
        #  ImageTk.PhotoImage used for Label image display in img windows.
        self.tkimg = {
            'input': None,
            'gray': None,
            'contrast': None,
            'redux': None,
            'filter': None,
            'thresh': None,
            'canny': None,
            'drawn_thresh': None,
            'drawn_canny': None,
            'circled_th': None,
            'circled_can': None,
            'shaped': None,
        }

        # Contour lists populated with cv2.findContours point sets.
        stub_array = np.ones((5, 5), 'uint8')
        self.contours = {
            'drawn_thresh': stub_array,
            'drawn_canny': stub_array,
            'selected_found_thresh': [stub_array],
            'selected_found_canny': [stub_array],
        }

        self.num_contours = {
            'th_all': tk.IntVar(),
            'th_select': tk.IntVar(),
            'canny_all': tk.IntVar(),
            'canny_select': tk.IntVar(),
        }

        self.img_window = {}
        self.img_label = {}
        self.filtered_img = stub_array
        self.reduced_noise_img = stub_array

        # Image processing parameters.
        self.sigma_color = 1
        self.sigma_space = 1
        self.sigma_x = 1
        self.sigma_y = 1
        self.computed_threshold = 0
        self.contour_limit = 0
        self.num_shapes = 0

        # The highlight color used to draw contours and shapes.
        if arguments['color'] == 'yellow':
            self.contour_color = const.CBLIND_COLOR_CV['yellow']
        else:  # is default CV2 contour color, green, as (B,G,R).
            self.contour_color = arguments['color']

    def adjust_contrast(self) -> None:
        """
        Adjust contrast of the input GRAY_IMG image.
        Updates contrast and brightness via alpha and beta sliders.
        Displays contrasted and redux noise images.
        Calls reduce_noise, manage.tk_image().

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
        self.tkimg['contrast'] = manage.tk_image(contrasted)
        self.img_label['contrast'].configure(image=self.tkimg['contrast'])

        return contrasted

    def reduce_noise(self) -> np.ndarray:
        """
        Reduce noise in grayscale image with erode and dilate actions of
        cv2.morphologyEx.
        Uses cv2.getStructuringElement params shape=self.morphshape_val.
        Uses cv2.morphologyEx params op=self.morph_op,
        kernel=<local structuring element>, iterations=self.noise_iter,
        borderType=self.border_val.
        Called only by adjust_contrast(). Calls manage.tk_image().

        Returns:
             The array defined in adjust_contrast() with noise reduction.
        """

        # Need integers for the cv2 function parameters.
        morph_shape = const.CV_MORPH_SHAPE[self.cbox_val['morphshape_pref'].get()]

        # kernel (ksize), used in cv2.getStructuringElement, needs to be a tuple.
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

        self.tkimg['redux'] = manage.tk_image(self.reduced_noise_img)
        self.img_label['redux'].configure(image=self.tkimg['redux'])

        return self.reduced_noise_img

    def filter_image(self) -> np.ndarray:
        """
        Applies a filter selection to blur the image for Canny edge
        detection or threshold contouring.
        Called from contour_threshold(). Calls manage.tk_image().

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

        self.tkimg['filter'] = manage.tk_image(self.filtered_img)
        self.img_label['filter'].configure(image=self.tkimg['filter'])

        return self.filtered_img

    def contour_threshold(self, event=None) -> int:
        """
        Identify object contours with cv2.threshold() and
        cv2.drawContours(). Threshold types limited to Otsu and Triangle.
        Called by process_*() methods. Calls manage.tk_image().

        Args:
            event: An implicit mouse button event.

        Returns: *event* as a formality; is functionally None.
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
            self.contours['selected_found_thresh'] = [
                _c for _c in found_contours
                if max_area > cv2.contourArea(_c) >= c_limit]
        else:  # c_type is cv2.arcLength; aka "perimeter"
            self.contours['selected_found_thresh'] = [
                _c for _c in found_contours
                if max_length > cv2.arcLength(_c, closed=False) >= c_limit]

        # Used only for reporting.
        self.num_contours['th_all'].set(len(found_contours))
        self.num_contours['th_select'].set(len(self.contours['selected_found_thresh']))

        contoured_img = INPUT_IMG.copy()

        # Draw hulls around selected contours when hull area is more than
        #   10% of contour area. This prevents obfuscation of drawn lines
        #   when hulls and contours are similar. 10% limit is arbitrary.
        if self.radio_val['hull_pref'].get():
            hull_pointset = []
            for i, _ in enumerate(self.contours['selected_found_thresh']):
                hull = cv2.convexHull(self.contours['selected_found_thresh'][i])
                if cv2.contourArea(hull) >= cv2.contourArea(
                        self.contours['selected_found_thresh'][i]) * 1.1:
                    hull_pointset.append(hull)

            cv2.drawContours(contoured_img,
                             contours=hull_pointset,
                             contourIdx=-1,  # all hulls.
                             color=const.CBLIND_COLOR_CV['sky blue'],
                             thickness=LINE_THICKNESS * 3,
                             lineType=cv2.LINE_AA)

        # NOTE: drawn_thresh is what is saved with the 'Save' button.
        self.contours['drawn_thresh'] = cv2.drawContours(
            contoured_img,
            contours=self.contours['selected_found_thresh'],
            contourIdx=-1,  # all contours.
            color=self.contour_color,
            thickness=LINE_THICKNESS * 2,
            lineType=cv2.LINE_AA)

        # Need to use self.*_img to keep attribute reference and thus
        #   prevent garbage collection.
        self.tkimg['thresh'] = manage.tk_image(thresh_img)
        self.img_label['thresh'].configure(image=self.tkimg['thresh'])

        self.tkimg['drawn_thresh'] = manage.tk_image(self.contours['drawn_thresh'])
        self.img_label['th_contour'].configure(image=self.tkimg['drawn_thresh'])

        return event

    def contour_canny(self, event=None) -> None:
        """
        Identify objects using cv2.Canny() edges and cv2.drawContours().
        Called by process_*() methods. Calls manage.tk_image().

        Args:
            event: An implicit mouse button event.

        Returns: *event* as a formality; is functionally None.
        """

        # Source of coding ideas:
        # https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
        # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

        # Note to dev: Much of this method is duplicated in contour_threshold();
        #   consider consolidating the two methods.

        # Canny recommended an upper:lower ratio between 2:1 and 3:1.
        canny_th_ratio = self.slider_val['canny_th_ratio'].get()
        canny_th_min = self.slider_val['canny_th_min'].get()
        canny_th_max = int(canny_th_min * canny_th_ratio)
        c_mode = const.CONTOUR_MODE[self.radio_val['c_mode_pref'].get()]
        c_method = const.CONTOUR_METHOD[self.cbox_val['c_method_pref'].get()]
        c_type = self.radio_val['c_type_pref'].get()
        c_limit = self.slider_val['c_limit'].get()

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
            self.contours['selected_found_canny'] = [
                _c for _c in found_contours
                if max_area > cv2.contourArea(_c) >= c_limit]
        else:  # type is cv2.arcLength; aka "perimeter"
            self.contours['selected_found_canny'] = [
                _c for _c in found_contours
                if max_length > cv2.arcLength(_c, closed=False) >= self.contour_limit]

        # Used only for reporting.
        self.num_contours['canny_all'].set(len(found_contours))
        self.num_contours['canny_select'].set(len(self.contours['selected_found_canny']))

        contoured_img = INPUT_IMG.copy()

        # Draw hulls around selected contours when hull area is more than
        #   10% of contour area. This prevents obfuscation of drawn lines
        #   when hulls and contours are similar. 10% limit is arbitrary.
        if self.radio_val['hull_pref'].get():
            hull_pointset = []
            for i, _ in enumerate(self.contours['selected_found_canny']):
                hull = cv2.convexHull(self.contours['selected_found_canny'][i])
                if cv2.contourArea(hull) >= cv2.contourArea(
                        self.contours['selected_found_canny'][i]) * 1.1:
                    hull_pointset.append(hull)

            cv2.drawContours(image=contoured_img,
                             contours=hull_pointset,
                             contourIdx=-1,  # all hulls.
                             color=const.CBLIND_COLOR_CV['sky blue'],
                             thickness=LINE_THICKNESS * 3,
                             lineType=cv2.LINE_AA)

        # NOTE: drawn_canny is what is saved with the 'Save' button.
        self.contours['drawn_canny'] = cv2.drawContours(
            image=contoured_img,
            contours=self.contours['selected_found_canny'],
            contourIdx=-1,  # all contours.
            color=self.contour_color,
            thickness=LINE_THICKNESS * 2,
            lineType=cv2.LINE_AA)

        self.tkimg['canny'] = manage.tk_image(canny_img)
        self.img_label['canny'].configure(image=self.tkimg['canny'])

        self.tkimg['drawn_canny'] = manage.tk_image(self.contours['drawn_canny'])
        self.img_label['can_contour'].configure(image=self.tkimg['drawn_canny'])

        return event

    def size_the_contours(self,
                          contour_pointset: list,
                          called_by: str) -> None:
        """
        Draws a circles around contoured objects. Objects are expected
        to be oblong so that circle diameter can represent object length.
        Called by process_*() methods. Calls manage.tk_image().
        Args:
            contour_pointset: List of selected contours from cv2.findContours.
            called_by: Descriptive name of calling function;
            e.g. 'thresh sized' or 'canny sized'. Needs to match string
            used for dict keys in const.WIN_NAME for the sized windows.

        Returns: None
        """

        circled_contours = INPUT_IMG.copy()
        center_xoffset = infile_dict['center_xoffset']

        for _c in contour_pointset:
            (_x, _y), radius = cv2.minEnclosingCircle(_c)
            center = (int(_x), int(_y))
            radius = int(radius)
            cv2.circle(circled_contours,
                       center=center,
                       radius=radius,
                       color=self.contour_color,
                       thickness=LINE_THICKNESS * 2,
                       lineType=cv2.LINE_AA)

            # Display pixel diameter of each circled contour.
            #  Draw a filled black circle to use for text background.
            cv2.circle(img=circled_contours,
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
                        fontScale=infile_dict['font_scale'],
                        color=self.contour_color,
                        thickness=LINE_THICKNESS,
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
            self.tkimg['circled_th'] = manage.tk_image(circled_contours)
            self.img_label['circled_th'].configure(image=self.tkimg['circled_th'])
        else:  # called by 'canny sized'
            self.tkimg['circled_can'] = manage.tk_image(circled_contours)
            self.img_label['circled_can'].configure(image=self.tkimg['circled_can'])

    def select_shape(self, contour_pointset: list) -> None:
        """
        Filter contoured objects of a specific approximated shape.
        Called from the process_shapes() handler that determines whether
        to pass contours from a point set list of selected contours from
        either the threshold or Canny image.
        Calls draw_shapes() with selected polygon contours.

        Args:
            contour_pointset: List of selected contours from cv2.findContours.

        Returns: None
        """

        # Inspiration from Adrian Rosebrock's
        #  https://pyimagesearch.com/2016/02/08/opencv-shape-detection/

        poly_choice = self.cbox_val['polygon'].get()

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

        # Finding circles is a special condition that uses Hough Transform
        #   on either the filtered or an Ostu threshold input image and thus
        #   sidesteps cv2.findContours and cv2.drawContours. Otherwise,
        #   proceed with finding one of the other selected shapes in either
        #   (or both) input contour set.
        if poly_choice == 'Circle':
            self.find_circles()
            return

        # Draw hulls around selected contours when hull area is 10% or
        #   more than contour area. This prevents obfuscation of outlines
        #   when hulls and contours are similar. 10% limit is arbitrary.
        hull_pointset = []
        for i, _ in enumerate(contour_pointset):
            hull = cv2.convexHull(contour_pointset[i])
            if cv2.contourArea(hull) >= cv2.contourArea(contour_pointset[i]) * 1.1:
                hull_pointset.append(hull)

        # Need to remove prior contours before finding new selected polygon.
        selected_polygon_contours = []
        self.num_shapes = len(selected_polygon_contours)
        self.draw_shapes(selected_polygon_contours)

        # NOTE: When using the sample4.jpg (shapes) image, the white border
        #  around the black background has a hexagon-shaped contour, but is
        #  difficult to see with the yellow contour lines. It will be counted
        #  as a hexagon shape unless, in main settings, it is not selected as
        #  a contour by setting cv2.arcLength instead of cv2.contourArea.
        def find_poly(point_set):
            len_arc = cv2.arcLength(point_set, True)
            epsilon = self.slider_val['epsilon'].get() * len_arc
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

        # The main engine for contouring the selected shape.
        if self.radio_val['hull_shape'].get() == 'yes' and hull_pointset:
            for _h in hull_pointset:
                find_poly(_h)
        else:
            for _c in contour_pointset:
                find_poly(_c)

        self.num_shapes = len(selected_polygon_contours)
        self.draw_shapes(selected_polygon_contours)

    def draw_shapes(self, selected_contours: list) -> None:
        """
        Draw *contours* around detected polygons, hulls, or circles.
        Calls show_settings(). Called from select_shape()

        Args:
            selected_contours: Contour list of polygons or circles.

        Returns: None
        """
        img4shaping = INPUT_IMG.copy()

        if self.radio_val['find_shape_in'].get() == 'threshold':
            shapeimg_win_name = 'thresh'
        else:  # == 'canny'
            shapeimg_win_name = 'canny'
        self.img_window['shaped'].title(const.WIN_NAME[shapeimg_win_name])

        use_hull = self.radio_val['hull_shape'].get()
        if use_hull == 'yes':
            cnt_color = const.CBLIND_COLOR_CV['sky blue']
        else:
            cnt_color = self.contour_color
        thickness_factor = 3 if use_hull == 'yes' else 2

        if selected_contours:
            for _c in selected_contours:
                cv2.drawContours(img4shaping,
                                 contours=[_c],
                                 contourIdx=-1,
                                 color=cnt_color,
                                 thickness=LINE_THICKNESS * thickness_factor,
                                 lineType=cv2.LINE_AA
                                 )

        self.tkimg['shaped'] = manage.tk_image(img4shaping)
        self.img_label['shaped'].configure(image=self.tkimg['shaped'])

    def find_circles(self) -> None:
        """
        Implements the cv2.HOUGH_GRADIENT_ALT method of cv2.HoughCircles()
        to approximate circles in the filtered/blured image or a threshold
        thereof, and shows overlay of circles on the input image.
        Called from select_shape(). Calls manage.tk_image().

        Returns: None
        """

        img4shaping = INPUT_IMG.copy()

        mindist = self.slider_val['circle_mindist'].get()
        param1 = self.slider_val['circle_param1'].get()
        param2 = self.slider_val['circle_param2'].get()
        min_radius = self.slider_val['circle_minradius'].get()
        max_radius = self.slider_val['circle_maxradius'].get()

        # Note: 'thresholded' needs to match the "value" kw value as configured for
        #  self.radiobtn['find_circle_in_th'] and self.radiobtn['find_circle_in_filtered'].
        if self.radio_val['find_circle_in'].get() == 'thresholded':
            self.img_window['shaped'].title(const.WIN_NAME['circle in thresh'])

            _, img4houghcircles = cv2.threshold(
                src=self.filter_image(),  # or use self.filter_image()
                thresh=0,
                maxval=255,
                type=8  # 8 == cv2.THRESH_OTSU, 16 == cv2.THRESH_TRIANGLE
            )

        else:  # is 'filtered', the default value.
            # Here HoughCircles works on the filtered image, not threshold or contours.
            self.img_window['shaped'].title(const.WIN_NAME['circle in filtered'])
            img4houghcircles = self.filter_image()

        # source: https://www.geeksforgeeks.org/circle-detection-using-opencv-python/
        # https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
        # Docs general recommendations for HOUGH_GRADIENT_ALT with good image contrast:
        #   dp=1.5, param1=300, param2=0.9, minRadius=20, maxRadius=400
        found_circles = cv2.HoughCircles(image=img4houghcircles,
                                         method=cv2.HOUGH_GRADIENT_ALT,
                                         dp=1.5,
                                         minDist=mindist,
                                         param1=param1,
                                         param2=param2,
                                         minRadius=min_radius,
                                         maxRadius=max_radius
                                         )

        if found_circles is not None:
            # Convert the circle parameters to integers to get the right data type.
            found_circles = np.uint16(np.round(found_circles))
            self.num_shapes = len(found_circles[0, :])

            for _circle in found_circles[0, :]:
                _x, _y, _r = _circle
                # Draw the circumference of the found circle.
                cv2.circle(img=img4shaping,
                           center=(_x, _y),
                           radius=_r,
                           color=self.contour_color,
                           thickness=LINE_THICKNESS * 2,
                           lineType=cv2.LINE_AA
                           )
                # Draw its center.
                cv2.circle(img=img4shaping,
                           center=(_x, _y),
                           radius=4,
                           color=self.contour_color,
                           thickness=LINE_THICKNESS * 2,
                           lineType=cv2.LINE_AA
                           )

                # Show found circles highlighted on the input image.
                self.tkimg['shaped'] = manage.tk_image(img4shaping)
                self.img_label['shaped'].configure(image=self.tkimg['shaped'])
        else:
            # No circles found, so display the input image as-is.
            self.tkimg['shaped'] = manage.tk_image(img4shaping)
            self.img_label['shaped'].configure(image=self.tkimg['shaped'])

        # Note: reporting of current metrics and settings is handled by
        #  ImageViewer.process_shapes().


class ImageViewer(ProcessImage):
    """
    A suite of methods to display cv2 contours based on chosen settings
    and parameters as applied in ProcessImage().
    Methods:
    master_setup
    shape_win_setup
    setup_styles
    setup_buttons
    display_input_images
    config_sliders
    config_comboboxes
    config_radiobuttons
    grid_contour_widgets
    grid_shape_widgets
    set_contour_defaults
    set_shape_defaults
    report_contour
    report_shape
    process_all
    process_contours
    process_shapes
    """

    def __init__(self):
        super().__init__()
        # Note: the tk1 param represents the inherited tk.Tk base class.
        self.contour_report_frame = tk.Frame()
        self.contour_selectors_frame = tk.Frame()
        # self.configure(bg='green')  # for dev.

        self.shape_settings_win = tk.Toplevel()
        self.shape_report_frame = tk.Frame(master=self.shape_settings_win)
        self.shape_selectors_frame = tk.Frame(master=self.shape_settings_win)

        # Note: The matching control variable attributes for the
        #   following 14 selector widgets are in ProcessImage __init__.
        self.slider = {
            'alpha': tk.Scale(master=self.contour_selectors_frame),
            'alpha_lbl': tk.Label(master=self.contour_selectors_frame),
            'beta': tk.Scale(master=self.contour_selectors_frame),
            'beta_lbl': tk.Label(master=self.contour_selectors_frame),
            'noise_k': tk.Scale(master=self.contour_selectors_frame),
            'noise_k_lbl': tk.Label(master=self.contour_selectors_frame),
            'noise_iter': tk.Scale(master=self.contour_selectors_frame),
            'noise_iter_lbl': tk.Label(master=self.contour_selectors_frame),
            'filter_k': tk.Scale(master=self.contour_selectors_frame),
            'filter_k_lbl': tk.Label(master=self.contour_selectors_frame),
            'canny_th_ratio': tk.Scale(master=self.contour_selectors_frame),
            'canny_th_ratio_lbl': tk.Label(master=self.contour_selectors_frame),
            'canny_th_min': tk.Scale(master=self.contour_selectors_frame),
            'canny_min_lbl': tk.Label(master=self.contour_selectors_frame),
            'c_limit': tk.Scale(master=self.contour_selectors_frame),
            'c_limit_lbl': tk.Label(master=self.contour_selectors_frame),
            # for shapes
            'epsilon': tk.Scale(master=self.shape_selectors_frame),
            'epsilon_lbl': tk.Label(master=self.shape_selectors_frame),
            'circle_mindist': tk.Scale(master=self.shape_selectors_frame),
            'circle_mindist_lbl': tk.Label(master=self.shape_selectors_frame),
            'circle_param1': tk.Scale(master=self.shape_selectors_frame),
            'circle_param1_lbl': tk.Label(master=self.shape_selectors_frame),
            'circle_param2': tk.Scale(master=self.shape_selectors_frame),
            'circle_param2_lbl': tk.Label(master=self.shape_selectors_frame),
            'circle_minradius': tk.Scale(master=self.shape_selectors_frame),
            'circle_minradius_lbl': tk.Label(master=self.shape_selectors_frame),
            'circle_maxradius': tk.Scale(master=self.shape_selectors_frame),
            'circle_maxradius_lbl': tk.Label(master=self.shape_selectors_frame),
        }

        self.cbox = {
            'choose_morphop': ttk.Combobox(master=self.contour_selectors_frame),
            'choose_morphop_lbl': tk.Label(master=self.contour_selectors_frame),

            'choose_morphshape': ttk.Combobox(master=self.contour_selectors_frame),
            'choose_morphshape_lbl': tk.Label(master=self.contour_selectors_frame),

            'choose_border': ttk.Combobox(master=self.contour_selectors_frame),
            'choose_border_lbl': tk.Label(master=self.contour_selectors_frame),

            'choose_filter': ttk.Combobox(master=self.contour_selectors_frame),
            'choose_filter_lbl': tk.Label(master=self.contour_selectors_frame),

            'choose_th_type': ttk.Combobox(master=self.contour_selectors_frame),
            'choose_th_type_lbl': tk.Label(master=self.contour_selectors_frame),

            'choose_c_method': ttk.Combobox(master=self.contour_selectors_frame),
            'choose_c_method_lbl': tk.Label(master=self.contour_selectors_frame),

            # for shapes
            'choose_shape_lbl': tk.Label(master=self.shape_selectors_frame),
            'choose_shape': ttk.Combobox(master=self.shape_selectors_frame),
        }

        # Note: c_ is for contour, th_ is for threshold.
        self.radio = {
            'c_mode_lbl': tk.Label(master=self.contour_selectors_frame),
            'c_mode_external': tk.Radiobutton(master=self.contour_selectors_frame),
            'c_mode_list': tk.Radiobutton(master=self.contour_selectors_frame),

            'c_type_lbl': tk.Label(master=self.contour_selectors_frame),
            'c_type_area': tk.Radiobutton(master=self.contour_selectors_frame),
            'c_type_length': tk.Radiobutton(master=self.contour_selectors_frame),

            'hull_lbl': tk.Label(master=self.contour_selectors_frame),
            'hull_yes': tk.Radiobutton(master=self.contour_selectors_frame),
            'hull_no': tk.Radiobutton(master=self.contour_selectors_frame),

            # for shapes
            'shape_hull_lbl': tk.Label(master=self.shape_selectors_frame),
            'shape_hull_yes': tk.Radiobutton(master=self.shape_selectors_frame),
            'shape_hull_no': tk.Radiobutton(master=self.shape_selectors_frame),

            'find_circle_lbl': tk.Label(master=self.shape_selectors_frame),
            'find_circle_in_th': tk.Radiobutton(master=self.shape_selectors_frame),
            'find_circle_in_filtered': tk.Radiobutton(master=self.shape_selectors_frame),

            'find_shape_lbl': tk.Label(master=self.shape_selectors_frame),
            'find_shape_in_thresh': tk.Radiobutton(master=self.shape_selectors_frame),
            'find_shape_in_canny': tk.Radiobutton(master=self.shape_selectors_frame),
        }

        # Is an instance attribute here only because it is used in call
        #  to utils.save_settings_and_img() from the Save button.
        self.contour_settings_txt = ''
        self.shape_settings_txt = ''

        # Separator used in shape report window.
        self.separator = ttk.Separator(master=self.shape_selectors_frame,
                                       orient='horizontal')

        # Attributes for shape windows.
        self.circle_msg_lbl = tk.Label(master=self.shape_selectors_frame)
        self.shapeimg_lbl = None
        self.resetshape_button = None
        self.circle_defaults_button = None
        self.saveshape_button = None

        self.setup_image_windows()
        self.display_input_images()
        self.master_setup()
        self.setup_styles()
        self.setup_buttons()
        self.config_sliders()
        self.config_comboboxes()
        self.config_radiobuttons()
        self.grid_contour_widgets()
        self.grid_img_labels()
        self.set_contour_defaults()
        self.report_contour()

        # Shape objects are not displayed at startup, but are processed and
        #  ready for display when called from the 'show shapes' Button.
        self.shape_win_setup()
        self.set_shape_defaults()
        self.grid_shape_widgets()
        self.report_shape()

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
        #  have windows in proper arranged in sensible order on the
        #  screen (input on bottom, sized or shaped on top).
        self.img_window = {
            'shaped': tk.Toplevel(),
            'canny sized': tk.Toplevel(),
            'thresh sized': tk.Toplevel(),
            'canny': tk.Toplevel(),
            'thresholded': tk.Toplevel(),
            'filtered': tk.Toplevel(),
            'contrasted': tk.Toplevel(),
            'input': tk.Toplevel(),
        }

        # Prevent user from inadvertently resizing a window too small to use.
        # Need to disable default window Exit in display windows b/c
        #  subsequent calls to them need a valid path name.
        for _, window in self.img_window.items():
            window.minsize(200, 200)
            window.protocol('WM_DELETE_WINDOW', no_exit_on_x)

        self.img_window['input'].title(const.WIN_NAME['input+gray'])
        self.img_window['contrasted'].title(const.WIN_NAME['contrast+redux'])
        self.img_window['filtered'].title(const.WIN_NAME['filtered'])
        self.img_window['thresholded'].title(const.WIN_NAME['th+contours'])
        self.img_window['canny'].title(const.WIN_NAME['canny+contours'])
        self.img_window['thresh sized'].title(const.WIN_NAME['thresh sized'])
        self.img_window['canny sized'].title(const.WIN_NAME['canny sized'])
        self.img_window['shaped'].title(const.WIN_NAME['shapes'])

        # The Labels to display scaled images, which are updated using
        #  .configure() for 'image=' in their respective processing methods.
        #  Labels are gridded in their respective img_window in
        #  ImageViewer.grid_img_labels().
        self.img_label = {
            'input': tk.Label(self.img_window['input']),
            'gray': tk.Label(self.img_window['input']),
            'contrast': tk.Label(self.img_window['contrasted']),
            'redux': tk.Label(self.img_window['contrasted']),
            'filter': tk.Label(self.img_window['filtered']),
            'thresh': tk.Label(self.img_window['thresholded']),
            'th_contour': tk.Label(self.img_window['thresholded']),
            'canny': tk.Label(self.img_window['canny']),
            'can_contour': tk.Label(self.img_window['canny']),
            'circled_th': tk.Label(self.img_window['thresh sized']),
            'circled_can': tk.Label(self.img_window['canny sized']),
            'shaped': tk.Label(self.img_window['shaped']),
        }

    def master_setup(self) -> None:
        """
        Master (main tk window) keybindings, configurations, and grids
        for settings and reporting frames, and utility buttons.
        """

        # The expected width of the settings report_contour window (app Toplevel)
        #  is 729. Need to set this window near the top right corner
        #  of the screen so that it doesn't cover up the img windows; also
        #  so that the bottom of it is, hopefully, not below the bottom
        #  of the screen.:
        if const.MY_OS in 'lin, dar':
            adjust_width = 740
        else:  # is Windows
            adjust_width = 760

        self.geometry(f'+{self.winfo_screenwidth() - adjust_width}+0')

        # Need to set min width so that entire contrast reporting line
        #  fits its maximum length., e.g, beta == -120 and SD == 39.0.
        if const.MY_OS == 'win':
            self.minsize(750, 400)
        elif const.MY_OS == 'lin':
            self.minsize(690, 400)

        # Need to color in all the master Frame and use yellow border;
        #   border highlightcolor changes to dark grey when click-dragged
        #   and loss of focus.
        self.config(
            bg=const.MASTER_BG,  # gray80 matches report_contour() txt fg.
            # bg=const.CBLIND_COLOR_TK['sky blue'],  # for dev.
            highlightthickness=5,
            highlightcolor=const.CBLIND_COLOR_TK['yellow'],
            highlightbackground='grey65'
        )
        # Need to provide exit info msg to Terminal.
        self.protocol('WM_DELETE_WINDOW', lambda: utils.quit_gui(app))

        self.bind_all('<Escape>', lambda _: utils.quit_gui(app))
        self.bind_all('<Control-q>', lambda _: utils.quit_gui(app))
        # ^^ Note: macOS Command-q will quit program without utils.quit_gui info msg.

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.contour_report_frame.configure(relief='flat',
                                            bg=const.CBLIND_COLOR_TK['sky blue']
                                            )  # bg doesn't show with grid sticky EW.
        self.contour_report_frame.columnconfigure(0, weight=1)
        self.contour_report_frame.columnconfigure(1, weight=1)

        self.contour_selectors_frame.configure(relief='raised',
                                               bg=const.DARK_BG,
                                               borderwidth=5)
        self.contour_selectors_frame.columnconfigure(0, weight=1)
        self.contour_selectors_frame.columnconfigure(1, weight=1)

        self.contour_report_frame.grid(column=0, row=0,
                                       columnspan=2,
                                       padx=(5, 5), pady=(5, 5),
                                       sticky=tk.EW)
        self.contour_selectors_frame.grid(column=0, row=1,
                                          columnspan=2,
                                          padx=5, pady=(0, 5),
                                          ipadx=4, ipady=4,
                                          sticky=tk.EW)

        # At startup, try to reduce screen clutter, so
        #  do not show the Shape settings or image windows.
        #  Subsequent show and hide are controlled with Buttons in setup_buttons().
        self.shape_settings_win.withdraw()
        self.img_window['shaped'].withdraw()

    def shape_win_setup(self) -> None:
        """
        Shape settings and reporting frames, buttons, configuration,
         keybindings, and grids.
        """

        def no_exit_on_x():
            """Notice in Terminal. Called from .protocol() in loop."""
            print('The Shape window cannot be closed from the window bar.\n'
                  'It can be closed with the "Close" button.\n'
                  'You may quit the program from the OpenCV Settings Report window bar'
                  ' or Esc or Ctrl-Q key.'
                  )

        self.separator = ttk.Separator(master=self.shape_selectors_frame,
                                       orient='horizontal')

        self.shape_settings_win.title(const.WIN_NAME['shape_report'])
        self.shape_settings_win.resizable(False, False)

        # Need to position window toward right of screen, overlapping
        #   contour settings window, so that it does not cover the img window.
        # Need to position window toward right of screen, overlapping
        #   contour settings window, so that it does not cover the img window.
        self.shape_settings_win.geometry(f'+{self.winfo_screenwidth() - 800}+100')

        # Configure Shapes report window to match that of app (contour) window.
        self.shape_settings_win.config(
            bg=const.MASTER_BG,  # gray80 matches report_contour() txt fg.
            # bg=const.CBLIND_COLOR_TK['sky blue'],  # for dev.
            highlightthickness=5,
            highlightcolor=const.CBLIND_COLOR_TK['yellow'],
            highlightbackground='grey65'
        )

        self.shape_report_frame.configure(relief='flat',
                                          bg=const.CBLIND_COLOR_TK['sky blue']
                                          )  # bg won't show with grid sticky EW.
        self.shape_report_frame.columnconfigure(1, weight=1)
        self.shape_report_frame.rowconfigure(0, weight=1)

        self.shape_selectors_frame.configure(relief='raised',
                                             bg=const.DARK_BG,
                                             borderwidth=5, )
        self.shape_selectors_frame.columnconfigure(0, weight=1)
        self.shape_selectors_frame.columnconfigure(1, weight=1)

        self.img_window['shaped'].geometry(f'+{self.winfo_screenwidth() - 830}+150')
        self.img_window['shaped'].title(const.WIN_NAME['shapes'])
        self.img_window['shaped'].protocol('WM_DELETE_WINDOW', no_exit_on_x)
        self.img_window['shaped'].columnconfigure(0, weight=1)
        self.img_window['shaped'].columnconfigure(1, weight=1)

        self.shapeimg_lbl = tk.Label(master=self.img_window['shaped'])

        self.circle_msg_lbl.config(
            text=('Note: Circles are found in the filtered image or'
                  ' an Otsu threshold of it, not from previously found contours.'),
            **const.LABEL_PARAMETERS)

        self.shape_report_frame.grid(column=0, row=0,
                                     columnspan=2,
                                     padx=5, pady=5,
                                     sticky=tk.EW)
        self.shape_selectors_frame.grid(column=0, row=1,
                                        columnspan=2,
                                        padx=5, pady=(0, 5),
                                        ipadx=4, ipady=4,
                                        sticky=tk.EW)

        self.circle_msg_lbl.grid(column=0, row=4,
                                 columnspan=2,
                                 padx=5,
                                 pady=(7, 0),
                                 sticky=tk.EW)

        def save_shape_cmd():
            if self.radio_val['find_shape_in'].get() == 'threshold':
                utils.save_settings_and_img(
                    img2save=self.tkimg['shaped'],
                    txt2save=self.shape_settings_txt,
                    caller='thresh_shape')
            else:  # == 'canny'
                utils.save_settings_and_img(
                    img2save=self.tkimg['shaped'],
                    txt2save=self.shape_settings_txt,
                    caller='canny_shape')

        # Note that ttk.Styles are defined in ContourViewer.setup_styles().
        self.resetshape_button.configure(text='Set contour defaults',
                                         style='My.TButton',
                                         width=0,
                                         command=self.set_shape_defaults)

        self.circle_defaults_button.configure(text='Set Circle defaults',
                                              style='My.TButton',
                                              width=0,
                                              command=self.set_shape_defaults)

        self.saveshape_button.configure(text='Save shape settings and image',
                                        style='My.TButton',
                                        width=0,
                                        command=save_shape_cmd)

        # Reset button should be centered under slider labels.
        # Save button should be on same row (bottom of frame), right side.
        if const.MY_OS == 'lin, dar':
            padx = (0, 60)
        else:  # is Windows
            padx = (0, 40)
        self.resetshape_button.grid(column=0, row=3,
                                    padx=(10, 0),
                                    pady=(0, 5),
                                    sticky=tk.W)
        self.circle_defaults_button.grid(column=0, row=3,
                                         padx=padx,
                                         pady=(0, 5),
                                         sticky=tk.E)
        self.saveshape_button.grid(column=1, row=3,
                                   padx=(0, 5),
                                   pady=(0, 5),
                                   sticky=tk.E)

    def setup_styles(self) -> None:
        """
        Configure ttk.Style for Buttons and Comboboxes.
        Called by __init__ and ShapeViewer.shape_win_setup().

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

    def setup_buttons(self) -> None:
        """
        Assign and grid Buttons in the main (app) and shape windows.
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

        def show_shapes_windows():
            self.shape_settings_win.deiconify()
            self.img_window['shaped'].deiconify()

        def hide_shapes_windows():
            self.shape_settings_win.withdraw()
            self.img_window['shaped'].withdraw()

        if const.MY_OS in 'lin, win':
            label_font = const.WIDGET_FONT
        else:  # is macOS
            label_font = 'TkTooltipFont', 11

        button_params = dict(
            style='My.TButton',
            width=0,
        )

        reset_btn = ttk.Button(text='Reset settings',
                               command=self.set_contour_defaults,
                               **button_params)

        save_btn_label = tk.Label(text='Save settings & contoured image for:',
                                  font=label_font,
                                  bg=const.MASTER_BG)
        save_th_btn = ttk.Button(text='Threshold',
                                 command=save_th_settings,
                                 **button_params)
        save_canny_btn = ttk.Button(text='Canny',
                                    command=save_can_settings,
                                    **button_params)

        show_shapes_btn = ttk.Button(text='Show Shapes windows',
                                     command=show_shapes_windows,
                                     **button_params)
        hide_shapes_btn = ttk.Button(text='Hide Shapes windows',
                                     command=hide_shapes_windows,
                                     **button_params)

        # Buttons for Shape window; are configured and gridded in
        #  shape_win_setup().
        self.resetshape_button = ttk.Button(master=self.shape_settings_win)
        self.circle_defaults_button = ttk.Button(master=self.shape_settings_win)
        self.saveshape_button = ttk.Button(master=self.shape_settings_win)

        # Widget grid for the main window.
        reset_btn.grid(column=0, row=2,
                       padx=(70, 0),
                       pady=(0, 5),
                       sticky=tk.W)

        save_btn_label.grid(column=0, row=3,
                            padx=(10, 0),
                            pady=(0, 5),
                            sticky=tk.W)
        save_th_btn.grid(column=0, row=3,
                         padx=(0, 75),
                         pady=(0, 5),
                         sticky=tk.E)
        save_canny_btn.grid(column=0, row=3,
                            padx=(0, 15),
                            pady=(0, 5),
                            sticky=tk.E)

        show_shapes_btn.grid(column=1, row=2,
                             padx=(0, 5),
                             pady=(0, 5),
                             sticky=tk.E)
        hide_shapes_btn.grid(column=1, row=3,
                             padx=(0, 5),
                             pady=(0, 5),
                             sticky=tk.E)

    def display_input_images(self) -> None:
        """
        Converts input image and its grayscale to tk image formate and
        displays them as panels gridded in their toplevel window.
        Calls manage.tkimage(), which applies scaling, cv2 -> tk array
        conversion, and updates the panel Label's image parameter.
        """

        # Display the input image and its grayscale; both are static, so
        #  do not need updating, but retain the image display statement
        #  structure of processed images that do need updating.
        # Note: Use 'self' to scope the ImageTk.PhotoImage in the Class,
        #  otherwise it will/may not show b/c of garbage collection.
        self.tkimg['input'] = manage.tk_image(INPUT_IMG)
        self.img_label['input'].configure(image=self.tkimg['input'])
        self.img_label['input'].grid(column=0, row=0,
                                     padx=5, pady=5)

        self.tkimg['gray'] = manage.tk_image(GRAY_IMG)
        self.img_label['gray'].configure(image=self.tkimg['gray'])
        self.img_label['gray'].grid(column=1, row=0,
                                    padx=5, pady=5)

    def config_sliders(self) -> None:
        """
        Configure arguments for all Scale() sliders in both the main
        (contour) and shape settings windows. Also set mouse button
        bindings for all sliders.

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

        # Sliders for shape settings ##########################################
        self.slider['epsilon_lbl'].configure(text='% polygon contour length\n'
                                                  '(epsilon coef.):',
                                             **const.LABEL_PARAMETERS)
        self.slider['epsilon'].configure(from_=0.001, to=0.06,
                                         resolution=0.001,
                                         tickinterval=0.01,
                                         variable=self.slider_val['epsilon'],
                                         **const.SHAPE_SCALE_PARAMETERS)

        self.slider['circle_mindist_lbl'].configure(text='Minimum px dist between circles:',
                                                    **const.LABEL_PARAMETERS)
        self.slider['circle_mindist'].configure(from_=10, to=200,
                                                resolution=1,
                                                tickinterval=20,
                                                variable=self.slider_val['circle_mindist'],
                                                **const.SHAPE_SCALE_PARAMETERS)

        self.slider['circle_param1_lbl'].configure(text='cv2.HoughCircles, param1:',
                                                   **const.LABEL_PARAMETERS)
        self.slider['circle_param1'].configure(from_=100, to=500,
                                               resolution=100,
                                               tickinterval=100,
                                               variable=self.slider_val['circle_param1'],
                                               **const.SHAPE_SCALE_PARAMETERS)

        self.slider['circle_param2_lbl'].configure(text='cv2.HoughCircles, param2:',
                                                   **const.LABEL_PARAMETERS)
        self.slider['circle_param2'].configure(from_=0.1, to=0.98,
                                               resolution=0.1,
                                               tickinterval=0.1,
                                               variable=self.slider_val['circle_param2'],
                                               **const.SHAPE_SCALE_PARAMETERS)

        self.slider['circle_minradius_lbl'].configure(text='cv2.HoughCircles, min px radius):',
                                                      **const.LABEL_PARAMETERS)
        self.slider['circle_minradius'].configure(from_=10, to=200,
                                                  resolution=10,
                                                  tickinterval=20,
                                                  variable=self.slider_val['circle_minradius'],
                                                  **const.SHAPE_SCALE_PARAMETERS)

        self.slider['circle_maxradius_lbl'].configure(text='cv2.HoughCircles, max px radius:',
                                                      **const.LABEL_PARAMETERS)
        self.slider['circle_maxradius'].configure(from_=10, to=1000,
                                                  resolution=10,
                                                  tickinterval=100,
                                                  variable=self.slider_val['circle_maxradius'],
                                                  **const.SHAPE_SCALE_PARAMETERS)

        # To avoid grabbing all the intermediate values between normal
        #  click and release movement, bind sliders to call the main
        #  processing and reporting function only on left button release.
        # All sliders are here bound to process_all(), but if the processing
        #   overhead of that is too great, then can add conditions in the loop
        #   to bind certain groups or individual sliders to more restrictive
        #   processing functions.
        for name, widget in self.slider.items():
            if '_lbl' not in name:
                widget.bind('<ButtonRelease-1>', self.process_all)
        # for _s in [_s for _s in self.slider if '_lbl' not in _s]:
        #     self.slider[_s].bind('<ButtonRelease-1>', self.process_all)

    def config_comboboxes(self) -> None:
        """
        Configure arguments and mouse button bindings for all Comboboxes
        in both the main (contour) and shape settings windows.

        Returns: None
        """

        # Different Combobox widths are needed to account for font widths
        #  and padding in different systems.
        width_correction = 2 if const.MY_OS == 'win' else 0  # is Linux or macOS

        self.cbox['choose_morphop_lbl'].config(text='Reduce noise, morphology operator:',
                                               **const.LABEL_PARAMETERS)
        self.cbox['choose_morphop'].config(textvariable=self.cbox_val['morphop_pref'],
                                           width=18 + width_correction,
                                           values=('cv2.MORPH_OPEN',  # cv2 returns 2
                                                   'cv2.MORPH_CLOSE',  # cv2 returns 3
                                                   'cv2.MORPH_GRADIENT',  # cv2 returns 4
                                                   'cv2.MORPH_BLACKHAT',  # cv2 returns 6
                                                   'cv2.MORPH_HITMISS'),  # cv2 returns 7
                                           **const.COMBO_PARAMETERS)
        self.cbox['choose_morphop'].bind('<<ComboboxSelected>>',
                                         func=self.process_all)

        self.cbox['choose_morphshape_lbl'].config(text='... shape:',
                                                  **const.LABEL_PARAMETERS)
        self.cbox['choose_morphshape'].config(textvariable=self.cbox_val['morphshape_pref'],
                                              width=16 + width_correction,
                                              values=('cv2.MORPH_RECT',  # cv2 returns 0
                                                      'cv2.MORPH_CROSS',  # cv2 returns 1
                                                      'cv2.MORPH_ELLIPSE'),  # cv2 returns 2
                                              **const.COMBO_PARAMETERS)
        self.cbox['choose_morphshape'].bind('<<ComboboxSelected>>',
                                            func=self.process_all)

        self.cbox['choose_border_lbl'].config(text='Border type:',
                                              **const.LABEL_PARAMETERS)
        self.cbox['choose_border'].config(textvariable=self.cbox_val['border_pref'],
                                          width=22 + width_correction,
                                          values=(
                                              'cv2.BORDER_REFLECT_101',  # cv2 returns 4, default
                                              'cv2.BORDER_REFLECT',  # cv2 returns 2
                                              'cv2.BORDER_REPLICATE',  # cv2 returns 1
                                              'cv2.BORDER_ISOLATED'),  # cv2 returns 16
                                          **const.COMBO_PARAMETERS)
        self.cbox['choose_border'].bind(
            '<<ComboboxSelected>>', lambda _: self.process_all())

        self.cbox['choose_filter_lbl'].config(text='Filter type:',
                                              **const.LABEL_PARAMETERS)
        self.cbox['choose_filter'].config(textvariable=self.cbox_val['filter_pref'],
                                          width=14 + width_correction,
                                          values=(
                                              'cv2.blur',  # is default, 0, a box filter.
                                              'cv2.bilateralFilter',  # cv2 returns 1
                                              'cv2.GaussianBlur',  # cv2 returns 2
                                              'cv2.medianBlur'),  # cv2 returns 3
                                          **const.COMBO_PARAMETERS)
        self.cbox['choose_filter'].bind(
            '<<ComboboxSelected>>', lambda _: self.process_all())

        self.cbox['choose_th_type_lbl'].config(text='Threshold type:',
                                               **const.LABEL_PARAMETERS)
        self.cbox['choose_th_type'].config(textvariable=self.cbox_val['th_type_pref'],
                                           width=26 + width_correction,
                                           values=('cv2.THRESH_BINARY',  # cv2 returns 0
                                                   'cv2.THRESH_BINARY_INVERSE',  # cv2 returns 1
                                                   'cv2.THRESH_OTSU',  # cv2 returns 8
                                                   'cv2.THRESH_OTSU_INVERSE',  # cv2 returns 9
                                                   'cv2.THRESH_TRIANGLE',  # cv2 returns 16
                                                   'cv2.THRESH_TRIANGLE_INVERSE'),  # returns 17
                                           **const.COMBO_PARAMETERS)
        self.cbox['choose_th_type'].bind(
            '<<ComboboxSelected>>', lambda _: self.process_contours())

        self.cbox['choose_c_method_lbl'].config(text='... method:',
                                                **const.LABEL_PARAMETERS)
        self.cbox['choose_c_method'].config(textvariable=self.cbox_val['c_method_pref'],
                                            width=26 + width_correction,
                                            values=('cv2.CHAIN_APPROX_NONE',  # cv2 returns 1
                                                    'cv2.CHAIN_APPROX_SIMPLE',  # cv2 returns 2
                                                    'cv2.CHAIN_APPROX_TC89_L1',  # cv2 returns 3
                                                    'cv2.CHAIN_APPROX_TC89_KCOS'),  # cv2 returns 4
                                            **const.COMBO_PARAMETERS)
        self.cbox['choose_c_method'].bind(
            '<<ComboboxSelected>>', lambda _: self.process_contours())

        # Shape Comboboxes:
        def process_shape_selection(event):
            """
            Use 'Circle' condition to automatically set default circle
            slider values, thus avoiding need to use "Set Circle defaults"
            button each time 'Circle' is selected. Without this, the
            circle sliders are all set to minimum values following the
            DISABLED state invoked when a different shape is selected.
            """
            self.process_all()
            if self.cbox_val['polygon'].get() == 'Circle':
                self.set_shape_defaults()

        self.cbox['choose_shape_lbl'].config(text='Select shape to find:',
                                             **const.LABEL_PARAMETERS)
        self.cbox['choose_shape'].config(
            textvariable=self.cbox_val['polygon'],
            width=12 + width_correction,
            values=('Triangle',
                    'Rectangle',
                    'Pentagon',
                    'Hexagon',
                    'Heptagon',
                    'Octagon',
                    '5-pointed Star',
                    'Circle'),
            **const.COMBO_PARAMETERS)
        self.cbox['choose_shape'].bind('<<ComboboxSelected>>',
                                       func=process_shape_selection)
        self.cbox['choose_shape'].current(0)

    def config_radiobuttons(self) -> None:
        """
        Configure arguments for all Radiobutton() widgets in both the
        main (contour) and shape settings windows.

        Returns: None
        """

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

        # Shape window Radiobuttons
        self.radio['shape_hull_lbl'].config(text='Find shapes as hull?',
                                            **const.LABEL_PARAMETERS)
        self.radio['shape_hull_yes'].configure(
            text='Yes',
            variable=self.radio_val['hull_shape'],
            value='yes',
            command=self.process_all,
            **const.RADIO_PARAMETERS
        )
        self.radio['shape_hull_no'].configure(
            text='No',
            variable=self.radio_val['hull_shape'],
            value='no',
            command=self.process_all,
            **const.RADIO_PARAMETERS,
        )

        self.radio['find_shape_lbl'].config(text='Find shapes in contours from:',
                                            **const.LABEL_PARAMETERS)
        self.radio['find_shape_in_thresh'].configure(
            text='Threshold',
            variable=self.radio_val['find_shape_in'],
            value='threshold',
            command=self.process_all,
            **const.RADIO_PARAMETERS
        )
        self.radio['find_shape_in_canny'].configure(
            text='Canny',
            variable=self.radio_val['find_shape_in'],
            value='canny',
            command=self.process_all,
            **const.RADIO_PARAMETERS
        )

        self.radio['find_circle_lbl'].config(text='Find Hough circles in:',
                                             **const.LABEL_PARAMETERS)
        self.radio['find_circle_in_th'].configure(
            text='Threshold img',
            variable=self.radio_val['find_circle_in'],
            value='thresholded',
            command=self.process_all,
            **const.RADIO_PARAMETERS
        )
        self.radio['find_circle_in_filtered'].configure(
            text='Filtered img',
            variable=self.radio_val['find_circle_in'],
            value='filtered',
            command=self.process_all,
            **const.RADIO_PARAMETERS
        )

    def grid_contour_widgets(self) -> None:
        """
        Developer: Grid as a group to make clear spatial relationships.
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

            # Used for some Combobox and Radiobutton widgets.
            grid_params = dict(
                padx=(8, 0),
                pady=(5, 0),
                sticky=tk.W)

            # Parameters for specific widgets:
            shape_cbox_param = dict(
                padx=(0, 15),
                pady=(5, 0),
                sticky=tk.E)
            filter_cbox_param = dict(
                padx=(0, 15),
                pady=(5, 0),
                sticky=tk.E)

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

            # Parameters for specific widgets:
            shape_cbox_param = dict(
                padx=(245, 0),
                pady=(5, 0),
                sticky=tk.W)
            filter_cbox_param = dict(
                padx=(245, 0),
                pady=(5, 0),
                sticky=tk.W)

        # Special cases where each platform is different. Messy, but, oh well.
        if const.MY_OS == 'lin':
            c_method_lbl_params = dict(
                padx=(145, 0),
                pady=(5, 0),
                sticky=tk.W)
            shape_lbl_param = dict(
                padx=(0, 155),
                pady=(5, 0),
                sticky=tk.E)
            filter_lbl_param = dict(
                padx=(0, 140),
                pady=(5, 0),
                sticky=tk.E)

        elif const.MY_OS == 'win':
            c_method_lbl_params = dict(
                padx=(0, 240),
                pady=(5, 0),
                sticky=tk.E)
            shape_lbl_param = dict(
                padx=(0, 170),
                pady=(5, 0),
                sticky=tk.E)
            filter_lbl_param = dict(
                padx=(0, 160),
                pady=(5, 0),
                sticky=tk.E)

        else:  # is macOS
            c_method_lbl_params = dict(
                padx=(160, 0),
                pady=(4, 0),
                sticky=tk.W)
            shape_lbl_param = dict(
                padx=(0, 180),
                pady=(5, 0),
                sticky=tk.E)
            filter_lbl_param = dict(
                padx=(0, 140),
                pady=(5, 0),
                sticky=tk.E)

        # Widgets gridded in the self.contour_selectors_frame Frame.
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
                                                **shape_lbl_param)

        self.cbox['choose_morphshape'].grid(column=1, row=2,
                                            **shape_cbox_param)

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
                                            **filter_lbl_param)
        self.cbox['choose_filter'].grid(column=1, row=6,
                                        **filter_cbox_param)

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

        self.cbox['choose_c_method_lbl'].grid(column=1, row=13,
                                              **c_method_lbl_params)
        self.cbox['choose_c_method'].grid(column=1, row=13,
                                          padx=(0, 15),
                                          pady=(5, 0),
                                          sticky=tk.E)

        self.slider['c_limit_lbl'].grid(column=0, row=15,
                                        **label_grid_params)
        self.slider['c_limit'].grid(column=1, row=15,
                                    **slider_grid_params)

    def grid_shape_widgets(self) -> None:
        """
        Grid all selector widgets in the shape_selectors_frame Frame.

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

        self.cbox['choose_shape_lbl'].grid(column=0, row=0,
                                           **label_grid_params)
        self.cbox['choose_shape'].grid(column=1, row=0,
                                       **selector_grid_params)

        self.radio['shape_hull_lbl'].grid(column=1, row=0,
                                          padx=(0, 110),
                                          pady=(7, 0),
                                          sticky=tk.E)
        self.radio['shape_hull_yes'].grid(column=1, row=0,
                                          padx=(0, 70),
                                          pady=(7, 0),
                                          sticky=tk.E)
        self.radio['shape_hull_no'].grid(column=1, row=0,
                                         padx=(0, 35),
                                         pady=(7, 0),
                                         sticky=tk.E)

        self.radio['find_shape_lbl'].grid(column=0, row=1,
                                          **label_grid_params)
        self.radio['find_shape_in_thresh'].grid(column=1, row=1,
                                                **selector_grid_params)
        self.radio['find_shape_in_canny'].grid(column=1, row=1,
                                               padx=(80, 0),
                                               pady=(7, 0),
                                               sticky=tk.W)

        self.slider['epsilon_lbl'].grid(column=0, row=2,
                                        **label_grid_params)
        self.slider['epsilon'].grid(column=1, row=2,
                                    **selector_grid_params)

        self.separator.grid(column=0, row=3,
                            columnspan=2,
                            padx=10,
                            pady=(8, 5),
                            sticky=tk.EW)

        self.radio['find_circle_lbl'].grid(column=0, row=5,
                                           **label_grid_params)
        self.radio['find_circle_in_th'].grid(column=1, row=5,
                                             **selector_grid_params)
        self.radio['find_circle_in_filtered'].grid(column=1, row=5,
                                                   padx=100,
                                                   pady=(7, 0),
                                                   sticky=tk.W)

        self.slider['circle_mindist_lbl'].grid(column=0, row=6,
                                               **label_grid_params)
        self.slider['circle_mindist'].grid(column=1, row=6,
                                           **selector_grid_params)

        self.slider['circle_param1_lbl'].grid(column=0, row=7,
                                              **label_grid_params)
        self.slider['circle_param1'].grid(column=1, row=7,
                                          **selector_grid_params)

        self.slider['circle_param2_lbl'].grid(column=0, row=8,
                                              **label_grid_params)
        self.slider['circle_param2'].grid(column=1, row=8,
                                          **selector_grid_params)

        self.slider['circle_minradius_lbl'].grid(column=0, row=9,
                                                 **label_grid_params)
        self.slider['circle_minradius'].grid(column=1, row=9,
                                             **selector_grid_params)

        self.slider['circle_maxradius_lbl'].grid(column=0, row=10,
                                                 **label_grid_params)
        self.slider['circle_maxradius'].grid(column=1, row=10,
                                             **selector_grid_params)

    def grid_img_labels(self) -> None:
        """
        Grid all image Labels inherited from ProcessImage().
        Labels' 'master' argument for the img window is defined in
        ProcessImage.setup_image_windows(). Label 'image' param is
        updated with .configure() in each PI processing method.
        """
        panel_left = dict(
            column=0, row=0,
            padx=5, pady=5,
            sticky=tk.NSEW)
        panel_right = dict(
            column=1, row=0,
            padx=5, pady=5,
            sticky=tk.NSEW)

        self.img_label['contrast'].grid(**panel_left)
        self.img_label['redux'].grid(**panel_right)

        self.img_label['filter'].grid(**panel_right)

        self.img_label['thresh'].grid(**panel_left)
        self.img_label['th_contour'].grid(**panel_right)

        self.img_label['canny'].grid(**panel_left)
        self.img_label['can_contour'].grid(**panel_right)

        self.img_label['circled_th'].grid(**panel_left)
        self.img_label['circled_can'].grid(**panel_left)
        self.img_label['shaped'].grid(**panel_left)

    def set_contour_defaults(self) -> None:
        """
        Sets controller widgets at startup. Called from "Reset" button.
        """

        # Set defaults for contour selector settings:
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
        self.cbox['choose_c_method'].current(1)  # cv2.CHAIN_APPROX_SIMPLE

        # Set/Reset Radiobutton widgets:
        self.radio['hull_no'].select()
        self.radio['c_type_area'].select()
        self.radio['c_mode_external'].select()

        # Apply the default settings.
        self.process_all()

    def set_shape_defaults(self) -> None:
        """
        Set default values for selectors. Called at startup.
        Returns: None
        """
        # Set/Reset Combobox widgets.
        # When 'Circle' is selected shape, do not reset it.
        if self.cbox_val['polygon'].get() != 'Circle':
            self.cbox_val['polygon'].set('Triangle')

        # Set/Reset Scale widgets.
        self.slider['epsilon'].set(0.01)
        self.slider['circle_mindist'].set(100)
        self.slider['circle_param1'].set(300)
        self.slider['circle_param2'].set(0.9)
        self.slider['circle_minradius'].set(20)
        self.slider['circle_maxradius'].set(500)

        # Set/Reset Radiobutton widgets:
        self.radio['shape_hull_no'].select()
        self.radio['find_shape_in_thresh'].select()
        self.radio['find_circle_in_filtered'].select()

        self.process_shapes()

    def report_contour(self) -> None:
        """
        Write the current settings and cv2 metrics in a Text widget of
        the report_frame. Same text is printed in Terminal from "Save"
        button. Called from __init__ and process_*() methods.
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
            f' beta={beta} (new alpha SD {new_std})\n'
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
            f'{"Contour chain size:".ljust(21)}type is {c_type},\n'
            f'{tab}minimum is {c_limit} pixels\n'
            f'{"# contours selected:".ljust(21)}Threshold {num_th_c_select}'
            f' (from {num_th_c_all} total)\n'
            f'{tab}Canny {num_canny_c_select} (from {num_canny_c_all} total)\n'
        )

        utils.display_report(frame=self.contour_report_frame,
                             report=self.contour_settings_txt)

    def report_shape(self) -> None:
        """
        Write the current settings and cv2 metrics in a Text widget of
        the shape_report_frame of the shape_setting_win.
        Same text format is printed in Terminal from "Save"
        button. Called from __init__ and process_shaper().
        """

        epsilon_coef = self.slider_val['epsilon'].get()
        epsilon_pct = round(self.slider_val['epsilon'].get() * 100, 2)
        shape_found_in = self.radio_val['find_shape_in'].get()
        hough_img = 'n/a'
        use_image = self.radio_val['find_circle_in'].get()
        mindist = self.slider_val['circle_mindist'].get()
        param1 = self.slider_val['circle_param1'].get()
        param2 = self.slider_val['circle_param2'].get()
        min_radius = self.slider_val['circle_minradius'].get()
        max_radius = self.slider_val['circle_maxradius'].get()
        poly_choice = self.cbox_val['polygon'].get()

        # Do not allow hull functions or reporting when Circle is selected.
        if poly_choice == 'Circle':
            self.radio_val['hull_shape'].set('no')
            app.update_idletasks()
            # Need to specify text based on selections.
            if use_image == 'thresholded':
                shape_found_in = ('\n     Hough transform of an Otsu threshold '
                                  'from the Filtered')
                hough_img = '...an Otsu threshold of the Filtered image.'
            else:  # is 'filtered'
                shape_found_in = '\n     Hough transform of the Filtered'
                hough_img = '...the Filtered image.'

        if self.radio_val['hull_shape'].get() == 'yes':
            shape_type = 'Selected hull shape'
        else:
            shape_type = 'Selected shape'

        # Text is formatted for clarity in window, terminal, and saved file.
        justify = 19
        indent = " ".ljust(justify)

        self.shape_settings_txt = (
            f'{"cv2.approxPolyDP:".ljust(justify)}epsilon coefficient is {epsilon_coef}\n'
            f'{indent}({epsilon_pct}% contour length, cv2.arcLength)\n'
            f'{indent}closed=True\n'
            f'{"cv2.HoughCircles:".ljust(justify)}image={hough_img}\n'
            f'{indent}method=cv2.HOUGH_GRADIENT_ALT\n'
            f'{indent}dp=1.5\n'
            f'{indent}minDist={mindist}\n'
            f'{indent}param1={param1}\n'
            f'{indent}param2={param2}\n'
            f'{indent}minRadius={min_radius}\n'
            f'{indent}maxRadius={max_radius}\n\n'
            f'{shape_type}: {poly_choice}, found: {self.num_shapes} in the '
            f'{shape_found_in} image.\n'
        )

        utils.display_report(frame=self.shape_report_frame,
                             report=self.shape_settings_txt)

    def process_all(self, event=None) -> None:
        """
        Runs all image processing methods from ProcessImage() and the
        settings report_contour.
        Calls adjust_contrast(), reduce_noise(), filter_image(), and
        contour_threshold() from ProcessImage.
        Calls report_contour() from ContourViewer.
        Args:
            event: The implicit mouse button event.

        Returns: *event* as a formality; is functionally None.

        """
        self.adjust_contrast()
        self.reduce_noise()
        self.filter_image()
        self.contour_threshold(event)
        self.contour_canny(event)
        self.size_the_contours(contour_pointset=self.contours['selected_found_thresh'],
                               called_by='thresh sized')
        self.size_the_contours(self.contours['selected_found_canny'], 'canny sized')
        self.report_contour()
        self.process_shapes(event)

        return event

    def process_contours(self, event=None) -> None:
        """
        Calls contour_threshold() from ProcessImage.
        Calls report_contour() from ContourViewer.
        Args:
            event: The implicit mouse button event.

        Returns: *event* as a formality; is functionally None.

        """
        self.contour_threshold(event)
        self.contour_canny(event)
        self.size_the_contours(contour_pointset=self.contours['selected_found_thresh'],
                               called_by='thresh sized')
        self.size_the_contours(self.contours['selected_found_canny'], 'canny sized')
        self.report_contour()
        # cv2.HoughCircles doesn't use contours, so can skip shape processing.
        if self.cbox_val['polygon'].get() != 'Circle':
            self.process_shapes(event)

        return event

    def toggle_circle_vs_shapes(self) -> None:
        """
        Make selector options obvious for the user depending on whether
        'Circle' shape is selected or not; gray out and disable those
        that are not relevant to the selection.

        Returns: None
        """
        grayout = const.MASTER_BG
        fg_default = const.CBLIND_COLOR_TK['yellow']

        if self.cbox_val['polygon'].get() == 'Circle':
            self.radio['find_shape_lbl'].config(fg=grayout)
            self.radio['find_shape_in_thresh'].config(state=tk.DISABLED)
            self.radio['find_shape_in_canny'].config(state=tk.DISABLED)
            self.radio['shape_hull_lbl'].config(fg=grayout)
            self.radio['shape_hull_yes'].config(state=tk.DISABLED)
            self.radio['shape_hull_no'].config(state=tk.DISABLED)
            self.slider['epsilon_lbl'].config(fg=grayout)
            self.slider['epsilon'].config(state=tk.DISABLED, fg=grayout)
            self.resetshape_button.configure(state=tk.DISABLED)

            self.circle_defaults_button.configure(state=tk.NORMAL)
            self.radio['find_circle_lbl'].config(fg=fg_default)
            self.radio['find_circle_in_th'].config(state=tk.NORMAL)
            self.radio['find_circle_in_filtered'].config(state=tk.NORMAL)
            for name, widget in self.slider.items():
                if 'circle' in name:
                    widget.config(state=tk.NORMAL, fg=fg_default)
        else:
            self.radio['find_shape_lbl'].config(fg=fg_default)
            self.radio['find_shape_in_thresh'].config(state=tk.NORMAL)
            self.radio['find_shape_in_canny'].config(state=tk.NORMAL)
            self.radio['shape_hull_lbl'].config(fg=fg_default)
            self.radio['shape_hull_yes'].config(state=tk.NORMAL)
            self.radio['shape_hull_no'].config(state=tk.NORMAL)
            self.slider['epsilon_lbl'].config(fg=fg_default)
            self.slider['epsilon'].config(state=tk.NORMAL, fg=fg_default)
            self.resetshape_button.configure(state=tk.NORMAL)

            self.circle_defaults_button.configure(state=tk.DISABLED)
            self.radio['find_circle_lbl'].config(fg=grayout)
            self.radio['find_circle_in_th'].config(state=tk.DISABLED)
            self.radio['find_circle_in_filtered'].config(state=tk.DISABLED)
            for name, widget in self.slider.items():
                if 'circle' in name:
                    widget.config(state=tk.DISABLED, fg=grayout)

    def process_shapes(self, event=None) -> None:
        """
        A handler for the command kw and button binding for the settings
        control widgets to call multiple methods. Also configures the
        text and disables/enables widgets with respect to the 'Circle'
        shape option.
        Calls select_shape() and report_shape().
        The contours object passed to select_shape() are those inherited
        from ProcessImage().

        Args:
            event: An implicit mouse button event.

        Returns: *event* as a formality; is functionally None.
        """

        self.toggle_circle_vs_shapes()

        if self.radio_val['find_shape_in'].get() == 'threshold':
            contours = self.contours['selected_found_thresh']
        else:  # is 'canny'
            contours = self.contours['selected_found_canny']

        self.select_shape(contours)
        self.report_shape()
        self.update_idletasks()

        return event


if __name__ == "__main__":
    # Program exits here if the system platform or Python version check fails.
    utils.check_platform()
    vcheck.minversion('3.7')
    arguments = manage.arguments()

    # All checks are good, so grab as a 'global' the dictionary of
    #   command line argument values to define often used values...
    infile_dict = manage.infile()
    INPUT_IMG = infile_dict['input_img']
    GRAY_IMG = infile_dict['gray_img']
    LINE_THICKNESS = infile_dict['line_thickness']

    try:
        app = ImageViewer()
        app.title('OpenCV Contour Settings Report')
        app.resizable(False, False)
        print(f'{Path(__file__).name} is now running...')
        app.mainloop()
    except KeyboardInterrupt:
        print('*** User quit the program from Terminal/Console ***\n')
