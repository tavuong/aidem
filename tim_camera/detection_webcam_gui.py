"""GUI for object detection with tensorflow and distance calculation."""

__version__ = "1.0.0"
__author__ = "Tim Rosenkranz"
__email__ = "tim.rosenkranz:stud.uni-frankfurt.de"
__credits__ = "Special thanks to The Anh Vuong who came up with the original idea."

# Description:
# This program uses a TensorFlow Lite model to perform object detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video and calculates the distance between each of these objects.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
# The code is also based off a raspberry pi tutorial for object detection:
# https://tutorials-raspberrypi.de/raspberry-pi-objekterkennung-mittels-tensorflow-und-kamera/
#

import os
import cv2
import tkinter as tk
import tkinter.messagebox as mb
from datetime import date
from Detection_Models import LiveDetection, VideoDetection, LiveFaceSwap


class DetectionGUI(tk.Frame):
    """GUI class"""

    def __init__(self, master=None):
        """GUI  for object detection with tensorflow and distance calculation.

        :param master: Tkinter root
        """

        super().__init__(master)

        if master is None:
            self.master = tk.Tk()
        else:
            self.master = master

        self.master.title("Object detection")
        self.master.geometry("500x200")

        # Create menubar
        self.menubar = tk.Menu(self.master)
        main_menu = tk.Menu(self.menubar, tearoff=0)
        main_menu.add_command(label="Calibrate", command=lambda: self.start_calibrate(0))
        main_menu.add_command(label="Chess calibration", command=lambda: self.start_calibrate(1))
        main_menu.add_separator()
        main_menu.add_command(label="Options", command=self.options)
        main_menu.add_command(label="Help", command=self.help)
        main_menu.add_separator()
        main_menu.add_command(label="Quit", command=self.__del__)
        self.menubar.add_cascade(label="Menu", menu=main_menu)
        self.master.config(menu=self.menubar)

        self.create_widgets()
        self.detection_obj = None

        # Initialise calibration variables
        self.focal = 0
        self.cal_vals = [None, None, None]
        self.calib_chess = [0, 0]

        # Initialise option variables
        self.autosave = False
        self.debug_mode = False
        self.detect_select = []
        self.no_detect_select = []
        self.video_folder = ""

        # Declare various widgets in advance
        self.options_master = None
        self.calib_master = None
        self.calib_entry_name = None
        self.calib_entry_distance = None
        self.calib_entry_width = None

        # Initialise various textvariables for entry widgets
        self.calib_width_textvar = tk.StringVar(value="")
        self.calib_dist_textvar = tk.StringVar(value="")
        self.calib_name_textvar = tk.StringVar(value="")
        self.calib_cols_textvar = tk.StringVar(value="")
        self.calib_rows_textvar = tk.StringVar(value="")
        self.options_select_textvar = tk.StringVar(value="")
        self.options_no_select_textvar = tk.StringVar(value="")
        self.options_video_folder_textvar = tk.StringVar(value="")
        self.options_autosave_boolvar = tk.BooleanVar()
        self.options_debug_boolvar = tk.BooleanVar()

    def create_widgets(self):
        """
        Create the needed widgets (Buttons, Labels) at startup.
        """

        # Main menu widgets
        self.live_detection_button = tk.Button(self.master, text="Start live detection", command=self.live_detection)
        self.live_detection_button.grid(column=0, row=0)
        self.video_detection_button = tk.Button(self.master, text="Start video detection", command=self.video_detection)
        self.video_detection_button.grid(column=1, row=0)
        self.face_swap_button = tk.Button(self.master, text="Start face detections", command=self.face_swapping)
        self.face_swap_button.grid(column=2, row=0)

        # Detection controllign buttons
        self.stop_button = tk.Button(self.master, text="End detection", command=self.stop_detection)
        self.start_button = tk.Button(self.master, text="Start detection", command=self.start_detection)
        self.swap_button = tk.Button(self.master, text="Start face swapping", command=lambda: self.start_detection(1))

        self.home_button = tk.Button(self.master, text="Back to main menu", command=self.go_home)

        # Detection information labels
        self.detection_info_label = tk.Label(self.master, text="To stop the video detection press 'q', "
                                                               "to pause press 'w', to resume 's'.")
        self.live_info_label = tk.Label(self.master,
                                        text="Info: don't forget to calibrate \n"
                                             "the camera via the options menu.")

        # Widgets for video detection
        self.vid_info_label = tk.Label(self.master, text="Focal width of the used camera:")
        self.vid_focal_entry = tk.Entry(self.master)
        self.vid_input_label = tk.Label(self.master, text="Video path:")
        self.vid_input_entry = tk.Entry(self.master)

    def start_calibrate(self, case: int = 0):
        """
        Create widgets for calibration and start preparations.
        """

        # Show error when trying to calibrate without active live detection.
        if (type(self.detection_obj) is not LiveDetection):
            mb.showerror("Error", "Calibration only applicable for live detection!")
        else:
            self.calib_master = tk.Toplevel(self.master)
            self.calib_master.title("Calibration")
            self.calib_master.geometry("300x200")
            if (case == 0):
                self.calib_entry_width_label = tk.Label(self.calib_master, text="Object width (cm):")
                self.calib_entry_distance_label = tk.Label(self.calib_master, text="Object distance (cm):")
                self.calib_entry_name_label = tk.Label(self.calib_master, text="Object name:")

                self.calib_entry_width = tk.Entry(self.calib_master, textvariable=self.calib_width_textvar)
                self.calib_entry_distance = tk.Entry(self.calib_master, textvariable=self.calib_dist_textvar)
                self.calib_entry_name = tk.Entry(self.calib_master, textvariable=self.calib_name_textvar)

                self.calib_entry_width_label.grid(column=0, row=1)
                self.calib_entry_distance_label.grid(column=0, row=2)
                self.calib_entry_name_label.grid(column=0, row=3)
                self.calib_entry_width.grid(column=1, row=1)
                self.calib_entry_distance.grid(column=1, row=2)
                self.calib_entry_name.grid(column=1, row=3)

                self.do_calibration = tk.Button(self.calib_master, text="Confirm calibration input",
                                                command=lambda: self.confirm_calibr(case))
                self.do_calibration.grid(column=0, row=0, columnspan=2)
            elif (case == 1):
                self.calib_entry_cols_label = tk.Label(self.calib_master, text="Rows:")
                self.calib_entry_rows_label = tk.Label(self.calib_master, text="Columns:")

                self.calib_entry_cols = tk.Entry(self.calib_master, textvariable=self.calib_cols_textvar)
                self.calib_entry_rows = tk.Entry(self.calib_master, textvariable=self.calib_rows_textvar)

                self.calib_entry_cols_label.grid(column=0, row=1)
                self.calib_entry_rows_label.grid(column=0, row=2)
                self.calib_entry_cols.grid(column=1, row=1)
                self.calib_entry_rows.grid(column=1, row=2)

                self.do_calibration = tk.Button(self.calib_master, text="Confirm calibration input",
                                                command=lambda: self.confirm_calibr(case))
                self.do_calibration.grid(column=0, row=0, columnspan=2)
            else:
                mb.showerror("Error", "Incorrect calibration invocation. Case: " + str(case))

    def confirm_calibr(self, case: int = 0):
        """
        Delete widgets for calibration and start the calibration with the given values.
        """

        self.calib_master.destroy()

        if (case == 0):
            obj_width = self.calib_width_textvar.get()
            obj_dist = self.calib_dist_textvar.get()
            obj_name = self.calib_name_textvar.get()

            self.cal_vals = [obj_width, obj_dist, obj_name]

            if (self.cal_vals[0].isdigit() and self.cal_vals[1].isdigit()):
                success = self.detection_obj.calibrate(int(self.cal_vals[0]), int(self.cal_vals[1]), self.cal_vals[2])
            else:
                mb.showerror("Type error", "Incorrect type(s) entered. Width and distance have to be integers.")
        elif (case == 1):
            cols = self.calib_cols_textvar.get()
            rows = self.calib_rows_textvar.get()
            self.calib_chess = [cols, rows]

            if (self.calib_chess[0].isdigit() and self.calib_chess[1].isdigit()):
                success = self.detection_obj.calibrate_board(cols=int(self.calib_chess[0]),
                                                             rows=int(self.calib_chess[1]),
                                                             debug=self.debug_mode)
                if (not success):
                    mb.showwarning("Calibration not successful",
                                   "The calibration was not successful. You may try again.")
            else:
                mb.showerror("Type error", "Incorrect type(s) entered. Only integers are allowed")
        else:
            mb.showerror("Error", "Incorrect calibration confirmation. Case: " + str(case))

        cv2.destroyAllWindows()

    def face_swapping(self):
        """
        Prepare and clean up GUI for face swapping
        """

        # Delete an existing (video or live) detection object
        if (self.detection_obj is not None):
            del self.detection_obj
            self.vid_info_label.grid_forget()

        self.video_detection_button.grid_forget()
        self.live_detection_button.grid_forget()
        self.face_swap_button.grid_forget()
        self.start_button.grid(column=0, row=0)
        self.swap_button.grid(column=0, row=1)
        self.home_button.grid(column=1, row=0)

        self.detection_obj = LiveFaceSwap()
        self.focal = 0

        #self.live_info_label.grid(column=0, row=1)

    def live_detection(self):
        """
        Prepare and clean up GUI for live detection.
        """

        # Delete an existing (video or live) detection object
        if (self.detection_obj is not None):
            del self.detection_obj
            self.vid_info_label.grid_forget()

        self.video_detection_button.grid_forget()
        self.live_detection_button.grid_forget()
        self.face_swap_button.grid_forget()
        self.start_button.grid(column=0, row=0)
        self.home_button.grid(column=1, row=0)

        self.detection_obj = LiveDetection()
        self.focal = 0

        self.live_info_label.grid(column=0, row=1)

    def video_detection(self):
        """
        Prepare and clean up GUI for video detection.
        """

        # Delete an existing (video or live) detection object
        if (self.detection_obj is not None):
            del self.detection_obj
            self.live_info_label.grid_forget()

        self.video_detection_button.grid_forget()
        self.live_detection_button.grid_forget()
        self.face_swap_button.grid_forget()
        self.start_button.grid(column=0, row=0)
        self.home_button.grid(column=1, row=0)

        self.detection_obj = VideoDetection()

        self.vid_info_label.grid(column=0, row=1)
        self.vid_focal_entry.grid(column=1, row=1)
        self.vid_input_label.grid(column=0, row=2)
        self.vid_input_entry.grid(column=1, row=2)

    def start_detection(self, arg: int = 0):
        """
        Start the detection.
        """

        self.start_button.grid_forget()
        self.swap_button.grid_forget()
        self.live_info_label.grid_forget()
        self.stop_button.grid(column=0, row=0)
        self.detection_info_label.grid(column=0, row=1, columnspan=2)
        if (type(self.detection_obj) is VideoDetection):
            print("Video Detection start")

            # Collect input information
            video_path = self.video_folder + self.vid_input_entry.get()
            focal_str = self.vid_focal_entry.get()
            if focal_str.isdigit():
                focal = int(focal_str)
            else:
                focal = 0

            self.vid_info_label.grid_forget()
            self.vid_focal_entry.grid_forget()
            self.vid_input_label.grid_forget()
            self.vid_input_entry.grid_forget()

            if (not (os.path.isfile(video_path))):
                mb.showerror("Error", "Video '" + video_path + "' does not exist!")
                self.stop_detection()
                self.video_detection()
            else:
                self.detection_obj.detect(video_name=video_path, focal_width=focal)

        elif (type(self.detection_obj) is LiveFaceSwap):
            print("Live face swap start")

            """
            if self.autosave:
                if not (os.path.isdir(self.video_folder)):
                    mb.showwarning("Invalid directory", "The specified directory for videos is invalid. \n"
                                                        "The video will be saved in the same directory as "
                                                        "the python file.")
                    video_name = "detection_" + date.now().strftime("%Y_%m_%d_%H_%M_%S")
                else:
                    video_name = self.video_folder + "detection_" + date.now().strftime("%Y_%m_%d_%H_%M_%S")
                self.detection_obj.detect(self.detect_select, self.no_detect_select, self.autosave, video_name)
            else:
            """
            #video_name = "detection_" + date.now().strftime("%Y_%m_%d_%H_%M_%S")
            video_name = "test123"

            if arg == 0:
                self.detection_obj.detect(autosave=self.autosave, video_title=video_name, debug=self.debug_mode)
            elif arg == 1:
                self.detection_obj.face_swap()
            else:
                return None

        else:
            print("Live detection start")

            if self.autosave:
                if not (os.path.isdir(self.video_folder)):
                    mb.showwarning("Invalid directory", "The specified directory for videos is invalid. \n"
                                                        "The video will be saved in the same directory as "
                                                        "the python file.")
                    video_name = "detection_" + date.now().strftime("%Y_%m_%d_%H_%M_%S")
                else:
                    video_name = self.video_folder + "detection_" + date.now().strftime("%Y_%m_%d_%H_%M_%S")
                self.detection_obj.detect(self.detect_select, self.no_detect_select, self.autosave, video_name)
            else:
                self.detection_obj.detect(self.detect_select, self.no_detect_select)

    def stop_detection(self):
        """
        Stop the current detection
        """

        if self.detection_obj is not None:
            del self.detection_obj
            self.detection_obj = None

        self.stop_button.grid_forget()
        self.detection_info_label.grid_forget()
        self.home_button.grid_forget()
        self.live_detection_button.grid(column=0, row=0)
        self.video_detection_button.grid(column=1, row=0)
        self.face_swap_button.grid(column=2, row=0)

    def go_home(self):
        """
        Go back to the main window
        """

        self.start_button.grid_forget()
        self.live_info_label.grid_forget()

        self.vid_info_label.grid_forget()
        self.vid_focal_entry.grid_forget()
        self.vid_input_label.grid_forget()
        self.vid_input_entry.grid_forget()

        self.stop_detection()

    def options(self):
        """
        Show options menu (WIP).
        """

        # Not yet implemented

        # Options: Autosave live detection
        # Select what to detect
        # Specify video folder

        self.options_master = tk.Toplevel(self.master)
        self.options_master.title("Options")
        self.options_master.geometry("500x200")
        self.option_autosave_label = tk.Label(self.options_master, text="Autosave live detection videostream")
        self.option_detect_select_label = tk.Label(self.options_master, text="Objects to detect:")
        self.option_no_detect_select_label = tk.Label(self.options_master, text="Objects not to detect:")
        self.option_video_folder_label = tk.Label(self.options_master, text="Standard video folder:")
        self.option_debug_label = tk.Label(self.options_master, text="Enable debug mode (for development)")

        self.option_autosave_checkbox = tk.Checkbutton(self.options_master, fg="#000000",
                                                       variable=self.options_autosave_boolvar)
        self.option_detect_select_entry = tk.Entry(self.options_master, textvariable=self.options_select_textvar)
        self.option_no_detect_select_entry = tk.Entry(self.options_master, textvariable=self.options_no_select_textvar)
        self.option_video_folder_entry = tk.Entry(self.options_master, textvariable=self.options_video_folder_textvar)
        self.option_debug_checkbox = tk.Checkbutton(self.options_master, fg="#000000",
                                                    variable=self.options_debug_boolvar)

        self.option_autosave_label.grid(column=0, row=1)
        self.option_detect_select_label.grid(column=0, row=2)
        self.option_no_detect_select_label.grid(column=0, row=3)
        self.option_video_folder_label.grid(column=0, row=4)
        self.option_debug_label.grid(column=0, row=5)

        self.option_autosave_checkbox.grid(column=1, row=1)
        self.option_detect_select_entry.grid(column=1, row=2)
        self.option_no_detect_select_entry.grid(column=1, row=3)
        self.option_video_folder_entry.grid(column=1, row=4)
        self.option_debug_checkbox.grid(column=1, row=5)

        self.apply_button = tk.Button(self.options_master, text="Apply",
                                      command=lambda: self.apply)
        self.ok_button = tk.Button(self.options_master, text="Ok",
                                   command=lambda: self.apply(True))
        self.cancel_button = tk.Button(self.options_master, text="Cacnel",
                                       command=lambda: self.options_master.destroy)

        self.apply_button.grid(column=0, row=7)
        self.ok_button.grid(column=1, row=7)
        self.cancel_button.grid(column=2, row=7)

    def apply(self, quit: bool = False):
        """ """

        self.autosave = self.options_autosave_boolvar.get()
        self.video_folder = self.options_video_folder_textvar.get()
        self.detect_select = self.options_select_textvar.get().replace(" ", "").split(",")
        self.no_detect_select = self.options_no_select_textvar.get().replace(" ", "").split(",")
        self.debug_mode = self.options_debug_boolvar.get()

        if quit:
            self.options_master.destroy()

    def help(self):
        """
        Show help (WIP).
        """

        # Short help; with link to source forge docu / readme

    def __del__(self):
        """
        Destructor; Destroy all opencv windows.
        """

        if self.detection_obj is not None:
            del self.detection_obj
        cv2.destroyAllWindows()

        try:
            self.master.destroy()
        except:
            pass


def main():
    """
    Main method: create an object DetectionGUI and start it.
    """
    root = tk.Tk()
    app = DetectionGUI(master=root)
    app.mainloop()


if __name__ == "__main__":
    main()
