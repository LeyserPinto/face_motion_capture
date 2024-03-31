from mediapipe.tasks.python import vision
from tkinter import filedialog, ttk
from datetime import datetime
from source.video_audio_handler import VideoAudioHandler
import asyncio
import os
import tkinter as tk
import cv2
import csv
import mediapipe as mp
import source.face_motion as fmotion
import source.video_audio_handler as VAHanlder

class Application:
    video_path:str
    output_path:str
    model_path       = './source/task/face_landmarker_v2_with_blendshapes.task'
    show_cv2_windows = False
    progress         = 0  
    output_file_name = 'blendfaces'
    # List of blendshape names
    blendshape_names = [
        "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft", "EyeLookUpLeft",
        "EyeSquintLeft", "EyeWideLeft", "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight",
        "EyeLookOutRight", "EyeLookUpRight", "EyeSquintRight", "EyeWideRight", "JawForward",
        "JawRight", "JawLeft", "JawOpen", "MouthClose", "MouthFunnel", "MouthPucker", "MouthRight",
        "MouthLeft", "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
        "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight",
        "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper", "MouthPressLeft",
        "MouthPressRight", "MouthLowerDownLeft", "MouthLowerDownRight", "MouthUpperUpLeft",
        "MouthUpperUpRight", "BrowDownLeft", "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft",
        "BrowOuterUpRight", "CheekPuff", "CheekSquintLeft", "CheekSquintRight", "NoseSneerLeft",
        "NoseSneerRight", "TongueOut", "HeadYaw", "HeadPitch", "HeadRoll", "LeftEyeYaw",
        "LeftEyePitch", "LeftEyeRoll", "RightEyeYaw", "RightEyePitch", "RightEyeRoll"
    ]
    
    def __init__(self, master: None):
        self._made_widget_1(master)
        self._made_widget_2(master)
        self._made_widget_3(master)
        self._made_widget_4(master)
            
    def get_file_path(self):
        file_path = filedialog.askopenfilename(initialdir='shell:My Video', 
                                               title="Select Video File", 
                                               filetypes=[("Formato MPEG-4 (MP4)","*.mp4"),
                                                          ("Formato AVI","*.avi"),
                                                          ("Formato MKV","*.mkv"),
                                                          ("Formato MOV","*.mov"),
                                                          ("formato DIVX","*.divx"),])
        if file_path:
            self.video_path   = file_path
            self.w3_button['state'] = tk.NORMAL
            self.w2_label.config(text= file_path)
        else:
            self.w2_label.config(text= "No file selected.")
            self.w3_button['state'] = tk.DISABLED
    
    def set_output_path(self):
        out_path = filedialog.askdirectory(title="Save Location", mustexist=True)
        if out_path:
            self.output_path = out_path     
            self.set_states_buttons()
            asyncio.run(self.process_generation())
   
    def update_current_val(self, progress_val:int):
        self.progresslabel.config(text=f"{progress_val}%")
        self.widget4.master.update()

    def set_states_buttons(self, tk_state = tk.DISABLED):
        self.w1_button.config(state=tk_state)
        self.w3_button.config(state=tk_state)
        
    def _made_widget_1(self, master):
         # Widget - 1
        self.widget1          = tk.Frame(master)
        self.widget1.config(bg='#222831', width=10, pady=20)
        self.widget1.pack(side=tk.TOP)
        # Components
        self.msg = tk.Label(self.widget1, )
        self.msg.config(text="Select Video File:", bg='#222831', fg='#FFFFFF', font=("Calibri", "12"))
        self.msg.pack(side=tk.LEFT)
        
        self.w1_button = tk.Button(self.widget1)
        self.w1_button["text"] = "..."
        self.w1_button["font"] = ("Calibri", "12")
        self.w1_button["width"] = 3
        self.w1_button["height"] = 1
        self.w1_button["command"] = self.get_file_path
        self.w1_button.pack(side=tk.LEFT)
        
    def _made_widget_2(self, master):
        # Widget - 2
        self.widget2          = tk.Frame(master)
        self.widget2.config(bg='#222831', width=10, pady=20)
        self.widget2.pack(side=tk.TOP)
        # Components
        self.w2_label = tk.Label(self.widget2)
        self.w2_label.configure(width= 40, bg='#FFFFFF', fg='#222831')
        self.w2_label.pack(side= tk.BOTTOM) 
        
    def _made_widget_3(self, master):
        # Widget - 3        
        self.widget3          = tk.Frame(master)
        self.widget3.config(bg='#222831', width= 5, pady= 5)
        self.widget3.pack(side=tk.BOTTOM)
                
        self.w3_button = tk.Button(self.widget3, state=tk.DISABLED)
        self.w3_button.config(text='Generate', font= ("Calibri", "12", "bold"), width=50, height= 1, command= self.set_output_path)
        self.w3_button.pack(side=tk.LEFT)
        
    def _made_widget_4(self, master):
        # Widget - 4
        self.widget4          = tk.Frame(master)
        self.widget4.config(bg= '#222831', width= 5, pady=5)
        self.widget4.pack(side=tk.BOTTOM)
        
        self.pbmsg = tk.Label(self.widget4, bg='#222831', fg='#FFFFFF', text='Current Task Process:', font=("Calibri", "12"))
        self.pbmsg.pack(side=tk.TOP)
        
        self.progresslabel = tk.Label(self.widget4, bg='#222831', fg='#FFFFFF', text='0%', font=("Calibri", "16"))
        self.progresslabel.pack(side=tk.TOP)
        
        self.progressAction = tk.Label(self.widget4, bg='#222831', fg='yellow', text='Task in progress:', font=("Calibri", "8"))
        self.progressAction.pack(side=tk.TOP)
            
    async def process_generation(self):    
        self.update_current_val(0)
        task_to_do = ['blendshapes', 'audio', 'end_process']        
        
        file_original_name, extension = os.path.splitext(os.path.basename(self.video_path))
        
        if len(file_original_name) > 0:
            self.output_file_name = f"output_{file_original_name}_{str(datetime.now().timestamp()).replace('.', '_')}"
        try:
            for task in task_to_do:
                task_result = False
                if task == 'blendshapes':
                    self.progressAction.config(text='Task in progress: Blendshapes Generation (CSV File)')
                    # Blendshapes Generations Process
                    blendshapes_task = asyncio.create_task(self.do_landmark_generation())        
                    task_result      = await blendshapes_task
                elif task == 'audio':
                    # After blendshapes task is finished then 
                    #Do audio extractions process
                    self.progressAction.config(text='Task in progress: Audio Generation (Wav File)')
                    audio_task  = asyncio.create_task(self.do_audio_extraction())
                    task_result =  await audio_task
                else:
                    # End process operations
                    self.set_states_buttons(tk.NORMAL)
                    self.progressAction.config(text=f"Generation made with sucess, \n file on directory: {self.output_path}. \n output file name is: {self.output_file_name}")
                    task_result = {'task_status' : 'ended'}
                # After blendshapes task is finished then 
                if task_result.get('task_status') == 'Sucess':        
                    continue
                elif task_result.get('task_status') == 'ended':
                    break
                
        except Exception as error:
            self.progressAction.config(text='Generation process not concluded with success,\n an error happening on task process')
            self.progressAction.config(fg='red')
            print(f'Error: {type(error).__name__} - {error}')
        
    async def do_landmark_generation(self):
        o_cap = cv2.VideoCapture(self.video_path)
        FaceMotion = fmotion.FaceMotion()
        output_location = f"{self.output_path}/{self.output_file_name}_blendshapes.csv"
        # Check if the video file was successfully opened
        if not o_cap.isOpened():
            print("Error: opening video file")
            return False;
        else:
            # Create a CSV file and write the header
            header = ["Timecode", "BlendShapeCount"] + self.blendshape_names
            
            
            with open(output_location, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)
                frame_count = int(o_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            
                while o_cap.isOpened():                    
                    ret, frame = o_cap.read()                       
                    if not ret: break
                    
                    current_process = int((o_cap.get(cv2.CAP_PROP_POS_FRAMES) / frame_count) * 50)
                    self.update_current_val(current_process)
                                        
                    FaceMotion.set_video_capture(o_cap)
                    FaceMotion.do_process_mediapipe_facelandmark(frame, 
                                                                self.model_path, 
                                                                vision.RunningMode.VIDEO,
                                                                writer = writer,
                                                                use_imshow=self.show_cv2_windows)                                           
                    
                    if cv2.waitKey(1) == ord('q'): 
                        break;                
                
        o_cap.release()
        cv2.destroyAllWindows()
        await asyncio.sleep(0.25)
        return {'task_status' : 'Sucess'}
        
    async def do_audio_extraction(self):
        self.update_current_val(55)
        output_file_name = f"{self.output_path}/{self.output_file_name}_audio.wav"
        VideoAudioHandler.extract_audio_from_video(video_file_path = self.video_path, output_location= output_file_name)
        await asyncio.sleep(1)
        self.update_current_val(100)
        return {'task_status' : 'Sucess'}
     