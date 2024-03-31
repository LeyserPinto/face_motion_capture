from mediapipe.tasks             import python
from mediapipe.tasks.python      import vision
from mediapipe                   import solutions
from mediapipe.framework.formats import landmark_pb2
from source.face_geometry        import (PCF, get_metric_landmarks)
import cv2
import numpy as np
import mediapipe as mp
import transforms3d

class FaceMotion():
    video_capture: cv2.VideoCapture
                      
    def do_process_mediapipe_facelandmark(self, video_frame, model_path, task_running_mode, writer, use_imshow = False):
        """
            Methods get the base process to get mediapipe Face LandMarker
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=video_frame)
        detector = self.get_landmark_detector(model_path, task_running_mode)
        frame_timestamp, time_formatted = self.get_current_frame_timestamp(self.get_video_capture())
        landmarker_results = detector.detect_for_video(mp_image, frame_timestamp)
        
        blendshape_data = self.process_landmarks_data_generation(time_formatted, landmarker_results, mp_image)
        
        if blendshape_data is not None:
            # Write the data row to the CSV file
            writer.writerow(blendshape_data)

        if use_imshow:                    
            annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), landmarker_results)
            self.showing_cv2_image(annotated_image, True)
        
    def get_landmark_detector(self, s_model_path: str, x_running_mode: any, i_num_faces = 1, b_blendshapes = True, b_transformation_matrix = True):
        """
            get the landmark detector base to execute order of detect image, video or live stream
            Returns:
            -------
            `mediapipe.task.python.vision.face_landmarker.FaceLandMarker` object that's created from `options`.
        """
        base_options = python.BaseOptions(model_asset_path = s_model_path)
        options      = vision.FaceLandmarkerOptions(base_options            = base_options,
                                                    output_face_blendshapes = b_blendshapes,
                                                    output_facial_transformation_matrixes = b_transformation_matrix,
                                                    running_mode            = x_running_mode,
                                                    num_faces               = i_num_faces)
        
        return vision.FaceLandmarker.create_from_options(options)       
    
    def get_current_frame_timestamp(self, video_capture: cv2.VideoCapture):
        # Get the frame rate of the video
        frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
        
        # Get the current frame index
        frame_index = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        
        # Calculate the timestamp for the current frame
        # Ex: frame_index = 1, frame_rate = 30.0 then (1 * (100 / 30)) = 33.33333333333333
        frame_timestamp_ms = int(frame_index * (1000 / frame_rate)) 
        # get Miliseconds from module between current frame_index and frame_rate
        milliseconds = int((frame_index) % frame_rate)   
        # get Seconds from frame_timestamp_ms operation 
        seconds = int((frame_timestamp_ms / 1000) % 60)
        # get Minutes from frame_timestamp_ms operation        
        minutes = int((frame_timestamp_ms / (1000 * 60)) % 60)
        # get Hours from frame_timestamp_ms operation 
        hours = int(frame_timestamp_ms / (1000 * 60 * 60))
                
        frame_index_formatted = int(frame_index % 1000)
        
        time_formatted = f"{(hours):02d}:{minutes:02d}:{seconds:02d}:{milliseconds:02}.{frame_index_formatted:03d}"
        
        return frame_timestamp_ms, time_formatted
    
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            # Draw landmarks circles
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec   = mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 106), thickness= 1),
                connection_drawing_spec =mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

        return annotated_image
    
    def process_landmarks_data_generation(self, time_formatted, landmark_result, mp_image):        
        #Perform the data generation of every landmark blendshape to write on CSV file to export
        if len(landmark_result.face_blendshapes) > 0:
            blendshapes = landmark_result.face_blendshapes[0]
        else:
            # Skip the frame and continue to the next iteration
            return
        
        all_blendshape_scores = []
        
        # Get eye Iris left and Right position x and y from landmarker results 
        left_iris_x  = landmark_result.face_landmarks[0][468].x
        left_iris_y  = landmark_result.face_landmarks[0][468].y
        right_iris_x = landmark_result.face_landmarks[0][473].x
        right_iris_y = landmark_result.face_landmarks[0][473].y
        
        # Get Blendshapes locations throught iteration for
        # Obs. Iteration will be execute starting on 1 index
        for blendshapes_category in blendshapes[1:]:
            blendshape_name = blendshapes_category.category_name
            blendshape_score = blendshapes_category.score
            formatted_score = "{:.8f}".format(blendshape_score)
            # print(blendshape_name + ":" + formatted_score)

            # Add the formatted score to the list
            all_blendshape_scores.append(formatted_score)
            
        #The order of the indexes in this list needs to be remade
        new_order = [8, 10, 12, 14, 16, 18, 20, 9, 11, 13, 15, 17, 19, 21, 22, 25, 23, 24, 26, 31, 37, 38, 32, 43, 44, 29, 30, 27, 28, 45, 46, 39, 40, 41, 42, 35, 36, 33, 34, 47, 48, 50, 1, 2, 3, 4, 5, 6, 7, 49, 50]
        all_blendshape_scores_sorted =[all_blendshape_scores[i] for i in new_order]
        
        #Num of blendshapes in this frame
        num_blendshapes = len(blendshapes[1:])
        
        # Tongue
        tongue = [0]
        transform_mat, metrics_landmarks, rotation_vector, transform_vector = self.get_frame_rotation(landmark_result, mp_image)
        headrotation =  self.get_head_rotation(transform_mat)
        eyes         = self.get_eyes_landmarks(left_iris=[left_iris_x, left_iris_y], right_iris=[right_iris_x, right_iris_y])
        
        shape_data = [time_formatted] +  [num_blendshapes] + all_blendshape_scores_sorted + tongue + headrotation + eyes 
        return shape_data
        
    def get_frame_rotation(self, Face_Landmarks, mp_image):
        # Frame Image sizes
        image_width = mp_image.width
        image_height = mp_image.height
        # points of the face model that will be used for SolvePnP later
        points_idx = [1, 33, 263, 61, 291, 199]
        #points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
        points_idx = list(set(points_idx))
        points_idx.sort()
        
        # Camera deep and pseudal informations
        focal_length  = image_width
        center        = (image_width / 2, image_height / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

        pcf = PCF(frame_height=image_height, frame_width=image_width, fy=camera_matrix[1, 1])
        
        dist_coeff = np.zeros((4, 1))
        
        landmarks = np.array([(lm.x, lm.y, lm.z) for lm in Face_Landmarks.face_landmarks[0][:468]])
        landmarks = landmarks.T
        
        metric_landmarks, pose_transform_mat = get_metric_landmarks(landmarks.copy(), pcf)
        
        model_points = metric_landmarks[0:3, points_idx].T
        image_points = ( landmarks[0:2, points_idx].T * np.array([image_width, image_height])[None, :])
        
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeff, flags=cv2.SOLVEPNP_ITERATIVE)
        
        return pose_transform_mat, metric_landmarks, rotation_vector, translation_vector
     
    def get_head_rotation(self, pose_transform_matrix):
        # calculate the head rotation out of the pose matrix
        
        eulerAngles = transforms3d.euler.mat2euler(pose_transform_matrix)
        pitch = -eulerAngles[0]
        yaw   = eulerAngles[1]
        roll  = eulerAngles[2]
        
        return [pitch, yaw, roll]
    
    def get_eyes_landmarks(self, left_iris: list, right_iris: list):
        return [left_iris[0], left_iris[1], 0, right_iris[0], right_iris[1], 0]
    
    def showing_cv2_image(self, img, resize_img = False):
        if resize_img:
            img_preview = cv2.resize(img, (1024, 728))
            cv2.imshow('Face Capture Preview', img_preview)
        else:    
            cv2.imshow('Face Capture Preview', img)
    
    def set_video_capture(self, video_capture: cv2.VideoCapture):
        self.video_capture = video_capture
        
    def get_video_capture(self):
        """
            Returns:
            `cv2.VideoCapture`
        """
        return self.video_capture
    