
import cv2
import mediapipe as mp
import numpy as np
import csv
import time

# Variables for time tracking
official_start_time = time.time()
start_time = time.time()
end_time = 0

# Variables to track time spent in different head pose directions
time_forward_seconds = 0
time_left_seconds = 0
time_right_seconds = 0
time_up_seconds = 0
time_down_seconds = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    if not success:
        print("Failed to read frame")
        break

    startTime = time.time()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = face_mesh.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            print(f"X Rotation: {angles[0]*10000}")
            print(f"Y Rotation: {angles[1]*10000}")

            if angles[1]*10000 < -200:
                text = "Looking Left"
                time_left_seconds += time.time() - start_time
                start_time = time.time()

            elif angles[1]*10000 > 200:
                text = "Looking Right"
                time_right_seconds += time.time() - start_time
                start_time = time.time()

            elif angles[0]*10000 < -150:
                text = "Looking Down"
                time_down_seconds += time.time() - start_time
                start_time = time.time()

            elif angles[0]*10000 > 350:
                text = "Looking Up"
                time_up_seconds += time.time() - start_time
                start_time = time.time()
                
            else:
                text = "Forward"
                time_forward_seconds += time.time() - start_time
                start_time = time.time()

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
            
            cv2.line(image, p1, p2, (255, 0, 0), 2)

            cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Open the CSV file in write mode and append the angles to it
    with open('headPoses.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row if the file is empty
        if file.tell() == 0:
            writer.writerow(["X Rotation", "Y Rotation"])
        
        # Write the angles to the CSV file
        writer.writerow([angles[0]*10000, angles[1]*10000])        

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

totalTime = time.time() - startTime

print(startTime)
print(totalTime)

cap.release()
cv2.destroyAllWindows()

# After the video loop ends
end_time = time.time()
elapsed_time_minutes = (end_time - official_start_time) / 60

# After the video loop ends
print(f"Time Looking Forward: {time_forward_seconds} seconds")
print(f"Time Looking Left: {time_left_seconds} seconds")
print(f"Time Looking Right: {time_right_seconds} seconds")
print(f"Time Looking Up: {time_up_seconds} seconds")
print(f"Time Looking Down: {time_down_seconds} seconds")

total_video_duration = end_time - official_start_time

percentage_forward = (time_forward_seconds / total_video_duration) * 100
percentage_left = (time_left_seconds / total_video_duration) * 100
percentage_right = (time_right_seconds / total_video_duration) * 100
percentage_up = (time_up_seconds / total_video_duration) * 100
percentage_down = (time_down_seconds / total_video_duration) * 100

print(f"Percentage Looking Forward: {percentage_forward:.2f}%")
print(f"Percentage Looking Left: {percentage_left:.2f}%")
print(f"Percentage Looking Right: {percentage_right:.2f}%")
print(f"Percentage Looking Up: {percentage_up:.2f}%")
print(f"Percentage Looking Down: {percentage_down:.2f}%")

