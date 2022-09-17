from inspect import classify_class_attrs
import numpy
import cv2
import customtkinter
import pickle
from PIL import ImageTk, Image
import pandas
import mediapipe
import tkinter
from sample_landmarks import landmarks

app = tkinter.Tk()
customtkinter.set_appearance_mode("dark")
app.title("Computer Vision Workout App")
app.geometry("480x700")

def reset():
    global reps
    reps = 0

#obtains the picture of the motion done on the camera and analyzes it via pose estimation
def detect_motions():
    global prob
    global current_motion_point
    global reps
    global class_current

    r, f = video_capture.read()

    captured_image = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

    process_image = pose_estimation.process(captured_image)

    draw_lines.draw_landmarks(captured_image, process_image.pose_landmarks, get_estimation.POSE_CONNECTIONS, draw_lines.DrawingSpec(circle_radius = 2, color = (0,0,255), thickness = 2), draw_lines.DrawingSpec(circle_radius = 2, color = (0,255,0), thickness = 2))
    
    try:
        landmark_row = numpy.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in process_image.pose_landmarks.landmark]).flatten().tolist()
        landmark_row_col = pandas.DataFrame([landmark_row], columns = landmarks)
        prob = train_model.predict_proba(landmark_row_col)[0]
        class_current = train_model.predict(landmark_row_col)[0]
        if(prob[prob.argmax()] > 0.7 and class_current == "up" and current_motion_point == "down"):
            current_motion_point = "up"
            reps = reps + 1
        elif(prob[prob.argmax()] > 0.7 and class_current == "down"):
            current_motion_point = "down"
    except Exception as error:
        pass

    slice_img = captured_image[:, :460, :] #slice the image
    array_img = Image.fromarray(slice_img) #convert it into an array
    tk_img = ImageTk.PhotoImage(array_img) #convert from array to tk image
    window_l.tk_img = tk_img
    window_l.configure(image = tk_img)
    window_l.after(10, detect_motions)
    probability_b.configure(text = prob[prob.argmax()])
    class_b.configure(text = current_motion_point)
    counter_b.configure(text = reps)

probability_l = customtkinter.CTkLabel(app, text_font = ("Arial", 18), text_color = "black", padx = 10, width = 120, height = 40)
probability_l.configure(text = "Odds")
probability_l.place(x = 300, y = 1)
probability_b = customtkinter.CTkLabel(app, text_font = ("Arial", 18), text_color = "black", fg_color = "red", width = 120, height = 40)
#probability_b.configure(text = "0")
probability_b.place(x = 300, y = 41)

class_l = customtkinter.CTkLabel(app, text_font = ("Arial", 18), text_color = "black", padx = 10, width = 120, height = 40)
class_l.configure(text = "Motion")
class_l.place(x = 10, y = 1)
class_b = customtkinter.CTkLabel(app, text_font = ("Arial", 18), text_color = "black", fg_color = "red", width = 120, height = 40)
class_b.configure(text = "0")
class_b.place(x = 10, y = 41)

counter_l = customtkinter.CTkLabel(app, text_font = ("Arial", 18), text_color = "black", padx = 10, width = 120, height = 40) 
counter_l.configure(text = "Reps")
counter_l.place(x = 160, y = 1)
counter_b = customtkinter.CTkLabel(app, text_font = ("Arial", 18), text_color = "black", fg_color = "red", width = 120, height = 40)
counter_b.configure(text = "0")
counter_b.place(x = 160, y = 41)

reset_button = customtkinter.CTkButton(app, text_font = ("Arial", 18), text_color = "black", command = reset, fg_color = "red", text = "Reset Reps", width = 120, height = 40)
reset_button.place(x = 10, y = 600)

window = tkinter.Frame(width = 480, height = 480)
window.place(x = 10, y = 90)
window_l = tkinter.Label(window)
window_l.place(x = 0, y = 0)

get_estimation = mediapipe.solutions.pose
pose_estimation = get_estimation.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

draw_lines = mediapipe.solutions.drawing_utils

with open('deadlift_training_data.pkl', 'rb') as data:
    train_model = pickle.load(data)

prob = numpy.array([0,0])
current_motion_point = ''
reps = 0
class_current = ''
video_capture = cv2.VideoCapture(0)
detect_motions()

app.mainloop()

