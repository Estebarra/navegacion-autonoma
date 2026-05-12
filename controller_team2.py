# Simple controller with onboard camera
from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver

# Requierement libraries
import numpy as np
import cv2

# Python standard libreries
from datetime import datetime
import os
import time

# Configuration constants
DEBOUNCE_TIME = 0.1 #100 milliseconds
MAX_ANGLE = 0.5
MAX_SPEED = 50.5
SPEED_INCR = 5
ANGLE_INCR = 0.017

# Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

# Image processing example
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

def canny(gray_img):
    canny_img = cv2.Canny(gray_img, threshold1=100, threshold2=200)
    return canny_img

def def_roi(canny_img):
    vertices = np.array([[(0,127),(70, 65), (210, 65), (230, 127)]], dtype = np.int32)
    # imagen en negro de las mismas dimensiones que la imagen original en escala de grises
    roi_img = np.zeros_like(canny_img)
    # la ROI se pinta de blanco sobre la imagen de negro
    cv2.fillPoly(roi_img, vertices, 255)
    canny_roi = cv2.bitwise_and(canny_img, roi_img)
    return canny_roi    

# parámetros usados para la transformada de Hough
rho = 1             # resolución de rho en pixeles
theta = np.pi/180   # resolución de theta en radianes
threshold = 25      # mínimo número de votos para ser considerado una línea
min_line_len = 10   # mínimo número de pixeles para que se forme una línea
max_line_gap = 25   # máximo espacio en pixeles entre segmentos de línea

def hough(canny_roi):
        lines = cv2.HoughLinesP(canny_roi, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        
        # Se crea un fondo negro del tamaño de la imagen con bordes
        img_lines = np.zeros_like(canny_roi)
        
        filtered_lines = []

        # Si Hough no detecta líneas, se regresa una lista vacía
        if lines is None:
            return canny_roi, img_lines, filtered_lines
        
        # Se dibuja cada línea detectada por Hough, descartando líneas mayormente horizontales
        for line in lines:
            for x1, y1, x2, y2 in line:
                dx = x2 - x1
                dy = y2 - y1

                # Ángulo de la línea detectada
                line_angle = abs(np.degrees(np.arctan2(dy, dx)))

                # Se descartan líneas casi horizontales para no alterar el PID
                if line_angle < 5 or line_angle > 175:
                    continue

                filtered_lines.append((x1, y1, x2, y2))
                cv2.line(img_lines, (x1, y1), (x2, y2), 255, 2) 

        img_lane_lines = cv2.addWeighted(canny_roi, 1, img_lines, 1, 0)
        return img_lane_lines, img_lines, filtered_lines
        
# Display image on onboard display
def display_image(display, image, title=""):
    # Image to display
    image_rgb = np.dstack((image, image,image,))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)
    display.imageDelete(image_ref)
    
    if title != "":
        display.setColor(0x000000)
        display.fillRectangle(0, 0, display.getWidth(), 18)

        display.setColor(0xFFFFFF)
        display.setFont("Arial", 12, True)
        display.drawText(title, 5, 2)

# Definición del PID controller
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def compute(self, process_variable, dt):
        if dt <= 0:
            return 0

        error = self.setpoint - process_variable

        P_out = self.Kp * error

        self.integral += error * dt
        I_out = self.Ki * self.integral

        derivative = (error - self.previous_error) / dt
        D_out = self.Kd * derivative

        output = P_out + I_out + D_out

        self.previous_error = error

        return output
    
def get_lane_error(img_lines):
    height, width = img_lines.shape[:2]

    ys, xs = np.nonzero(img_lines)

    if len(xs) == 0:
        return None
    
    lower_half = ys > int(height * 0.5)

    if np.any(lower_half):
        xs = xs[lower_half]

    lane_center = np.mean(xs)
    image_center = width / 2

    # Error normalizado respecto al centro de la imagen.
    # Si el centro detectado coincide con el centro de la imagen, error = 0.
    error = (lane_center - image_center) / image_center

    return error

# Main function
def main():
    speed = 50
    angle = 0.0
    last_press = {}

    # Create the Robot instance.
    robot = Car()
    driver = Driver()
    display_speed = Display("display_speed")

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())
    dt = timestep /1000

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    # PID controller del steering
    image_width = camera.getWidth()
    image_center = image_width / 2

    pid_steering = PIDController(Kp=1, Ki=0, Kd=0, setpoint=0)
    last_debug_time = 0

    # Processing display
    display_img = Display("display_image")
    display_canny = Display("display_canny")
    display_lines = Display("display_lines")
 
    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)

        # Process and display image canny
        gray_img = greyscale_cv2(image)
        canny_img = canny(gray_img)
        display_image(display_canny, canny_img, "Canny")

        # Define ROI
        canny_roi = def_roi(canny_img)
        
        #Se aplica la transformada de Hough
        img_lane_lines, img_lines, filtered_lines = hough(canny_roi)
        
        #Se muestra imagen final con lineas detectadas
        display_image(display_lines, img_lines, "Lines Hough")

        display_image(display_img, canny_roi, "canny roi")
        
        # PID controller - Steering
        lane_error = get_lane_error(img_lines)

        if lane_error is not None:
            angle = pid_steering.compute(lane_error, dt)

            if angle > MAX_ANGLE:
                 angle = MAX_ANGLE
            elif angle < -MAX_ANGLE:
                angle = -MAX_ANGLE
        else:
            angle = angle

        current_debug_time = time.time()

        if current_debug_time - last_debug_time > 0.5:
            print(
               "PID ON | "
                f"hough_pixels={cv2.countNonZero(img_lines)} | "
                f"lane_error={lane_error} | "
                f"angle={angle:.3f}"
            )
            last_debug_time = current_debug_time

        #Obtain current speed
        current_speed = abs(driver.getCurrentSpeed())
        display_speed.setColor(0x000000)
        display_speed.fillRectangle(0, 0, display_speed.getWidth(), display_speed.getHeight())
        display_speed.setColor(0xFFFFFF)
        display_speed.setFont("Arial", 14, True)
        display_speed.drawText("Vehicle Speed", 10, 8)
        display_speed.setColor(0x00FF00)
        display_speed.setFont("Arial", 24, True)
        display_speed.drawText(f"{current_speed:.1f} kph", 10, 35)

        # To reduce rebounds
        current_time = time.time()

        # Read keyboard
        key=keyboard.getKey()

        if key in last_press and (current_time - last_press[key] < DEBOUNCE_TIME):
            continue # Ignore rebound

        # Pressed key accepted, update
        last_press[key] = current_time

        if key == keyboard.UP: #up
            if speed < MAX_SPEED:
                speed += SPEED_INCR
                print("up")
        elif key == keyboard.DOWN: #down
            if speed >= SPEED_INCR:
                speed -= SPEED_INCR
                print("down")
        elif key == keyboard.RIGHT: #right
            # Change_steer_angle(+1)
            angle += ANGLE_INCR
            if angle > MAX_ANGLE:
                angle = MAX_ANGLE
            print("right")
        elif key == keyboard.LEFT: #left
            # Change_steer_angle(-1)
            angle -= ANGLE_INCR
            if angle < -MAX_ANGLE:
                angle = -MAX_ANGLE
            print("left")
        elif key == ord('A'):
            # Filename with timestamp and saved in current directory
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage(os.getcwd() + "/" + file_name, 1)
            
        # Update angle and speed
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

if __name__ == "__main__":
    main()