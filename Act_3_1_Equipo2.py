# ==============================================================================
# Actividad 3.1 - Detección de Peatones con SVM
# Equipo 2
# ==============================================================================

# Recursos de Webots
from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver

# Librerías
import numpy as np
import cv2
from skimage.feature import hog
import joblib

# ==============================================================================
# CONSTANTES INICIALES
# ==============================================================================
MAX_ANGLE = 0.5     # Rad
CRUISING_SPEED = 50 # km/hr

# CONSTANTES DE CONTROL PID
KP = 0.01
KI = 0.0
KD = 0.002

# CONSTANTES DE RECONOCIMIENTO DE PEATONES
SVM_MODEL_PATH = "svm_pedestrian_detector.pkl"

OBSTACLE_DISTANCE_LIMIT = 20.0  # m, distancia máxima para considerar obstáculo
STOP_DISTANCE = 10.0            # m, distancia para detenerse si hay peatón

MIN_PEDESTRIAN_SCORE = 1.5      # Umbral de detección de peatones

CAUTION_SPEED = 20.0            # km/hr, velocidad reducida
CAUTION_BRAKE = 0.3             # frenado moderado
EMERGENCY_BRAKE = 1.0           # frenado máximo

# ==============================================================================
# FUNCIONES DE VISIÓN CON DETECCIÓN DE BORDES
# ==============================================================================

# Extracción de imágen de la cámara (BGRA = 64 x 128 px, 4 canales)
def get_image(camera):
    raw_image = camera.getImage()

    # Obtenemos la imágen BRG de ña cámara de Webots
    image = np.frombuffer(raw_image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))

    return image

# Obtención de impagenes: carriles aislados, escala de grises, Canny y Canny con ROI
def get_images_CV2(img_bgr, detect_white_lines=False):
    
    # Se guardan las dimensiones de la imagen
    img_height, img_width = img_bgr.shape[:2]

    # Aplicamos Hue-Saturation-Value para detectar mejor las líneas amarillas
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # ========== 1. Máscara para amarillo, saturación (100 -> 50) y brillo (100 -> 50) ==========
    lower_yellow = np.array([15, 50, 50]) 
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    # Por defecto, usamos solo amarillo
    mask_lane = mask_yellow

    # ========== 2. Máscara para blanco ==========
    if detect_white_lines:
        img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)

        # HLS: H = Hue, L = Lightness, S = Saturation
        # Blanco: alta luminosidad, saturación baja o media
        lower_white = np.array([0, 140, 0])
        upper_white = np.array([180, 255, 120])

        mask_white_hls = cv2.inRange(img_hls, lower_white, upper_white)

        # También se refuerza con escala de grises
        img_gray_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask_white_gray = cv2.threshold(img_gray_original, 140, 255, cv2.THRESH_BINARY)

        # Combinamos ambas máscaras de blanco
        mask_white = cv2.bitwise_or(mask_white_hls, mask_white_gray)

        # Engrosar y conectar líneas blancas detectadas
        kernel_white = np.ones((3, 3), np.uint8)
        mask_white = cv2.dilate(mask_white, kernel_white, iterations=2)
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel_white, iterations=2)

        # Combinar amarillo y blanco
        mask_lane = cv2.bitwise_or(mask_yellow, mask_white)

    # ========== 3. Aislar carriles ==========
    img_HSVLane = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_lane)

    # ========== 4. Canny ==========
    img_gray = cv2.cvtColor(img_HSVLane, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0, 0)
    img_canny = cv2.Canny(img_blur, 40, 120)

    # ========== 5. Región de interés (ROI) ==========
    vertices = np.array([[(img_width*0, img_height),
                          (img_width*0.2, img_height*0.65),
                          (img_width*0.8, img_height*0.65),
                          (img_width*1, img_height)
                          ]],dtype=np.int32)
    img_roi = np.zeros_like(img_gray)       # ROI, máscara de ceros
    cv2.fillPoly(img_roi, vertices, 255)
    img_cannyROI = cv2.bitwise_and(img_canny, img_roi)

    return img_HSVLane, img_gray, img_canny, img_cannyROI

# Función para mostrar imágen en display de Webots
def display_image(display, image, title=None):

    # Si la imagen está en escala de grises: (height, width)
    if len(image.shape) == 2:
        image_to_display = np.dstack((image, image, image))

    # Si la imagen ya tiene 3 canales: (height, width, 3)
    elif image.shape[2] == 3:
        image_to_display = image

    # Si la imagen tiene 4 canales: (height, width, 4)
    elif image.shape[2] == 4:
        image_to_display = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    # Crear una imágen compatible con Webots
    image_ref = display.imageNew(image_to_display.tobytes(),
                                 Display.RGB,
                                 width=image_to_display.shape[1],
                                 height=image_to_display.shape[0])
    
    # Pegado de imágen en display
    display.imagePaste(image_ref, 0, 0, False)

    # Se añade título en texto sobre la imágen
    if title is not None:
        display.setColor(0xFFFFFF)      # White
        display.drawText(title, 5, 5)   # Position

# Función para mostrar el estátus en display de Webots
def display_status(display, speed, angle, brake_intensity, hazard_status):
    display.setColor(0x000000)
    display.fillRectangle(0, 0, display.getWidth(), display.getHeight())

    display.setColor(0xFFFFFF)
    display.drawText(f"Speed: {speed:.1f} km/h", 5, 5)
    display.drawText(f"Angle: {angle:.2f} rad", 5, 18)
    display.drawText(f"Breake intensity: {brake_intensity:.2f}", 5, 31)
    display.drawText(f"Hazard lights: {hazard_status}", 5, 44)

# Función que aplica la transformada de Hough a las imágenes
def get_HoughLines(img_mask, image):
    rho = 1                 # resolución de rho en píxeles
    theta = np.pi / 180     # resolución de theta en radianes
    threshold = 5           # mínimo número de votos para ser considerada una línea
    min_line_len = 5        # mínimo número de píxeles para que se forme una línea
    max_line_gap = 20       # máximo espacio en píxeles entre segmentos de línea

    lines = cv2.HoughLinesP(img_mask,
                            rho,
                            theta,
                            threshold,
                            np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    # Se crea un fondo negro del tamaño de la imagen RGB con bordes
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_lines = np.zeros_like(img_rgb)

    # Se dibujan cada una de las líneas sobre la imagen con fondo negro
    if lines is not None:
        # print(f"Líneas detectadas: {len(lines)}")
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_lines, (x1, y1), (x2, y2),(255, 0, 0), 1)

    # Se combinan la imagen original y las líneas encontradas
    img_HoughLines = cv2.addWeighted(img_rgb, 1, img_lines, 1, 1)
    
    return img_HoughLines, lines

# ==============================================================================
# FUNCIONES DE RECONOCIMIENTO DE PEATONES
# ==============================================================================

# Función que obtiene los histogramas de gradientes orientados de la imágen
def get_HOGFeatures(img_window):

    # El modelo SVM espera imágenes con tamaño 64x128
    img_resized = cv2.resize(img_window, (64, 128))

    # Se convierte a escala de grises
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Se extraen las características HOG igual que como se entrenó el modelo SVM
    features = hog(img_gray,
                   orientations=11,
                   pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2),
                   transform_sqrt=False,
                   visualize=False,
                   feature_vector=True)

    # Se regresa como matriz de una fila para usar con svm_model.predict()
    return features.reshape(1, -1)

# Función para determinar el perímetro en la imagen peatones usando SVM
def detect_pedestrian(image, svm_model):
    img_height, img_width = image.shape[:2]

    pedestrian_boxes = []
    pedestrian_scores = []

    # Ventanas verticales con proporción cercana a 1:2
    window_sizes = [(32, 64), (40, 80)]

    # Píxeles que se moverá la ventana deslizante con cada aparición
    step_size = 16

    # Evitar buscar demasiado arriba si hay cielo o edificios
    y_start = int(img_height * 0.10)
    y_end = img_height

    for win_w, win_h in window_sizes:
        if win_w > img_width or win_h > img_height:
            continue

        for y in range(y_start, y_end - win_h + 1, step_size):

            # Si la ventana está más arriba, limitar búsqueda al centro
            if y < int(img_height * 0.35):
                x_start = int(img_width * 0.25)
                x_end = int(img_width * 0.75)

            # Si la ventana está más abajo, permitir búsqueda más amplia
            else:
                x_start = int(img_width * 0.10)
                x_end = int(img_width * 0.90)

            for x in range(x_start, x_end - win_w + 1, step_size):
                img_window = image[y:y + win_h, x:x + win_w]

                features = get_HOGFeatures(img_window)
                prediction = svm_model.predict(features)

                if prediction[0] == 1:
                    try:
                        score = svm_model.decision_function(features)[0]
                    except Exception:
                        score = 1.0

                    if score > MIN_PEDESTRIAN_SCORE:
                        pedestrian_boxes.append([x, y, win_w, win_h])
                        pedestrian_scores.append(float(score))

    # Si no hubo detecciones
    if len(pedestrian_boxes) == 0:
        return False, []

    # Non-Maximum Suppression para eliminar cajas repetidas sobre el mismo peatón
    indices = cv2.dnn.NMSBoxes(
        pedestrian_boxes,
        pedestrian_scores,
        score_threshold=MIN_PEDESTRIAN_SCORE,
        nms_threshold=0.3
    )

    final_boxes = []

    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(pedestrian_boxes[i])

    pedestrian_detected = len(final_boxes) > 0

    return pedestrian_detected, final_boxes

# Función que detecta peatones en frente del vehículo para diferenciar de obstáculos/barriles
def get_close_pedestrian_boxes(pedestrian_boxes, image):
    img_height, img_width = image.shape[:2]

    close_pedestrian_boxes = []

    for box in pedestrian_boxes:
        x, y, w, h = box

        box_center_x = x + w / 2.0
        box_bottom_y = y + h

        # Zona central aproximada del frente del vehículo
        in_front_area = (box_center_x > img_width * 0.25 and box_center_x < img_width * 0.75)

        # Evitar que peatones muy arriba/lejanos cuenten como causa del frenado
        is_low_enough = box_bottom_y > img_height * 0.55

        if in_front_area and is_low_enough:
            close_pedestrian_boxes.append(box)

    return close_pedestrian_boxes

# Función para procesar datos del LiDAR Sick LMS 291
def process_sick_data(sick_data, sick_width):
    HALF_AREA = 20
    collision_count = 0
    obstacle_dist = 0.0

    center = int(sick_width / 2)
    start_idx = max(0, center - HALF_AREA)
    end_idx = min(sick_width, center + HALF_AREA)

    for x in range(start_idx, end_idx):
        range_val = sick_data[x]

        if not np.isinf(range_val) and range_val < 20.0:
            collision_count += 1
            obstacle_dist += range_val

    if collision_count == 0:
        return 999.0

    obstacle_dist = obstacle_dist / collision_count

    return obstacle_dist

# Función para generar la imagen con detección de peatones
def get_pedestrianDisplayImage(image, pedestrian_detected, pedestrian_boxes, obstacle_dist, event_type):

    # Se crea una copia para no modificar la imagen original
    img_pedestrian = image.copy()
    img_height, img_width = img_pedestrian.shape[:2]

    # Colores en BGR
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)

    # Si no hay obstáculo cercano, no mostrar distancia artificial de 999 m
    if obstacle_dist >= 999.0:
        distance_text = "lejos"
    else:
        distance_text = f"a {obstacle_dist:.1f} m"

    # Dibujar cajas de peatones si existen
    if pedestrian_detected and len(pedestrian_boxes) > 0:

        # Rojo solo cuando el evento principal es peatón cercano
        if event_type == "PEATON CERCANO":
            box_color = red
        else:
            box_color = yellow

        for box in pedestrian_boxes:
            x, y, w, h = box

            cv2.rectangle(img_pedestrian,
                          (x, y),
                          (x + w, y + h),
                          box_color,
                          2)

            cv2.putText(img_pedestrian,
                        "Peaton",
                        (x, max(y - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        box_color,
                        1)

    # Texto principal abajo a la izquierda
    if event_type == "PEATON CERCANO":
        status_text = f"Freno por peaton {distance_text}"
        status_color = red

    elif event_type == "OBSTACULO":
        status_text = f"Freno por obstaculo {distance_text}"
        status_color = red

    elif event_type == "PEATON LEJANO":
        status_text = "Peaton lejano"
        status_color = yellow

    else:
        status_text = "Libre"
        status_color = green

    cv2.putText(img_pedestrian,
                status_text,
                (10, img_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                status_color,
                1)

    return img_pedestrian

# ==============================================================================
# FUNCIÓN DE CONTROL
# ==============================================================================

# Cálculo del error
def get_error(lines, setpoint):
    if lines is None:
        return None, False
    
    # Lista para guardar puntos medios horitzontales
    line_midpoints_x = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Ignorar líneas casi horizontales
        if abs(y2 - y1) < 5:
            continue

        # Punto medio horizontal de la línea detectada
        mid_x = (x1 + x2) / 2.0
        line_midpoints_x.append(mid_x)

    if len(line_midpoints_x) == 0:
        return None, False

    # Promedio de los puntos medios detectados
    detected_line_x = sum(line_midpoints_x) / len(line_midpoints_x)

    # Error respecto al centro de la imagen
    error = detected_line_x - setpoint

    return error, True

# Función PID
def computePID(error, dt, Kp, Ki, Kd, previous_error, integral, output_limits=None):

    # Término proporcional
    P_out = Kp * error

    # Término integral
    integral += error * dt
    I_out = Ki * integral

    # Término derivativo
    if dt > 0:
        derivative = (error - previous_error) / dt
    else:
        derivative = 0
    D_out = Kd * derivative

    # Salida total del PID
    output = P_out + I_out + D_out

    # Limitar salida, si se especifican límites
    if output_limits is not None:
        min_output, max_output = output_limits
        output = max(min_output, min(max_output, output))

    # Actualizar error anterior
    previous_error = error

    return output, previous_error, integral

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main():
    # Condiciones iniciales
    angle = 0.0

    # Instancias de Robot y Driver
    robot = Car()
    driver = Driver() 

    # Se obtiene el tiempo del mundo de Webots
    timestep = int(robot.getBasicTimeStep())

    # Instancia de la cámara
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Instancia del LiDAR Sick
    lidar = robot.getDevice("Sick LMS 291")
    lidar.enable(timestep)
    sick_width = int(lidar.getHorizontalResolution())

    # Cargar modelo SVM para detección de peatones
    svm_model = joblib.load(SVM_MODEL_PATH)
    print("Modelo SVM de peatones cargado correctamente.")

    # Procesado de displays
    display_speed = Display("display_speed")
    display_HSVLane = Display("display_HSVLane")
    display_gray = Display("display_gray")
    display_canny = Display("display_canny")
    display_cannyROI = Display("display_cannyROI")
    display_HoughLines = Display("display_HoughLines")
    display_pedestrian = Display("display_pedestrian")

    # Declaramos constantes de error del control PID
    setpoint = camera.getWidth() / 2.0  # Usamos el punto medio de la cámara como setpoint
    previous_error = 0.0
    integral = 0.0

    # Delcaramos frames para mejorar el rendimiento durante la detección de peatones
    frame_count = 0
    last_pedestrian_detected = False
    last_pedestrian_boxes = []

    while robot.step() != -1:
        # Se obtiene la imágen de la cámara
        image = get_image(camera)
        
        # Se obtiene la información del LiDAR
        sick_data = lidar.getRangeImage()
        obstacle_dist = process_sick_data(sick_data, sick_width)

        # ========== Detección de peatones ==========
        frame_count += 1

        # Por defecto, no se detecta peatón en el frame actual
        pedestrian_detected = False
        pedestrian_boxes = []

        # Se reutiliza la última detección para mejorar rendimiento
        pedestrian_detected = last_pedestrian_detected
        pedestrian_boxes = last_pedestrian_boxes

        # Ejecutar detección cada 5 frames, independientemente de la distancia
        if frame_count % 5 == 0:
            pedestrian_detected, pedestrian_boxes = detect_pedestrian(image, svm_model)

            last_pedestrian_detected = pedestrian_detected
            last_pedestrian_boxes = pedestrian_boxes

        # ========== Detección de carril ==========
        img_HSVLane, img_gray, img_canny, img_cannyROI = get_images_CV2(image, detect_white_lines=False)
        img_HoughLines, lines = get_HoughLines(img_cannyROI, image)
        
        display_image(display_HSVLane, img_HSVLane, "HSV Isolated Lane")
        display_image(display_gray, img_gray, "Grayscale")
        display_image(display_canny, img_canny, "Canny")
        display_image(display_cannyROI, img_cannyROI, "Canny ROI")
        display_image(display_HoughLines, img_HoughLines, "Hough Lines")

        # ========== Cálculo del error ==========
        current_error, line_detected = get_error(lines, setpoint)

        # Time step en segundos
        dt = timestep / 1000.0

        # Control PID para giro del vehículo
        if line_detected:
            angle, previous_error, integral = computePID(error = current_error,
                                                         dt = dt,
                                                         Kp = KP,
                                                         Ki = KI,
                                                         Kd = KD,
                                                         previous_error = previous_error,
                                                         integral = integral,
                                                         output_limits = (-MAX_ANGLE, MAX_ANGLE))

            print(f"Error: {current_error:.2f}, Ángulo de giro: {angle:.3f}")

        else:
            # Si no se detecta nincuna línea, ir derecho
            angle = 0.0
            previous_error = 0.0        # Se vuelve 0 el error e integral para limpiar
            integral = 0.0              # memoria y no arrastrar errores acumulados
            print("No hay linea detectada")
        
        # ========== Lógica de decisión: peatón / obstáculo / vía libre ==========

        close_pedestrian_boxes = get_close_pedestrian_boxes(pedestrian_boxes, image)
        close_pedestrian_detected = len(close_pedestrian_boxes) > 0

        # Caso 1: Peatón u obstáculo cercano detectado
        if obstacle_dist < OBSTACLE_DISTANCE_LIMIT:
            target_speed = 0.0
            target_angle = angle
            brake_intensity = EMERGENCY_BRAKE

            # Peaton
            if close_pedestrian_detected:
                #driver.setHazardFlashers(False)    # Intermitentes apagadas
                hazard_status = "OFF"
                event_type = "PEATON CERCANO"
                print(f"Freno por PEATÓN a {obstacle_dist:.2f} m")
            
            # Obstáculo
            else:
                #driver.setHazardFlashers(True)     # Intermitentes encendidas
                hazard_status = "ON"
                event_type = "OBSTACULO"
                print(f"Freno por OBSTÁCULO a {obstacle_dist:.2f} m")

        # Caso 2: Vía libre
        else:
            target_speed = CRUISING_SPEED
            target_angle = angle
            brake_intensity = 0.0
            #driver.setHazardFlashers(False)        # Intermitentes apagadas
            hazard_status = "OFF"

            if pedestrian_detected and len(pedestrian_boxes) > 0:
                event_type = "PEATON LEJANO"
            else:
                event_type = "LIBRE"
        
        # Display de detección de peatones y obstáculos
        img_pedestrian = get_pedestrianDisplayImage(image,pedestrian_detected, pedestrian_boxes, obstacle_dist, event_type)
        display_image(display_pedestrian, img_pedestrian, "Pedestrian Detection")

        # ========== Actualizar comandos del vehículo ==========
        driver.setSteeringAngle(target_angle)
        driver.setCruisingSpeed(target_speed)
        driver.setBrakeIntensity(brake_intensity)

        # Mostrar velocidad y ángulo de giro actual
        current_speed = driver.getCurrentSpeed()
        display_status(display_speed, current_speed, target_angle, brake_intensity, hazard_status)

if __name__ == "__main__":
    main()