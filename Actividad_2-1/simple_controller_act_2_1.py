import os
import sys

# ==============================================================================
# SOLUCIÓN DE RUTAS (Webots + Python)
# ==============================================================================
webots_path = r'C:\Program Files\Webots\lib\controller\python'
if webots_path not in sys.path:
    sys.path.append(webots_path)

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2

# ==============================================================================
# CONFIGURACIÓN PID Y CONSTANTES
# ==============================================================================
MAX_ANGLE = 0.5
CRUISING_SPEED = 50 

Kp = 0.006  
Ki = 0.000
Kd = 0.002  

prev_error = 0.0
integral = 0.0

# ==============================================================================
# FUNCIONES DE VISIÓN
# ==============================================================================

def get_image(camera):
    raw_image = camera.getImage()  
    # Webots entrega BGRA, lo pasamos a matriz numpy
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

def process_lane_detection(image):
    height, width = image.shape[:2]
    
    # 1. FILTRADO HSV MÁS AMPLIO (Permisivo con sombras y luz)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Bajamos el valor mínimo de saturación (100 -> 50) y brillo (100 -> 50)
    # para detectar amarillo incluso en zonas oscuras o con sombras de edificios
    lower_yellow = np.array([15, 50, 50]) 
    upper_yellow = np.array([35, 255, 255])
    
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_isolated = cv2.bitwise_and(image, image, mask=mask_yellow)
    
    # 2. PROCESAMIENTO
    gray_img = cv2.cvtColor(yellow_isolated, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 30, 120) # Umbrales de Canny más bajos para detectar bordes suaves
    
    # 3. ROI MÁS ANCHA (Para no perder la línea en curvas)
    mask_roi = np.zeros_like(edges)
    # Abrimos la parte superior del trapecio (0.40 -> 0.10 y 0.60 -> 0.90)
    # Esto permite que el robot "vea" las líneas que se van a las esquinas en la curva
    polygon = np.array([[
        (0 , height),
        (width * 0.1, height * 0.6), 
        (width * 0.9, height * 0.6), 
        (width, height) 
    ]], np.int32)
    
    cv2.fillPoly(mask_roi, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask_roi)
    
    # 4. HOUGH MÁS SENSIBLE
    # Bajamos el threshold (15 -> 10) y minLineLength (10 -> 5)
    # para que detecte incluso trozos pequeños de línea amarilla
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 10, minLineLength=5, maxLineGap=25)
    
    return masked_edges, lines

def draw_lines(image, lines):
    """Dibuja las líneas detectadas sobre una copia de la imagen original."""
    line_image = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3) # Verde
    return line_image

# ==============================================================================
# BUCLE PRINCIPAL
# ==============================================================================

# ... (importaciones y funciones previas iguales)

def main():
    global prev_error, integral
    
    robot = Car() # Tu instancia se llama 'robot'
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)
    
    robot.step() 
    camera_width = camera.getWidth()
    setpoint = camera_width / 2.0

    while robot.step() != -1:
        image = get_image(camera)
        masked_edges, lines = process_lane_detection(image)
        
        # --- OBTENER VELOCIDAD ---
        # Se obtiene dentro del loop. 'driver' tiene el método getCurrentSpeed()
        velocidad_actual = driver.getCurrentSpeed()

        # --- VISUALIZACIÓN EN VENTANA ---
        frame_with_lines = draw_lines(image, lines)
        
        # Dibujamos el texto de la velocidad en la imagen
        # cv2.putText(imagen, texto, posicion, fuente, escala, color, grosor)
        cv2.putText(frame_with_lines, f"Vel: {velocidad_actual:.2f} km/h", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imshow("Algoritmo: Canny + ROI", masked_edges)
        cv2.imshow("Deteccion de Carriles - Webots", frame_with_lines) # Descomentada para ver velocidad
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # --- LÓGICA PID CORREGIDA ---
        current_error = 0
        line_detected = False

        if lines is not None:
            all_mids = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 15: continue 
                all_mids.append((x1 + x2) / 2.0)
            
            if len(all_mids) > 0:
                mid_x = sum(all_mids) / len(all_mids)
                # Calculamos el error una sola vez
                current_error = mid_x - setpoint 
                line_detected = True

        if line_detected:
            proporcional = Kp * current_error
            integral += current_error
            derivativo = Kd * (current_error - prev_error)
            
            steer_angle = proporcional + (Ki * integral) + derivativo
            prev_error = current_error
        else:
            steer_angle = 0.0

        driver.setSteeringAngle(np.clip(steer_angle, -MAX_ANGLE, MAX_ANGLE))
        driver.setCruisingSpeed(CRUISING_SPEED)
        
        # Imprimir en consola de Webots también
        # print(f"Velocidad actual: {velocidad_actual}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()