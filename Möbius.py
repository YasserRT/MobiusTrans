import cv2
import numpy as np

def mobius_transformation(image):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    max_radius = min(center)

    # Crea una meshgrid de coordenadas
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    # Convertir a coordenadas polares
    X -= center[0]
    Y -= center[1]
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # Define la función transformación(creo que este es el problema)
    def transformation(r, t):
        a = 1
        b = 1
        c = 1
        d = -1
        f = 0.5
        g = 0.5
        h = 0
        return (a * r + b) / (c * r + d) * np.exp(1j * (t + f)) + complex(g, h)

    # Aplica la transformación
    R_transformed = np.abs(transformation(R / max_radius, theta)) * max_radius
    X_transformed = R_transformed * np.cos(theta) + center[0]
    Y_transformed = R_transformed * np.sin(theta) + center[1]

    # Interpola valores para conseguir la imagen transformada
    transformed_image = cv2.remap(image, X_transformed.astype(np.float32), Y_transformed.astype(np.float32),
                                  interpolation=cv2.INTER_LINEAR)

    return transformed_image

# Carga la imagen a tratar
image = cv2.imread('original.png')

# Comprobar si la imagen es escala de grises(me daba problemas si lo era)
if len(image.shape) == 2:
    # Si la imagen es escala de grises pasarla a 3-canales
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Aplica transformación de Möbius
transformed_image = mobius_transformation(image)

# Muestra las imagenes original y tratada.
cv2.imshow('Original Image', image)
cv2.imshow('Transformed Image', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
