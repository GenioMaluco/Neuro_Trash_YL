"""
Configuración central de Neuro_Trash_YL:
Taxonomía de residuos, paleta de colores Arcoíris (BGR y RGB/Hex),
mapeo de clases de YOLO (COCO / YOLO-World / Clasificador TrashNet) y parámetros AIoT.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_DIR
RUNS_DIR = PROJECT_DIR / "runs"
DATASET_DIR = PROJECT_DIR / "dataset"
TEST_IMAGES_DIR = DATASET_DIR / "img_test"

# Pesos de modelos disponibles en el repositorio
MODEL_DETECTOR_WORLD = PROJECT_DIR / "yolov8s-world.pt"
MODEL_DETECTOR_BASE = PROJECT_DIR / "yolov8n.pt"
MODEL_CLASSIFIER_TRAINED = RUNS_DIR / "residuos" / "weights" / "best.pt"
MODEL_CLASSIFIER_FALLBACK = PROJECT_DIR / "yolov8n-cls.pt"

# ============================================================================
# CATEGORÍAS Y PALETA ARCOÍRIS (7 COLORES)
# OpenCV usa formato BGR; para la interfaz web / leyendas proveemos HEX y RGB.
# ============================================================================
CATEGORIES = {
    "Plástico": {
        "id": 0,
        "rainbow_name": "Rojo (Arcoíris #1)",
        "bgr": (36, 28, 237),       # Rojo vibrante en BGR
        "rgb": (237, 28, 36),
        "hex": "#ED1C24",
        "container": "Contenedor Azul/Rojo - Plásticos Reciclables (PET/HDPE)",
        "servo_angle": 30,
        "recyclable": True,
    },
    "Metal": {
        "id": 1,
        "rainbow_name": "Naranja (Arcoíris #2)",
        "bgr": (0, 140, 255),       # Naranja brillante en BGR
        "rgb": (255, 140, 0),
        "hex": "#FF8C00",
        "container": "Contenedor Amarillo - Metales y Latas de Aluminio",
        "servo_angle": 60,
        "recyclable": True,
    },
    "Papel/Cartón": {
        "id": 2,
        "rainbow_name": "Amarillo (Arcoíris #3)",
        "bgr": (0, 225, 255),       # Amarillo intenso en BGR
        "rgb": (255, 225, 0),
        "hex": "#FFE100",
        "container": "Contenedor Gris/Azul - Papel y Cartón Limpio",
        "servo_angle": 90,
        "recyclable": True,
    },
    "Orgánico": {
        "id": 3,
        "rainbow_name": "Verde (Arcoíris #4)",
        "bgr": (50, 205, 50),       # Verde esmeralda en BGR
        "rgb": (50, 205, 50),
        "hex": "#32CD32",
        "container": "Contenedor Verde - Residuos Orgánicos / Compostaje",
        "servo_angle": 120,
        "recyclable": True,
    },
    "Vidrio": {
        "id": 4,
        "rainbow_name": "Azul Cian (Arcoíris #5)",
        "bgr": (255, 191, 0),       # Azul Cian / Celeste en BGR
        "rgb": (0, 191, 255),
        "hex": "#00BFFF",
        "container": "Contenedor Verde Claro - Envases de Vidrio",
        "servo_angle": 150,
        "recyclable": True,
    },
    "Desecho General": {
        "id": 5,
        "rainbow_name": "Índigo (Arcoíris #6)",
        "bgr": (205, 90, 106),      # Índigo / Azul violáceo en BGR
        "rgb": (106, 90, 205),
        "hex": "#6A5ACD",
        "container": "Contenedor Negro - Residuos No Valorizables / Especiales",
        "servo_angle": 175,
        "recyclable": False,
    },
    "No Identificado": {
        "id": 6,
        "rainbow_name": "Violeta / Magenta (Arcoíris #7)",
        "bgr": (255, 0, 212),       # Violeta / Magenta en BGR
        "rgb": (212, 0, 255),
        "hex": "#D400FF",
        "container": "Revisión Manual / Espera de Confirmación",
        "servo_angle": 0,
        "recyclable": False,
    },
}

# Clases a ignorar cuando provienen del detector base (personas/fondo) para no marcar a la persona como basura
IGNORED_DETECTOR_CLASSES = {
    "person",
    "chair",
    "couch",
    "bed",
    "dining table",
    "tv",
    "laptop",
    "keyboard",
    "car",
    "truck",
    "bus",
}

# Prompts optimizados para YOLOWorld (detección abierta centrada en residuos y objetos)
WORLD_WASTE_PROMPTS = [
    # Plástico
    "plastic bottle",
    "water bottle",
    "plastic cup",
    "plastic bag",
    "plastic container",
    # Metal
    "aluminum can",
    "soda can",
    "tin can",
    "metal can",
    "coin",
    # Papel / Cartón
    "cardboard box",
    "paper sheet",
    "document",
    "paper cup",
    "newspaper",
    "carton",
    # Vidrio
    "glass bottle",
    "wine glass",
    "glass jar",
    # Orgánico
    "banana",
    "apple",
    "orange",
    "fruit peel",
    "food waste",
    "vegetable",
    "bread",
    "leaf",
    # Desecho General / Peligroso
    "AA battery",
    "battery",
    "snack wrapper",
    "trash",
    "computer mouse",
    "cell phone",
]

# Mapeo de prompts de YOLOWorld y clases COCO hacia nuestras 7 categorías y sub-etiquetas en español
PROMPT_TO_CATEGORY = {
    # Plástico
    "plastic bottle": ("Plástico", "Botella plástica"),
    "water bottle": ("Plástico", "Botella PET"),
    "plastic cup": ("Plástico", "Vaso plástico"),
    "plastic bag": ("Plástico", "Bolsa plástica"),
    "plastic container": ("Plástico", "Envase plástico"),
    "bottle": ("Plástico", "Botella"),
    "cup": ("Plástico", "Vaso/Recipiente"),
    # Metal
    "aluminum can": ("Metal", "Lata de aluminio"),
    "soda can": ("Metal", "Lata de refresco"),
    "tin can": ("Metal", "Lata metálica"),
    "metal can": ("Metal", "Envase metálico"),
    "coin": ("Metal", "Moneda metálica"),
    "fork": ("Metal", "Cubierto metálico"),
    "knife": ("Metal", "Cubierto metálico"),
    "spoon": ("Metal", "Cuchara"),
    "scissors": ("Metal", "Tijeras metálicas"),
    # Papel / Cartón
    "cardboard box": ("Papel/Cartón", "Caja de cartón"),
    "paper sheet": ("Papel/Cartón", "Hoja de papel"),
    "document": ("Papel/Cartón", "Documento/Papel"),
    "paper cup": ("Papel/Cartón", "Vaso de cartón"),
    "newspaper": ("Papel/Cartón", "Periódico/Papel"),
    "carton": ("Papel/Cartón", "Empaque Tetra/Cartón"),
    "book": ("Papel/Cartón", "Libro/Papel"),
    # Vidrio
    "glass bottle": ("Vidrio", "Botella de vidrio"),
    "wine glass": ("Vidrio", "Copa/Vaso de vidrio"),
    "glass jar": ("Vidrio", "Frasco de vidrio"),
    "vase": ("Vidrio", "Recipiente de vidrio"),
    # Orgánico
    "banana": ("Orgánico", "Banano / Cáscara"),
    "apple": ("Orgánico", "Manzana / Fruta"),
    "orange": ("Orgánico", "Naranja / Cítrico"),
    "fruit peel": ("Orgánico", "Cáscara orgánica"),
    "food waste": ("Orgánico", "Resto de comida"),
    "vegetable": ("Orgánico", "Vegetal orgánico"),
    "bread": ("Orgánico", "Pan / Alimento"),
    "leaf": ("Orgánico", "Hoja / Biomasa"),
    "broccoli": ("Orgánico", "Vegetal (Brócoli)"),
    "carrot": ("Orgánico", "Vegetal (Zanahoria)"),
    "sandwich": ("Orgánico", "Resto de alimento"),
    "pizza": ("Orgánico", "Resto de comida"),
    "donut": ("Orgánico", "Resto de alimento"),
    "cake": ("Orgánico", "Resto de alimento"),
    "potted plant": ("Orgánico", "Residuo vegetal"),
    # Desecho General / Peligroso
    "AA battery": ("Desecho General", "Batería AA (Especial)"),
    "battery": ("Desecho General", "Batería (Especial)"),
    "snack wrapper": ("Desecho General", "Empaque metalizado"),
    "trash": ("Desecho General", "Desecho mixto"),
    "computer mouse": ("Desecho General", "RAEE (Mouse)"),
    "cell phone": ("Desecho General", "RAEE (Celular)"),
    "toothbrush": ("Desecho General", "Cepillo dental"),
}

# Mapeo del modelo entrenado en Neuro_Trash_YL (runs/residuos/weights/best.pt - TrashNet 6 clases)
TRASHNET_TO_CATEGORY = {
    "plastic": ("Plástico", "Plástico reciclable"),
    "metal": ("Metal", "Metal / Lata"),
    "paper": ("Papel/Cartón", "Papel reciclable"),
    "cardboard": ("Papel/Cartón", "Cartón reciclable"),
    "glass": ("Vidrio", "Vidrio reciclable"),
    "trash": ("Desecho General", "Desecho no valorizable"),
}

# Parámetros por defecto del motor de inferencia y telemetría Edge/Fog (Nodo Raspberry Pi)
EDGE_NODE_HARDWARE = "Raspberry Pi 4 / 5 (Single-Board Computer - Linux ARM64)"
EDGE_ACTUATOR_GPIO_PINS = {
    "servo_compuerta_pwm": 18,    # GPIO 18 (Pin físico 12 - PWM0 por hardware en Raspberry Pi)
    "ultrasonico_trig": 23,       # GPIO 23 (Sensor HC-SR04 para nivel de llenado)
    "ultrasonico_echo": 24,       # GPIO 24
    "led_estado_verde": 27,       # GPIO 27 (Indicador de residuo reciclable/orgánico)
    "led_estado_rojo": 22,        # GPIO 22 (Indicador de alerta / no identificado)
}
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_UNKNOWN_THRESHOLD = 0.42  # Si la confianza combinada es menor a esto, se marca como "No Identificado"
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_STABILITY_WINDOW = 5      # Número de frames para estabilizar la categoría de cada objeto
TARGET_LATENCY_MS = 180.0         # Meta de tiempo real definida en los Entregables 1 a 5 (120-180 ms)
