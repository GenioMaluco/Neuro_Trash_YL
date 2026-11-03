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
HITL_DIR = DATASET_DIR / "hitl"
HITL_CATALOG_PATH = HITL_DIR / "catalog.json"
CAPTURES_DIR = DATASET_DIR / "capturas"

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
        "category_id": "CAT-PL",
        "code": "PL",
        "train_class": "plastic",
        "rainbow_name": "Rojo neón (Arcoíris #1)",
        "bgr": (68, 23, 255),
        "rgb": (255, 23, 68),
        "hex": "#FF1744",
        "label_fg": "#FFFFFF",
        "container": "Contenedor Azul/Rojo - Plásticos Reciclables (PET/HDPE)",
        "servo_angle": 30,
        "recyclable": True,
    },
    "Metal": {
        "id": 1,
        "category_id": "CAT-MT",
        "code": "MT",
        "train_class": "metal",
        "rainbow_name": "Naranja neón (Arcoíris #2)",
        "bgr": (0, 145, 255),
        "rgb": (255, 145, 0),
        "hex": "#FF9100",
        "label_fg": "#111111",
        "container": "Contenedor Amarillo - Metales y Latas de Aluminio",
        "servo_angle": 60,
        "recyclable": True,
    },
    "Papel/Cartón": {
        "id": 2,
        "category_id": "CAT-PC",
        "code": "PC",
        "train_class": "paper",
        "rainbow_name": "Amarillo neón (Arcoíris #3)",
        "bgr": (0, 234, 255),
        "rgb": (255, 234, 0),
        "hex": "#FFEA00",
        "label_fg": "#111111",
        "container": "Contenedor Gris/Azul - Papel y Cartón Limpio",
        "servo_angle": 90,
        "recyclable": True,
    },
    "Orgánico": {
        "id": 3,
        "category_id": "CAT-OR",
        "code": "OR",
        "train_class": "organic",
        "rainbow_name": "Verde neón (Arcoíris #4)",
        "bgr": (20, 255, 57),
        "rgb": (57, 255, 20),
        "hex": "#39FF14",
        "label_fg": "#111111",
        "container": "Contenedor Verde - Residuos Orgánicos / Compostaje",
        "servo_angle": 120,
        "recyclable": True,
    },
    "Vidrio": {
        "id": 4,
        "category_id": "CAT-GL",
        "code": "GL",
        "train_class": "glass",
        "rainbow_name": "Cian neón (Arcoíris #5)",
        "bgr": (255, 245, 0),
        "rgb": (0, 245, 255),
        "hex": "#00F5FF",
        "label_fg": "#111111",
        "container": "Contenedor Verde Claro - Envases de Vidrio",
        "servo_angle": 150,
        "recyclable": True,
    },
    "Desecho General": {
        "id": 5,
        "category_id": "CAT-DG",
        "code": "DG",
        "train_class": "trash",
        "rainbow_name": "Violeta neón (Arcoíris #6)",
        "bgr": (255, 77, 124),
        "rgb": (124, 77, 255),
        "hex": "#7C4DFF",
        "label_fg": "#FFFFFF",
        "container": "Contenedor Negro - Residuos No Valorizables / Especiales",
        "servo_angle": 175,
        "recyclable": False,
    },
    "No Identificado": {
        "id": 6,
        "category_id": "CAT-NI",
        "code": "NI",
        "train_class": None,
        "rainbow_name": "Magenta neón (Arcoíris #7)",
        "bgr": (229, 0, 255),
        "rgb": (255, 0, 229),
        "hex": "#FF00E5",
        "label_fg": "#FFFFFF",
        "container": "Revisión Manual / Espera de Confirmación",
        "servo_angle": 0,
        "recyclable": False,
    },
}

CATEGORY_BY_ID = {meta["category_id"]: name for name, meta in CATEGORIES.items()}
TRAINABLE_CLASSES = ("plastic", "metal", "paper", "cardboard", "glass", "trash", "organic")

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

# Prompts ampliados para YOLOWorld (botellas, tetra, wrappers, utensilios, restos de comida)
WORLD_WASTE_PROMPTS = [
    # Plástico
    "plastic bottle",
    "water bottle",
    "PET bottle",
    "plastic cup",
    "plastic bag",
    "plastic container",
    "plastic wrapper",
    "food wrapper",
    "plastic utensil",
    "disposable fork",
    "drinking straw",
    "shampoo bottle",
    "soda bottle",
    # Metal
    "aluminum can",
    "soda can",
    "tin can",
    "metal can",
    "aluminum foil",
    "fork",
    "spoon",
    "knife",
    "coin",
    # Papel / Cartón
    "cardboard box",
    "cardboard",
    "paper sheet",
    "crumpled paper",
    "document",
    "paper cup",
    "coffee cup",
    "newspaper",
    "carton",
    "tetra pak",
    "milk carton",
    "paper bag",
    "pizza box",
    # Vidrio
    "glass bottle",
    "wine glass",
    "glass jar",
    "jar",
    "beer bottle",
    "glass container",
    # Orgánico
    "banana",
    "banana peel",
    "apple",
    "orange",
    "fruit peel",
    "food waste",
    "food leftovers",
    "vegetable",
    "bread",
    "leaf",
    "eggshell",
    # Desecho General / Peligroso
    "AA battery",
    "battery",
    "snack wrapper",
    "candy wrapper",
    "chip bag",
    "styrofoam",
    "trash",
    "computer mouse",
    "cell phone",
]

# Mapeo de prompts de YOLOWorld y clases COCO hacia nuestras 7 categorías y sub-etiquetas en español
PROMPT_TO_CATEGORY = {
    # Plástico
    "plastic bottle": ("Plástico", "Botella plástica"),
    "water bottle": ("Plástico", "Botella PET"),
    "PET bottle": ("Plástico", "Botella PET"),
    "plastic cup": ("Plástico", "Vaso plástico"),
    "plastic bag": ("Plástico", "Bolsa plástica"),
    "plastic container": ("Plástico", "Envase plástico"),
    "plastic wrapper": ("Plástico", "Envoltorio plástico"),
    "food wrapper": ("Plástico", "Envoltorio de comida"),
    "plastic utensil": ("Plástico", "Utensilio plástico"),
    "disposable fork": ("Plástico", "Cubierto desechable"),
    "soda bottle": ("Plástico", "Botella PET"),
    "drinking straw": ("Plástico", "Pajilla / Pitillo"),
    "shampoo bottle": ("Plástico", "Envase de shampoo"),
    "bottle": ("Plástico", "Botella"),
    "cup": ("Plástico", "Vaso/Recipiente"),
    # Metal
    "aluminum can": ("Metal", "Lata de aluminio"),
    "soda can": ("Metal", "Lata de refresco"),
    "tin can": ("Metal", "Lata metálica"),
    "metal can": ("Metal", "Envase metálico"),
    "aluminum foil": ("Metal", "Papel aluminio"),
    "coin": ("Metal", "Moneda metálica"),
    "fork": ("Metal", "Cubierto metálico"),
    "knife": ("Metal", "Cubierto metálico"),
    "spoon": ("Metal", "Cuchara"),
    "scissors": ("Metal", "Tijeras metálicas"),
    # Papel / Cartón
    "cardboard box": ("Papel/Cartón", "Caja de cartón"),
    "cardboard": ("Papel/Cartón", "Cartón"),
    "paper sheet": ("Papel/Cartón", "Hoja de papel"),
    "crumpled paper": ("Papel/Cartón", "Papel arrugado"),
    "document": ("Papel/Cartón", "Documento/Papel"),
    "paper cup": ("Papel/Cartón", "Vaso de cartón"),
    "coffee cup": ("Papel/Cartón", "Vaso de café"),
    "newspaper": ("Papel/Cartón", "Periódico/Papel"),
    "carton": ("Papel/Cartón", "Empaque Tetra/Cartón"),
    "tetra pak": ("Papel/Cartón", "Tetra Pak"),
    "milk carton": ("Papel/Cartón", "Envase de cartón"),
    "paper bag": ("Papel/Cartón", "Bolsa de papel"),
    "pizza box": ("Papel/Cartón", "Caja de pizza"),
    "book": ("Papel/Cartón", "Libro/Papel"),
    # Vidrio
    "glass bottle": ("Vidrio", "Botella de vidrio"),
    "wine glass": ("Vidrio", "Copa/Vaso de vidrio"),
    "glass jar": ("Vidrio", "Frasco de vidrio"),
    "jar": ("Vidrio", "Frasco"),
    "beer bottle": ("Vidrio", "Botella de vidrio"),
    "glass container": ("Vidrio", "Envase de vidrio"),
    "vase": ("Vidrio", "Recipiente de vidrio"),
    # Orgánico
    "banana": ("Orgánico", "Banano / Cáscara"),
    "banana peel": ("Orgánico", "Cáscara de banano"),
    "apple": ("Orgánico", "Manzana / Fruta"),
    "orange": ("Orgánico", "Naranja / Cítrico"),
    "fruit peel": ("Orgánico", "Cáscara orgánica"),
    "food waste": ("Orgánico", "Resto de comida"),
    "food leftovers": ("Orgánico", "Sobras de comida"),
    "vegetable": ("Orgánico", "Vegetal orgánico"),
    "bread": ("Orgánico", "Pan / Alimento"),
    "leaf": ("Orgánico", "Hoja / Biomasa"),
    "egg shell": ("Orgánico", "Cáscara de huevo"),
    "eggshell": ("Orgánico", "Cáscara de huevo"),
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
    "candy wrapper": ("Desecho General", "Envoltorio de dulce"),
    "chip bag": ("Desecho General", "Bolsa de snacks"),
    "styrofoam": ("Desecho General", "Poliestireno / Unicel"),
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
    "organic": ("Orgánico", "Residuo orgánico"),
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
DEFAULT_CONF_THRESHOLD = 0.20
DEFAULT_UNKNOWN_THRESHOLD = 0.35  # Si la confianza combinada es menor a esto, se marca como "No Identificado"
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_STABILITY_WINDOW = 5      # Número de frames para estabilizar la categoría de cada objeto
TARGET_LATENCY_MS = 180.0         # Meta de tiempo real definida en los Entregables 1 a 5 (120-180 ms)
