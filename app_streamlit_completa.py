"""
TAREA 2: FILTROS Y DESCRIPTORES DE IMÁGENES
Interfaz gráfica completa con Streamlit

Funcionalidades:
- PARTE 1: Aplicación interactiva de filtros
- PARTE 2: Extracción de descriptores (HOG, LBP) y clasificación (SVM, CNN)
- Entrenamiento de modelos
- Clasificación de nuevas imágenes
- Visualización de métricas y resultados
"""

import streamlit as st
from pathlib import Path
from io import BytesIO
import pickle
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Machine Learning
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from skimage.feature import hog, local_binary_pattern

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Configuración de la página
st.set_page_config(
    page_title="Filtros y Descriptores - Tarea 2",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths de modelos
CNN_MODEL_PATH = MODEL_DIR / 'cnn_plate_classifier.pth'
SVM_HOG_PATH = MODEL_DIR / 'svm_hog_classifier.pkl'
SVM_LBP_PATH = MODEL_DIR / 'svm_lbp_classifier.pkl'
CLASSES_PATH = MODEL_DIR / 'classes.npy'
CONFIG_PATH = MODEL_DIR / 'descriptor_config.pkl'

# ============================================================================
# DEFINICIÓN DE ARQUITECTURA CNN
# ============================================================================

class PlateCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

@st.cache_data
def load_image(image_file):
    """Carga una imagen desde el archivo subido."""
    img = Image.open(image_file).convert('RGB')
    return img

def apply_filter(image_np, filter_type, **kwargs):
    """
    Aplica un filtro específico a la imagen.
    
    Args:
        image_np: Imagen en formato numpy (grayscale)
        filter_type: Tipo de filtro a aplicar
        kwargs: Parámetros adicionales del filtro
    """
    if filter_type == "Media":
        kernel_size = kwargs.get('kernel_size', 5)
        return cv2.blur(image_np, (kernel_size, kernel_size))
    
    elif filter_type == "Mediana":
        kernel_size = kwargs.get('kernel_size', 5)
        return cv2.medianBlur(image_np, kernel_size)
    
    elif filter_type == "Logarítmico":
        normalized = image_np.astype(np.float32) / 255.0
        c = 255.0 / np.log(1 + 1)
        result = c * np.log(1 + normalized)
        return np.uint8(result)
    
    elif filter_type == "Cuadro Normalizado":
        kernel_size = kwargs.get('kernel_size', 7)
        return cv2.boxFilter(image_np, -1, (kernel_size, kernel_size), normalize=True)
    
    elif filter_type == "Gaussiano":
        kernel_size = kwargs.get('kernel_size', 5)
        sigma = kwargs.get('sigma', 1.0)
        return cv2.GaussianBlur(image_np, (kernel_size, kernel_size), sigma)
    
    elif filter_type == "Laplace":
        laplacian = cv2.Laplacian(image_np, cv2.CV_64F)
        return np.uint8(np.absolute(laplacian))
    
    elif filter_type == "Sobel":
        sobel_x = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        return np.uint8(sobel)
    
    elif filter_type == "Canny":
        threshold1 = kwargs.get('threshold1', 50)
        threshold2 = kwargs.get('threshold2', 150)
        return cv2.Canny(image_np, threshold1, threshold2)
    
    return image_np

def extract_hog_features(image_gray, hog_params):
    """Extrae características HOG."""
    features = hog(image_gray, **hog_params)
    return features

def extract_lbp_features(image_gray, radius, n_points):
    """Extrae características LBP."""
    lbp = local_binary_pattern(image_gray, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

# ============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================================

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, eval_transform

def create_dataloaders(data_root: Path, batch_size: int):
    train_dir = data_root / 'train'
    val_dir = data_root / 'val'
    
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError('Se requieren carpetas train/ y val/ dentro de la ruta indicada.')

    train_transform, eval_transform = get_transforms()
    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_ds.classes, eval_transform

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc

def train_cnn_model(data_root: Path, epochs: int, lr: float, batch_size: int, weight_decay: float):
    """Entrena el modelo CNN."""
    train_loader, val_loader, class_names, eval_transform = create_dataloaders(data_root, batch_size)
    model = PlateCNN(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(
            f"Época {epoch + 1}/{epochs} — "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    progress_bar.empty()
    status_text.empty()
    
    return model, class_names, eval_transform, history

# ============================================================================
# FUNCIONES DE GUARDADO/CARGA
# ============================================================================

def save_models(cnn_model, svm_hog, svm_lbp, class_names, config):
    """Guarda todos los modelos y configuraciones."""
    if cnn_model is not None:
        torch.save(cnn_model.cpu().state_dict(), CNN_MODEL_PATH)
        cnn_model.to(DEVICE)
    
    if svm_hog is not None:
        with open(SVM_HOG_PATH, 'wb') as f:
            pickle.dump(svm_hog, f)
    
    if svm_lbp is not None:
        with open(SVM_LBP_PATH, 'wb') as f:
            pickle.dump(svm_lbp, f)
    
    if class_names:
        np.save(CLASSES_PATH, np.array(class_names))
    
    if config:
        with open(CONFIG_PATH, 'wb') as f:
            pickle.dump(config, f)

def load_models():
    """Carga todos los modelos guardados."""
    models = {}
    
    if CNN_MODEL_PATH.exists() and CLASSES_PATH.exists() and CONFIG_PATH.exists():
        class_names = np.load(CLASSES_PATH, allow_pickle=True)
        
        with open(CONFIG_PATH, 'rb') as f:
            config = pickle.load(f)
        
        # Cargar CNN
        cnn_model = PlateCNN(num_classes=len(class_names))
        cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE, weights_only=False))
        cnn_model = cnn_model.to(DEVICE)
        cnn_model.eval()
        models['cnn'] = cnn_model
        
        # Cargar SVM+HOG
        if SVM_HOG_PATH.exists():
            with open(SVM_HOG_PATH, 'rb') as f:
                models['svm_hog'] = pickle.load(f)
        
        # Cargar SVM+LBP
        if SVM_LBP_PATH.exists():
            with open(SVM_LBP_PATH, 'rb') as f:
                models['svm_lbp'] = pickle.load(f)
        
        _, eval_transform = get_transforms()
        
        return models, class_names, config, eval_transform
    
    return None, None, None, None

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    st.title("🖼️ Filtros y Descriptores de Imágenes")
    st.markdown("### Tarea 2: Procesamiento de Imágenes y Clasificación")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Selección de modo
        modo = st.radio(
            "Selecciona el modo:",
            ["📚 Teoría de Filtros", "🔍 Filtros (Parte 1)", "🤖 Descriptores y Clasificación (Parte 2)", "🎯 Clasificar Imagen"]
        )
        
        st.divider()
        
        # Inicializar variables por defecto
        data_root_input = 'data'
        train_cnn = False
        train_svm_hog = False
        train_svm_lbp = False
        epochs_cnn = 10
        batch_size = 32
        learning_rate = 1e-3
        weight_decay = 1e-4
        
        if modo == "🤖 Descriptores y Clasificación (Parte 2)":
            st.subheader("Configuración de Entrenamiento")
            data_root_input = st.text_input('Ruta del dataset', 'data')
            
            st.subheader("Selecciona modelos a entrenar")
            train_cnn = st.checkbox("CNN (PyTorch)", value=True)
            train_svm_hog = st.checkbox("SVM + HOG", value=True)
            train_svm_lbp = st.checkbox("SVM + LBP", value=True)
            
            if train_cnn:
                st.markdown("**Hiperparámetros CNN:**")
                epochs_cnn = st.slider('Épocas', 1, 30, 10)
                batch_size = st.selectbox('Batch size', [16, 32, 64], index=1)
                learning_rate = st.number_input('Learning rate', 1e-5, 1e-2, 1e-3, format='%e')
                weight_decay = st.number_input('Weight decay', 0.0, 1e-1, 1e-4, format='%e')
    
    # ========================================================================
    # MODO 0: TEORÍA DE FILTROS
    # ========================================================================
    
    if modo == "📚 Teoría de Filtros":
        st.header("Parte 1: Investigación Teórica de Filtros")
        
        # Crear tabs para cada filtro
        tabs = st.tabs([
            "📋 Resumen",
            "Media",
            "Mediana", 
            "Logarítmico",
            "Cuadro Normalizado",
            "Gaussiano",
            "Laplace",
            "Sobel",
            "Canny"
        ])
        
        # Tab 0: Resumen general
        with tabs[0]:
            st.markdown("""
            ## 🎯 Resumen de Filtros de Imágenes
            
            Los filtros de imágenes son operaciones fundamentales en el procesamiento digital de imágenes
            que transforman una imagen de entrada en una imagen de salida mediante la aplicación de 
            operadores matemáticos específicos.
            
            ### Clasificación General
            
            #### 🔷 Filtros de Suavizado
            - **Filtro de Media**: Promedio aritmético de píxeles vecinos
            - **Filtro de Mediana**: Valor mediano de píxeles vecinos
            - **Filtro de Cuadro Normalizado**: Promedio con máscara uniforme
            - **Filtro Gaussiano**: Convolución con función gaussiana
            
            #### 🔶 Filtros de Transformación
            - **Filtro Logarítmico**: Compresión de rango dinámico
            
            #### 🔴 Detectores de Bordes
            - **Filtro Laplace**: Derivada de segundo orden
            - **Filtro Sobel**: Derivada de primer orden (gradiente)
            - **Filtro Canny**: Detector multi-etapa optimizado
            
            ### 📊 Comparativa Rápida
            """)
            
            # Tabla comparativa
            comparison_data = {
                "Filtro": ["Media", "Mediana", "Logarítmico", "Cuadro Norm.", "Gaussiano", "Laplace", "Sobel", "Canny"],
                "Tipo": ["Suavizado", "Suavizado", "Transformación", "Suavizado", "Suavizado", "Bordes", "Bordes", "Bordes"],
                "Velocidad": ["⚡⚡⚡", "⚡⚡", "⚡⚡⚡", "⚡⚡⚡", "⚡⚡", "⚡⚡⚡", "⚡⚡⚡", "⚡"],
                "vs Ruido": ["Media", "Excelente", "Baja", "Media", "Buena", "Baja", "Media", "Buena"],
                "Uso Principal": ["Ruido gaussiano", "Ruido impulsivo", "HDR", "Suavizado", "Ruido+bordes", "Realce", "Detección", "Detección óptima"]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, hide_index=True, use_container_width=True)
            
            st.info("👉 **Selecciona una pestaña arriba** para ver la teoría detallada de cada filtro.")
        
        # Tab 1: Filtro de Media
        with tabs[1]:
            st.markdown("""
            ## Filtro de Media
            
            ### 📖 Definición
            Sustituye cada píxel por el **promedio aritmético** de sus vecinos dentro de una ventana rectangular,
            suavizando el ruido gaussiano o de intensidad baja.
            
            ### 📐 Fórmula Matemática
            Para una máscara de tamaño $m \\times n$:
            """)
            
            st.latex(r"g(x,y) = \frac{1}{mn}\sum_{i=-a}^{a}\sum_{j=-b}^{b} f(x+i, y+j)")
            
            st.markdown("""
            Donde:
            - $f(x,y)$ es la imagen original
            - $g(x,y)$ es la imagen filtrada
            - $m, n$ son las dimensiones de la máscara
            - La suma recorre todos los píxeles en la ventana
            
            ### 💡 Ejemplo Explicado
            Con una ventana $5 \\times 5$ sobre una imagen en escala de grises:
            
            1. Se toma el píxel central y sus 24 vecinos (total 25 píxeles)
            2. Se calcula el promedio: suma de todos / 25
            3. El valor central se reemplaza por este promedio
            4. Se repite para cada píxel de la imagen
            
            **Ejemplo numérico:**
            ```
            Ventana 3×3:        Promedio:
            [10  15  12]        
            [14  20  16]   →    (10+15+12+14+20+16+13+18+11)/9 = 14.3 ≈ 14
            [13  18  11]        
            ```
            
            ### ✅ Ventajas
            - ✓ **Sencillo y eficiente**: Implementación computacionalmente rápida
            - ✓ **Atenúa ruido gaussiano**: Reduce variaciones aleatorias de intensidad
            - ✓ **Preserva la media global**: Mantiene el brillo promedio de la imagen
            - ✓ **Fácil de implementar**: Algoritmo simple y directo
            
            ### ❌ Desventajas
            - ✗ **Difumina bordes y detalles finos**: Pierde definición en contornos
            - ✗ **Sensible a valores atípicos**: Un píxel muy diferente afecta el promedio
            - ✗ **Pérdida de nitidez**: La imagen resultante es más borrosa
            - ✗ **No preserva bordes**: Los límites entre regiones se suavizan
            
            ### 🔧 Implementación en OpenCV
            ```python
            import cv2
            
            # Cargar imagen
            imagen = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)
            
            # Aplicar filtro de media con kernel 5x5
            filtro_media = cv2.blur(imagen, (5, 5))
            
            # También se puede usar:
            kernel_size = (7, 7)
            filtro_media_7x7 = cv2.blur(imagen, kernel_size)
            ```
            
            ### 📊 Casos de Uso
            - Preprocesamiento para reducir ruido antes de otras operaciones
            - Suavizado general de imágenes con ruido moderado
            - Reducción de artefactos de compresión
            """)
        
        # Tab 2: Filtro de Mediana
        with tabs[2]:
            st.markdown("""
            ## Filtro de Mediana
            
            ### 📖 Definición
            Reemplaza cada píxel por la **mediana** de los valores en su vecindad, lo que elimina eficazmente
            el ruido impulsivo (sal y pimienta) sin alterar tanto los bordes.
            
            ### 📐 Fórmula Matemática
            Para una ventana $W$ con $N$ elementos ordenados $\\{p_1, p_2, \\ldots, p_N\\}$:
            """)
            
            st.latex(r"g(x,y) = \text{mediana}\{f(x+i, y+j) : (i,j) \in W\}")
            
            st.markdown("""
            Donde la mediana es:
            """)
            
            st.latex(r"\text{mediana} = \begin{cases} p_{(N+1)/2} & \text{si } N \text{ es impar} \\ \frac{p_{N/2} + p_{N/2+1}}{2} & \text{si } N \text{ es par} \end{cases}")
            
            st.markdown("""
            ### 💡 Ejemplo Explicado
            Con un kernel $3 \\times 3$:
            
            1. Se extraen los 9 valores de la ventana
            2. Se ordenan de menor a mayor
            3. Se toma el valor del medio (posición 5)
            4. Este valor reemplaza al píxel central
            
            **Ejemplo numérico:**
            ```
            Ventana 3×3:           Ordenados:         Mediana:
            [120  10  115]         [10, 15, 18,
            [ 15 250   18]    →     115, 120, 122,   →  valor[4] = 120
            [122  20  125]          125, 220, 250]
            ```
            El valor atípico 250 (ruido "sal") no afecta la mediana.
            
            ### ✅ Ventajas
            - ✓ **Excelente para ruido impulsivo**: Elimina píxeles "sal y pimienta"
            - ✓ **Mantiene bordes nítidos**: Preserva transiciones abruptas
            - ✓ **Robusto a valores atípicos**: Los extremos no afectan el resultado
            - ✓ **No lineal**: Puede preservar mejor ciertas estructuras
            
            ### ❌ Desventajas
            - ✗ **Computacionalmente costoso**: Requiere ordenar píxeles
            - ✗ **Más lento que la media**: Operación más compleja
            - ✗ **Puede distorsionar texturas**: Con ventanas grandes pierde detalles finos
            - ✗ **No es separable**: No se puede optimizar como filtros lineales
            
            ### 🔧 Implementación en OpenCV
            ```python
            import cv2
            
            # Aplicar filtro de mediana con kernel 5x5
            filtro_mediana = cv2.medianBlur(imagen, 5)
            
            # Nota: El parámetro debe ser impar (3, 5, 7, 9, etc.)
            filtro_mediana_7 = cv2.medianBlur(imagen, 7)
            ```
            
            ### 📊 Casos de Uso
            - **Eliminación de ruido sal y pimienta** (píxeles blancos/negros aleatorios)
            - Preprocesamiento de imágenes escanadas
            - Limpieza de imágenes con defectos puntuales
            - Preservación de bordes al reducir ruido
            """)
        
        # Tab 3: Filtro Logarítmico
        with tabs[3]:
            st.markdown("""
            ## Filtro Logarítmico
            
            ### 📖 Definición
            Aplica una **transformación logarítmica** punto a punto que comprime rangos dinámicos altos
            y expande bajos, realzando detalles en sombras.
            
            ### 📐 Fórmula Matemática
            Con constante de escala $c > 0$:
            """)
            
            st.latex(r"g(x,y) = c \cdot \log\big(1 + f(x,y)\big)")
            
            st.markdown("""
            Para imágenes de 8 bits, típicamente:
            """)
            
            st.latex(r"c = \frac{255}{\log(1 + f_{max})}")
            
            st.markdown("""
            Donde $f_{max}$ es el valor máximo de intensidad en la imagen original.
            
            ### 💡 Ejemplo Explicado
            Para una imagen de 8 bits con valores en $[0, 255]$:
            
            1. Normalizar a $[0, 1]$: $f_{norm} = f / 255$
            2. Aplicar logaritmo: $g = c \\cdot \\log(1 + f_{norm})$
            3. Reescalar a $[0, 255]$
            
            **Efecto:**
            - Píxeles oscuros (ej: 10) → se expanden más (ej: 85)
            - Píxeles claros (ej: 200) → se comprimen (ej: 230)
            
            **Ejemplo numérico:**
            ```
            Original:  [10,  50, 100, 200, 250]
            c = 255/log(2) ≈ 368
            Resultado: [85, 142, 175, 224, 241]
            ```
            
            ### ✅ Ventajas
            - ✓ **Realza detalles en sombras**: Mejora visibilidad en zonas oscuras
            - ✓ **Útil para HDR**: Comprime alto rango dinámico
            - ✓ **Preserva información**: No hay pérdida de datos
            - ✓ **Rápido**: Transformación punto a punto
            
            ### ❌ Desventajas
            - ✗ **Amplifica ruido en bajas intensidades**: El ruido en sombras se hace visible
            - ✗ **Puede saturar luces**: Reduce contraste en zonas brillantes
            - ✗ **Requiere normalización**: Necesita ajustar rango de salida
            - ✗ **No uniforme**: El efecto varía según la intensidad
            
            ### 🔧 Implementación en OpenCV
            ```python
            import cv2
            import numpy as np
            
            # Normalizar a [0, 1]
            imagen_norm = imagen.astype(np.float32) / 255.0
            
            # Calcular constante de escala
            c = 255.0 / np.log(1 + 1)  # o usar max de la imagen
            
            # Aplicar transformación logarítmica
            filtro_log = c * np.log(1 + imagen_norm)
            filtro_log = np.uint8(filtro_log)
            ```
            
            ### 📊 Casos de Uso
            - **Imágenes HDR**: Compresión de rango dinámico
            - Imágenes subexpuestas (muy oscuras)
            - Mejora de detalles en sombras
            - Procesamiento de imágenes médicas
            """)
        
        # Tab 4: Filtro de Cuadro Normalizado
        with tabs[4]:
            st.markdown("""
            ## Filtro de Cuadro Normalizado
            
            ### 📖 Definición
            Variante del filtro de media implementada con una **máscara uniforme** cuyos coeficientes suman 1,
            también llamado *normalized box filter*, que preserva la energía total.
            
            ### 📐 Fórmula Matemática
            Para un kernel cuadrado de lado $k$:
            """)
            
            st.latex(r"g(x,y) = \sum_{i=-r}^{r}\sum_{j=-r}^{r} \frac{1}{k^2} f(x+i, y+j), \quad r = \frac{k-1}{2}")
            
            st.markdown("""
            Donde:
            - $k$ es el tamaño del kernel (típicamente impar: 3, 5, 7, etc.)
            - $r$ es el radio: distancia del centro al borde
            - Cada coeficiente vale $1/k^2$ (garantiza que la suma sea 1)
            
            ### 💡 Ejemplo Explicado
            Con $k = 7$ (kernel $7 \\times 7$):
            
            - Número de píxeles: $7^2 = 49$
            - Cada coeficiente: $1/49 \\approx 0.0204$
            - La suma de todos los coeficientes = 1
            
            **Matriz del kernel 3×3:**
            ```
            [1/9  1/9  1/9]
            [1/9  1/9  1/9]
            [1/9  1/9  1/9]
            ```
            
            ### ✅ Ventajas
            - ✓ **Suavizado controlado**: Magnitud predecible del efecto
            - ✓ **Implementación optimizada**: OpenCV usa integrales para velocidad
            - ✓ **Preserva brillo**: La suma de pesos es 1
            - ✓ **Separable**: Se puede descomponer en 1D para eficiencia
            
            ### ❌ Desventajas
            - ✗ **Similar a filtro de media**: Mismas limitaciones de difuminado
            - ✗ **Difumina bordes**: Pierde definición en contornos
            - ✗ **Artefactos de bloque**: Aplicación iterativa puede crear patrones
            - ✗ **No adaptat ivo**: Mismo peso para todos los píxeles
            
            ### 🔧 Implementación en OpenCV
            ```python
            import cv2
            
            # Aplicar filtro de cuadro normalizado
            filtro_cuadro = cv2.boxFilter(imagen, -1, (7, 7), normalize=True)
            
            # Parámetros:
            # -1: mismo tipo que la imagen de entrada
            # (7, 7): tamaño del kernel
            # normalize=True: normalizar (dividir por área del kernel)
            
            # Sin normalizar (suma simple):
            filtro_sin_norm = cv2.boxFilter(imagen, -1, (5, 5), normalize=False)
            ```
            
            ### 📊 Casos de Uso
            - Suavizado rápido de imágenes
            - Preprocesamiento eficiente
            - Reducción de ruido cuando velocidad es crítica
            - Base para algoritmos más complejos (como blur gaussiano aproximado)
            """)
        
        # Tab 5: Filtro Gaussiano
        with tabs[5]:
            st.markdown("""
            ## Filtro Gaussiano
            
            ### 📖 Definición
            Suavizado lineal mediante la **convolución con una función gaussiana**, que reduce ruido
            preservando mejor los bordes que un filtro uniforme.
            
            ### 📐 Fórmula Matemática
            Con desviación estándar $\\sigma$:
            """)
            
            st.latex(r"G(i,j) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{i^2 + j^2}{2\sigma^2}\right)")
            
            st.markdown("""
            La imagen filtrada es:
            """)
            
            st.latex(r"g(x,y) = \sum_{i}\sum_{j} G(i,j) \cdot f(x+i, y+j)")
            
            st.markdown("""
            **Propiedades clave:**
            - El kernel es simétrico y radial
            - Los píxeles cercanos al centro tienen mayor peso
            - La suma de todos los coeficientes es 1 (normalizado)
            - $\\sigma$ controla el "ancho" de la campana
            
            ### 💡 Ejemplo Explicado
            Con $\\sigma = 1.0$ y máscara $5 \\times 5$:
            
            **Kernel gaussiano aproximado:**
            ```
            [1   4   7   4   1]
            [4  16  26  16   4]
            [7  26  41  26   7]  × 1/273
            [4  16  26  16   4]
            [1   4   7   4   1]
            ```
            
            El píxel central (41/273 ≈ 15%) tiene el mayor peso.
            
            ### ✅ Ventajas
            - ✓ **Atenúa ruido gaussiano eficientemente**: Óptimo para este tipo de ruido
            - ✓ **Separable en 1D**: Se puede aplicar horizontal y verticalmente (más rápido)
            - ✓ **Conserva bordes suaves**: Menos difuminado que media
            - ✓ **Control con** $\\sigma$: Fácil ajustar el nivel de suavizado
            - ✓ **Base teórica sólida**: Derivado de procesos estocásticos
            
            ### ❌ Desventajas
            - ✗ **Difumina detalles muy finos**: Aún suaviza la imagen
            - ✗ **Requiere elegir** $\\sigma$ **apropiado**: Parámetro crítico
            - ✗ **Más costoso que media**: Cálculo de exponenciales
            - ✗ **Tamaño de kernel variable**: Debe ser suficientemente grande para $\\sigma$
            
            ### 🔧 Implementación en OpenCV
            ```python
            import cv2
            
            # Aplicar filtro gaussiano
            filtro_gaussiano = cv2.GaussianBlur(imagen, (5, 5), 1.0)
            
            # Parámetros:
            # (5, 5): tamaño del kernel (debe ser impar)
            # 1.0: sigma (desviación estándar)
            
            # Si sigma = 0, se calcula automáticamente desde el tamaño
            filtro_auto = cv2.GaussianBlur(imagen, (5, 5), 0)
            
            # Mayor sigma = más suavizado
            filtro_suave = cv2.GaussianBlur(imagen, (9, 9), 2.0)
            ```
            
            ### 📊 Casos de Uso
            - **Reducción de ruido gaussiano** (el más común)
            - Preprocesamiento antes de detección de bordes (ej: Canny)
            - Suavizado de imágenes para visualización
            - Creación de pirámides gaussianas
            - Base para algoritmos de visión (SIFT, SURF, etc.)
            """)
        
        # Tab 6: Filtro Laplace
        with tabs[6]:
            st.markdown("""
            ## Filtro Laplace
            
            ### 📖 Definición
            Operador **derivativo de segundo orden** que responde a cambios bruscos de intensidad,
            útil para realzar bordes y detectar zonas de variación rápida.
            
            ### 📐 Fórmula Matemática
            Aproximación discreta del laplaciano:
            """)
            
            st.latex(r"\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}")
            
            st.markdown("""
            Aproximación en diferencias finitas:
            """)
            
            st.latex(r"g(x,y) \approx -4f(x,y) + f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1)")
            
            st.markdown("""
            **Kernels comunes:**
            
            Kernel básico (4-vecinos):
            ```
            [ 0   1   0]
            [ 1  -4   1]
            [ 0   1   0]
            ```
            
            Kernel extendido (8-vecinos):
            ```
            [ 1   1   1]
            [ 1  -8   1]
            [ 1   1   1]
            ```
            
            ### 💡 Ejemplo Explicado
            El laplaciano mide la **curvatura** de la intensidad:
            
            - Región plana → Laplaciano ≈ 0
            - Borde → Laplaciano alto (positivo o negativo)
            - Esquina → Laplaciano muy alto
            
            **Proceso:**
            1. Multiplicar imagen por el kernel
            2. El resultado tiene valores positivos y negativos
            3. Tomar valor absoluto para visualizar
            4. Opcional: sumar a imagen original para realzar
            
            ### ✅ Ventajas
            - ✓ **Detecta bordes en todas direcciones**: Isotrópico (sin sesgo direccional)
            - ✓ **Sencillo de implementar**: Kernel pequeño y simple
            - ✓ **Realza detalles finos**: Enfatiza cambios rápidos
            - ✓ **Una sola operación**: No requiere combinar gradientes
            
            ### ❌ Desventajas
            - ✗ **Muy sensible al ruido**: Derivada de segundo orden amplifica ruido
            - ✗ **Requiere suavizado previo**: Típicamente se usa Gaussiano antes
            - ✗ **No directamente visualizable**: Necesita reescalado y valor absoluto
            - ✗ **Bordes dobles**: Genera líneas dobles en transiciones
            
            ### 🔧 Implementación en OpenCV
            ```python
            import cv2
            import numpy as np
            
            # Aplicar filtro Laplaciano
            laplacian = cv2.Laplacian(imagen, cv2.CV_64F)
            
            # Convertir a formato visualizable
            laplacian_abs = np.uint8(np.absolute(laplacian))
            
            # Con suavizado gaussiano previo (recomendado)
            blur = cv2.GaussianBlur(imagen, (3, 3), 0)
            laplacian_suave = cv2.Laplacian(blur, cv2.CV_64F)
            
            # Realce de bordes (Laplacian sharpening)
            sharpened = imagen - laplacian_abs
            ```
            
            ### 📊 Casos de Uso
            - **Realce de bordes** (image sharpening)
            - Detección de contornos en imágenes médicas
            - Análisis de texturas
            - Detección de puntos de interés (esquinas, blobs)
            """)
        
        # Tab 7: Filtro Sobel
        with tabs[7]:
            st.markdown("""
            ## Filtro Sobel
            
            ### 📖 Definición
            Operador de **derivada de primer orden** con máscaras separables que calculan gradientes
            horizontales y verticales, suavizando ligeramente para reducir ruido.
            
            ### 📐 Fórmula Matemática
            Kernels clásicos de Sobel:
            """)
            
            st.latex(r"G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}")
            
            st.markdown("""
            El módulo del gradiente (magnitud):
            """)
            
            st.latex(r"|\nabla f| = \sqrt{G_x^2 + G_y^2}")
            
            st.markdown("""
            La dirección del gradiente:
            """)
            
            st.latex(r"\theta = \arctan\left(\frac{G_y}{G_x}\right)")
            
            st.markdown("""
            ### 💡 Ejemplo Explicado
            
            **Proceso completo:**
            1. Aplicar kernel $G_x$ → detecta bordes verticales
            2. Aplicar kernel $G_y$ → detecta bordes horizontales
            3. Combinar: $\\sqrt{G_x^2 + G_y^2}$ → magnitud del gradiente
            4. Opcional: calcular dirección con arctan
            
            **Interpretación:**
            - $G_x$ alto → cambio horizontal (borde vertical)
            - $G_y$ alto → cambio vertical (borde horizontal)
            - Magnitud alta → borde fuerte (cualquier dirección)
            
            **Aproximación rápida (menos precisa pero más rápida):**
            """)
            
            st.latex(r"|\nabla f| \approx |G_x| + |G_y|")
            
            st.markdown("""
            ### ✅ Ventajas
            - ✓ **Detecta bordes con reducción de ruido**: Suavizado incorporado
            - ✓ **Computacionalmente eficiente**: Kernels pequeños (3×3)
            - ✓ **Proporciona dirección del gradiente**: Útil para análisis direccional
            - ✓ **Separable**: Se puede optimizar la implementación
            - ✓ **Bien establecido**: Estándar en visión por computadora
            
            ### ❌ Desventajas
            - ✗ **Sensible a ruido fuerte**: Aunque mejor que derivadas simples
            - ✗ **Requiere umbralización**: El resultado es escala de grises, no binario
            - ✗ **Bordes gruesos**: No tan precisos como Canny
            - ✗ **Sesgo diagonal**: Los bordes diagonales tienen magnitud √2 veces mayor
            
            ### 🔧 Implementación en OpenCV
            ```python
            import cv2
            import numpy as np
            
            # Calcular gradientes en X e Y
            sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calcular magnitud del gradiente
            magnitud = np.sqrt(sobel_x**2 + sobel_y**2)
            magnitud = np.uint8(magnitud)
            
            # Aproximación rápida
            magnitud_aprox = np.uint8(np.abs(sobel_x) + np.abs(sobel_y))
            
            # Calcular dirección
            direccion = np.arctan2(sobel_y, sobel_x)
            
            # Kernels más grandes (mayor suavizado)
            sobel_x_5 = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=5)
            ```
            
            ### 📊 Casos de Uso
            - **Detección de bordes** en imágenes generales
            - Preprocesamiento para segmentación
            - Análisis de orientación de texturas
            - Base para algoritmos más complejos (HOG, descriptores)
            - Cálculo de gradientes para optimización
            """)
        
        # Tab 8: Filtro Canny
        with tabs[8]:
            st.markdown("""
            ## Filtro Canny
            
            ### 📖 Definición
            Detector de bordes **multi-etapa** que optimiza la relación señal/ruido. Considerado el
            detector de bordes óptimo según criterios de buena detección, buena localización y
            respuesta única.
            
            ### 📐 Algoritmo (Pasos Matemáticos)
            
            **1. Suavizado Gaussiano**
            """)
            
            st.latex(r"I_{smooth} = G_\sigma * I")
            
            st.markdown("""
            **2. Cálculo del Gradiente (Sobel)**
            """)
            
            st.latex(r"G_x = \frac{\partial I}{\partial x}, \quad G_y = \frac{\partial I}{\partial y}")
            st.latex(r"|\nabla I| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan(G_y / G_x)")
            
            st.markdown("""
            **3. Supresión No-Máxima**
            
            Para cada píxel, verificar si es máximo local en dirección del gradiente:
            - Si $|\\nabla I(x,y)| < |\\nabla I(x', y')|$ en dirección $\\theta$ → suprimir
            - Si es máximo local → mantener
            
            **4. Umbralización con Histéresis**
            - $T_{high}$: umbral alto (bordes fuertes)
            - $T_{low}$: umbral bajo (bordes débiles)
            - Bordes fuertes: $|\\nabla I| > T_{high}$ → aceptar
            - Bordes débiles: $T_{low} < |\\nabla I| < T_{high}$ → aceptar si conectados a fuertes
            - Píxeles débiles: $|\\nabla I| < T_{low}$ → rechazar
            
            ### 💡 Ejemplo Explicado
            
            **Proceso completo con parámetros típicos:**
            
            1. **Suavizado**: $\\sigma = 1.4$, kernel 5×5
            2. **Gradiente**: Sobel 3×3
            3. **Supresión**: Adelgazar bordes a 1 píxel
            4. **Histéresis**: $T_{low} = 50$, $T_{high} = 150$
            
            **¿Por qué funciona tan bien?**
            - Suavizado → reduce ruido
            - Supresión no-máxima → bordes delgados (1 píxel)
            - Histéresis → conecta bordes débiles, evita fragmentación
            
            **Relación de umbrales:**
            Típicamente: $T_{high} = 2 \\times T_{low}$ a $3 \\times T_{low}$
            
            ### ✅ Ventajas
            - ✓ **Mejor detector de bordes**: Óptimo según criterios de Canny
            - ✓ **Bordes delgados y continuos**: Supresión no-máxima + histéresis
            - ✓ **Control sobre sensibilidad**: Dos umbrales ajustables
            - ✓ **Reduce falsos positivos**: Histéresis elimina ruido
            - ✓ **Bien localizado**: Bordes en posición precisa
            
            ### ❌ Desventajas
            - ✗ **Sensible a parámetros**: Requiere ajustar $T_{low}$, $T_{high}$, $\\sigma$
            - ✗ **Más costoso computacionalmente**: Múltiples etapas
            - ✗ **No da dirección directamente**: Solo bordes binarios
            - ✗ **Puede fallar con ruido extremo**: A pesar del suavizado
            
            ### 🔧 Implementación en OpenCV
            ```python
            import cv2
            
            # Aplicar detector de bordes Canny
            bordes = cv2.Canny(imagen, threshold1=50, threshold2=150)
            
            # Parámetros:
            # threshold1: umbral bajo (T_low)
            # threshold2: umbral alto (T_high)
            
            # Con suavizado explícito (opcional, Canny ya suaviza)
            blur = cv2.GaussianBlur(imagen, (5, 5), 1.4)
            bordes_blur = cv2.Canny(blur, 50, 150)
            
            # Ajuste fino de parámetros:
            # - Más sensible (detecta más bordes)
            bordes_sensible = cv2.Canny(imagen, 30, 90)
            
            # - Menos sensible (solo bordes fuertes)
            bordes_fuerte = cv2.Canny(imagen, 100, 200)
            
            # Con tamaño de Sobel personalizado
            bordes_sobel5 = cv2.Canny(imagen, 50, 150, apertureSize=5)
            
            # Con gradiente L2 (más preciso pero lento)
            bordes_l2 = cv2.Canny(imagen, 50, 150, L2gradient=True)
            ```
            
            ### 📊 Casos de Uso
            - **Detección de bordes de alta calidad** en cualquier tipo de imagen
            - Segmentación de objetos
            - Reconocimiento de formas y contornos
            - Visión robótica y navegación
            - Procesamiento de imágenes médicas
            - Detección de líneas (con Hough Transform)
            - Preprocesamiento para OCR
            
            ### 🔬 Criterios de Optimalidad de Canny
            
            1. **Buena detección**: Baja tasa de falsos positivos y negativos
            2. **Buena localización**: Bordes cerca de su posición real
            3. **Respuesta única**: Un solo detector por borde real
            
            ### 📈 Comparación de Umbrales
            
            | $T_{low}$ | $T_{high}$ | Efecto |
            |-----------|------------|--------|
            | 30 | 90 | Más bordes, más ruido |
            | 50 | 150 | **Balanceado (recomendado)** |
            | 100 | 200 | Solo bordes muy fuertes |
            | Auto | Auto×2.5 | Basado en mediana de gradientes |
            """)
    
    # ========================================================================
    # MODO 1: FILTROS
    # ========================================================================
    
    elif modo == "🔍 Filtros (Parte 1)":
        st.header("Parte 1: Aplicación de Filtros")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Cargar Imagen")
            uploaded_file = st.file_uploader(
                "Selecciona una imagen",
                type=['jpg', 'jpeg', 'png', 'bmp']
            )
            
            image_gray = None
            
            if uploaded_file is not None:
                image_pil = load_image(uploaded_file)
                image_np = np.array(image_pil)
                image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                
                st.image(image_pil, caption='Imagen Original', use_column_width=True)
        
        with col2:
            if uploaded_file is not None:
                st.subheader("🎨 Seleccionar Filtro")
                
                filter_type = st.selectbox(
                    "Tipo de filtro:",
                    ["Media", "Mediana", "Logarítmico", "Cuadro Normalizado",
                     "Gaussiano", "Laplace", "Sobel", "Canny"]
                )
                
                # Parámetros específicos por filtro
                params = {}
                
                if filter_type in ["Media", "Mediana", "Cuadro Normalizado"]:
                    params['kernel_size'] = st.slider(
                        "Tamaño del kernel",
                        3, 15, 5, step=2
                    )
                
                elif filter_type == "Gaussiano":
                    params['kernel_size'] = st.slider(
                        "Tamaño del kernel",
                        3, 15, 5, step=2
                    )
                    params['sigma'] = st.slider(
                        "Sigma (σ)",
                        0.1, 5.0, 1.0, step=0.1
                    )
                
                elif filter_type == "Canny":
                    params['threshold1'] = st.slider(
                        "Umbral bajo",
                        0, 200, 50
                    )
                    params['threshold2'] = st.slider(
                        "Umbral alto",
                        0, 300, 150
                    )
                
                if st.button("🔄 Aplicar Filtro", type="primary"):
                    if image_gray is not None:
                        with st.spinner("Aplicando filtro..."):
                            filtered_image = apply_filter(image_gray, filter_type, **params)
                            
                            st.image(filtered_image, caption=f'Filtro: {filter_type}',  # type: ignore
                                    use_column_width=True, clamp=True)
                            
                            # Información del filtro
                            st.info(f"""
                            **Filtro aplicado:** {filter_type}
                            **Parámetros:** {params if params else 'Valores por defecto'}
                            **Dimensiones:** {filtered_image.shape}
                            """)
                            
                            # Opción de descarga
                            is_success, buffer = cv2.imencode(".png", filtered_image)  # type: ignore
                            if is_success:
                                st.download_button(
                                    label="💾 Descargar imagen filtrada",
                                    data=buffer.tobytes(),
                                    file_name=f"filtro_{filter_type.lower()}.png",
                                    mime="image/png"
                                )
                    else:
                        st.error("❌ Por favor, carga una imagen primero")
    
    # ========================================================================
    # MODO 2: ENTRENAMIENTO
    # ========================================================================
    
    elif modo == "🤖 Descriptores y Clasificación (Parte 2)":
        st.header("Parte 2: Entrenamiento de Modelos")
        
        if st.button("🚀 Iniciar Entrenamiento", type="primary"):
            data_root = Path(data_root_input)
            
            if not data_root.exists():
                st.error(f"❌ La ruta {data_root} no existe")
                return
            
            results = {}
            class_names = []
            
            # Entrenar CNN
            if train_cnn:
                st.subheader("🧠 Entrenando CNN...")
                try:
                    cnn_model, class_names, eval_transform, history_cnn = train_cnn_model(
                        data_root, epochs_cnn, learning_rate, batch_size, weight_decay
                    )
                    results['cnn'] = {
                        'model': cnn_model,
                        'history': history_cnn
                    }
                    st.success("✅ CNN entrenada exitosamente!")
                    
                    # Visualizar curvas
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.plot(history_cnn['train_loss'], label='Train')
                    ax1.plot(history_cnn['val_loss'], label='Val')
                    ax1.set_title('Loss')
                    ax1.legend()
                    ax1.grid(True)
                    
                    ax2.plot(history_cnn['train_acc'], label='Train')
                    ax2.plot(history_cnn['val_acc'], label='Val')
                    ax2.set_title('Accuracy')
                    ax2.legend()
                    ax2.grid(True)
                    
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error entrenando CNN: {e}")
            
            # Entrenar SVM+HOG
            if train_svm_hog:
                st.subheader("📊 Entrenando SVM + HOG...")
                try:
                    import time
                    from sklearn.model_selection import train_test_split
                    
                    # Cargar datos
                    train_ds = datasets.ImageFolder(data_root / 'train')
                    total_images = len(train_ds)
                    
                    st.info(f"📦 Extrayendo características HOG de {total_images} imágenes...")
                    
                    # Extraer características
                    x_hog = []  # Renombrado para seguir convenciones
                    y_labels = []
                    
                    progress = st.progress(0)
                    status = st.empty()
                    time_status = st.empty()
                    
                    start_time = time.time()
                    
                    for idx, (img, label) in enumerate(train_ds):  # type: ignore
                        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                        img_resized = cv2.resize(img_gray, (128, 64))
                        
                        hog_feat = extract_hog_features(
                            img_resized,
                            {
                                'orientations': 9,
                                'pixels_per_cell': (8, 8),
                                'cells_per_block': (2, 2),
                                'visualize': False,
                                'feature_vector': True
                            }
                        )
                        x_hog.append(hog_feat)
                        y_labels.append(label)
                        
                        # Actualizar progreso cada 10 imágenes para mejor feedback
                        if idx % 10 == 0 or idx == total_images - 1:
                            progress_pct = (idx + 1) / total_images
                            progress.progress(progress_pct)
                            
                            # Calcular tiempo estimado
                            elapsed = time.time() - start_time
                            if idx > 0:
                                avg_time_per_img = elapsed / (idx + 1)
                                remaining_imgs = total_images - (idx + 1)
                                eta_seconds = avg_time_per_img * remaining_imgs
                                eta_minutes = eta_seconds / 60
                                
                                status.text(f"🔄 Procesando: {idx + 1}/{total_images} imágenes ({progress_pct*100:.1f}%)")
                                time_status.text(f"⏱️ Tiempo estimado restante: {eta_minutes:.1f} min")
                            else:
                                status.text(f"🔄 Procesando: {idx + 1}/{total_images} imágenes")
                    
                    progress.empty()
                    status.empty()
                    time_status.empty()
                    
                    extraction_time = time.time() - start_time
                    st.success(f"✅ Características HOG extraídas en {extraction_time:.1f}s")
                    
                    x_hog = np.array(x_hog)
                    y_labels = np.array(y_labels)
                    
                    st.info(f"📊 Matriz de características: {x_hog.shape}")
                    
                    # Entrenar SVM
                    svm_hog = make_pipeline(
                        StandardScaler(),
                        LinearSVC(max_iter=5000, random_state=42, dual=True),
                        memory=None
                    )
                    
                    with st.spinner("Entrenando SVM..."):
                        svm_hog.fit(x_hog, y_labels)
                    
                    results['svm_hog'] = svm_hog
                    st.success("✅ SVM+HOG entrenado exitosamente!")
                    
                except Exception as e:
                    st.error(f"❌ Error entrenando SVM+HOG: {e}")
            
            # Entrenar SVM+LBP
            if train_svm_lbp:
                st.subheader("🔬 Entrenando SVM + LBP...")
                try:
                    import time
                    
                    train_ds = datasets.ImageFolder(data_root / 'train')
                    total_images = len(train_ds)
                    
                    st.info(f"📦 Extrayendo características LBP de {total_images} imágenes...")
                    
                    x_lbp = []  # Renombrado para seguir convenciones
                    y_labels = []
                    
                    progress = st.progress(0)
                    status = st.empty()
                    time_status = st.empty()
                    
                    start_time = time.time()
                    
                    for idx, (img, label) in enumerate(train_ds):  # type: ignore
                        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                        img_resized = cv2.resize(img_gray, (128, 64))
                        
                        lbp_feat = extract_lbp_features(img_resized, 3, 24)
                        x_lbp.append(lbp_feat)
                        y_labels.append(label)
                        
                        # Actualizar progreso cada 10 imágenes
                        if idx % 10 == 0 or idx == total_images - 1:
                            progress_pct = (idx + 1) / total_images
                            progress.progress(progress_pct)
                            
                            # Calcular tiempo estimado
                            elapsed = time.time() - start_time
                            if idx > 0:
                                avg_time_per_img = elapsed / (idx + 1)
                                remaining_imgs = total_images - (idx + 1)
                                eta_seconds = avg_time_per_img * remaining_imgs
                                eta_minutes = eta_seconds / 60
                                
                                status.text(f"🔄 Procesando: {idx + 1}/{total_images} imágenes ({progress_pct*100:.1f}%)")
                                time_status.text(f"⏱️ Tiempo estimado restante: {eta_minutes:.1f} min")
                            else:
                                status.text(f"🔄 Procesando: {idx + 1}/{total_images} imágenes")
                    
                    progress.empty()
                    status.empty()
                    time_status.empty()
                    
                    extraction_time = time.time() - start_time
                    st.success(f"✅ Características LBP extraídas en {extraction_time:.1f}s")
                    
                    x_lbp = np.array(x_lbp)
                    y_labels = np.array(y_labels)
                    
                    st.info(f"📊 Matriz de características: {x_lbp.shape}")
                    
                    svm_lbp = make_pipeline(
                        StandardScaler(),
                        LinearSVC(max_iter=5000, random_state=42, dual=True),
                        memory=None
                    )
                    
                    with st.spinner("Entrenando SVM..."):
                        svm_lbp.fit(x_lbp, y_labels)
                    
                    results['svm_lbp'] = svm_lbp
                    st.success("✅ SVM+LBP entrenado exitosamente!")
                    
                except Exception as e:
                    st.error(f"❌ Error entrenando SVM+LBP: {e}")
            
            # Guardar modelos
            if results:
                st.divider()
                st.subheader("💾 Guardando modelos...")
                
                config = {
                    'hog_params': {
                        'orientations': 9,
                        'pixels_per_cell': (8, 8),
                        'cells_per_block': (2, 2),
                        'visualize': False,
                        'feature_vector': True
                    },
                    'lbp_radius': 3,
                    'lbp_n_points': 24,
                    'patch_size': (128, 64)
                }
                
                save_models(
                    results.get('cnn', {}).get('model'),
                    results.get('svm_hog'),
                    results.get('svm_lbp'),
                    class_names if 'cnn' in results else [],
                    config
                )
                
                st.success("✅ Todos los modelos guardados en la carpeta 'models/'")
    
    # ========================================================================
    # MODO 3: CLASIFICACIÓN
    # ========================================================================
    
    elif modo == "🎯 Clasificar Imagen":
        st.header("Clasificación de Nuevas Imágenes")
        
        # Cargar modelos
        models, class_names, config, eval_transform = load_models()
        
        if models is None:
            st.warning("⚠️ No se encontraron modelos entrenados. Entrena primero en la Parte 2.")
            return
        
        st.success(f"✅ Modelos cargados: {', '.join(models.keys())}")
        
        # Cargar imagen
        uploaded_file = st.file_uploader(
            "Selecciona una imagen para clasificar",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            image_pil = load_image(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image_pil, caption='Imagen cargada', use_column_width=True)
            
            with col2:
                # Seleccionar modelo
                available_models = list(models.keys())
                model_display_names = {
                    'cnn': 'CNN (PyTorch)',
                    'svm_hog': 'SVM + HOG',
                    'svm_lbp': 'SVM + LBP'
                }
                
                model_name = st.selectbox(
                    "Selecciona el modelo:",
                    available_models,
                    format_func=lambda x: model_display_names.get(x, "Desconocido")
                )
                
                if st.button("🔍 Clasificar", type="primary"):
                    with st.spinner("Clasificando..."):
                        # Verificar que tenemos los recursos necesarios
                        if eval_transform is None or class_names is None or config is None:
                            st.error("❌ Error: Los modelos no se cargaron correctamente")
                            return
                        
                        image_np = np.array(image_pil)
                        pred_class = "Unknown"
                        confidence = 0.0
                        probs = np.array([])
                        
                        if model_name == 'cnn':
                            # CNN
                            # type: ignore - eval_transform retorna un tensor con unsqueeze
                            tensor = eval_transform(image_pil).unsqueeze(0).to(DEVICE)  # type: ignore
                            model = models['cnn']
                            model.eval()
                            
                            with torch.no_grad():
                                logits = model(tensor)
                                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                            
                            pred_idx = int(np.argmax(probs))
                            pred_class = class_names[pred_idx]
                            confidence = float(probs[pred_idx])
                            
                        elif model_name == 'svm_hog':
                            # SVM + HOG
                            image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                            image_resized = cv2.resize(image_gray, tuple(config['patch_size']))
                            
                            hog_feat = extract_hog_features(
                                image_resized,
                                config['hog_params']
                            ).reshape(1, -1)
                            
                            pred_idx = int(models['svm_hog'].predict(hog_feat)[0])
                            pred_class = class_names[pred_idx]
                            
                            # SVM no da probabilidades directamente
                            decision = models['svm_hog'].decision_function(hog_feat)[0]
                            probs = np.exp(decision) / np.sum(np.exp(decision))
                            confidence = float(probs[pred_idx]) if len(probs) > pred_idx else 0.95
                            
                        elif model_name == 'svm_lbp':
                            # SVM + LBP
                            image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                            image_resized = cv2.resize(image_gray, tuple(config['patch_size']))
                            
                            lbp_feat = extract_lbp_features(
                                image_resized,
                                config['lbp_radius'],
                                config['lbp_n_points']
                            ).reshape(1, -1)
                            
                            pred_idx = int(models['svm_lbp'].predict(lbp_feat)[0])
                            pred_class = class_names[pred_idx]
                            
                            decision = models['svm_lbp'].decision_function(lbp_feat)[0]
                            probs = np.exp(decision) / np.sum(np.exp(decision))
                            confidence = float(probs[pred_idx]) if len(probs) > pred_idx else 0.95
                        
                        # Mostrar resultados
                        st.success(f"### 🎯 Predicción: **{pred_class}**")
                        st.metric("Confianza", f"{confidence:.2%}")
                        
                        # Mostrar todas las probabilidades (solo para CNN)
                        if model_name == 'cnn' and len(probs) > 0 and class_names is not None:
                            st.divider()
                            st.subheader("📊 Distribución de Probabilidades")
                            
                            # Top 5 predicciones
                            top_k = min(5, len(probs))
                            top_indices = np.argsort(probs)[-top_k:][::-1]
                            
                            prob_data = {
                                'Clase': [str(class_names[i]) for i in top_indices],
                                'Probabilidad': [float(probs[i]) for i in top_indices]
                            }
                            
                            df_probs = pd.DataFrame(prob_data)
                            st.dataframe(df_probs, hide_index=True, use_container_width=True)
                            
                            # Gráfico de barras
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.barh(df_probs['Clase'], df_probs['Probabilidad'])
                            ax.set_xlabel('Probabilidad')
                            ax.set_title('Top 5 Predicciones')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            plt.close(fig)

if __name__ == "__main__":
    main()
