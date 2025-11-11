# 📊 INFORME TÉCNICO - TAREA 2
## Procesamiento Digital de Imágenes: Filtros y Clasificación

---

### 📋 INFORMACIÓN DEL PROYECTO

**Estudiante:** Emmanuel Bustamante  
**Institución:** Universidad de Antioquia  
**Curso:** Procesamiento Digital de Imágenes  
**Tarea:** Tarea 2 - Filtros y Descriptores de Características  
**Fecha:** Noviembre 2025  
**Plataforma:** [Streamlit Cloud](https://clasificaci-n-de-imagenes-6je33ygv8xeoekkn2dl8qd.streamlit.app/)  
**Repositorio:** [GitHub - Emma-Ok/Clasificaci-n-de-imagenes](https://github.com/Emma-Ok/Clasificaci-n-de-imagenes)

---

## 📌 RESUMEN EJECUTIVO

Este proyecto implementa una aplicación web interactiva para procesamiento de imágenes que abarca dos componentes principales:

1. **PARTE 1 (30%)**: Sistema de aplicación de filtros digitales con visualización en tiempo real
2. **PARTE 2 (70%)**: Sistema completo de clasificación de caracteres alfanuméricos usando descriptores de características y aprendizaje automático

La aplicación fue desarrollada en Python usando Streamlit como framework de interfaz gráfica, OpenCV para procesamiento de imágenes, scikit-learn para modelos tradicionales de ML, y PyTorch para redes neuronales convolucionales.

**Características principales:**
- Interfaz web interactiva accesible desde cualquier navegador
- 8 filtros digitales implementados con parámetros ajustables
- 3 modelos de clasificación entrenables (SVM+HOG, SVM+LBP, CNN)
- Dataset de 1,080 imágenes (36 clases: 0-9, A-Z)
- Métricas de evaluación completas (accuracy, precision, recall, F1, matriz de confusión)
- Despliegue en la nube con Streamlit Cloud

---

## 🎯 OBJETIVOS

### Objetivos Generales
1. Implementar y analizar filtros de procesamiento de imágenes en el dominio espacial
2. Desarrollar un sistema de clasificación automática de caracteres usando descriptores de características
3. Comparar el desempeño de métodos tradicionales (SVM) vs. aprendizaje profundo (CNN)

### Objetivos Específicos
- Programar 8 filtros digitales con parámetros configurables
- Extraer descriptores HOG y LBP de imágenes
- Entrenar clasificadores SVM con diferentes descriptores
- Implementar y entrenar una red neuronal convolucional
- Evaluar y comparar los modelos mediante métricas estándar
- Crear una interfaz gráfica intuitiva para demostración

---

## 🛠️ METODOLOGÍA

### 1. Arquitectura del Sistema

El proyecto está estructurado en cuatro módulos principales:

```
app_streamlit_completa.py (1,580 líneas)
├── Configuración global y constantes
├── Módulo de filtros (PARTE 1 - 30%)
│   ├── apply_filter(): Aplica filtros con parámetros
│   └── 8 filtros implementados
├── Módulo de descriptores (PARTE 2 - 70%)
│   ├── extract_hog_features(): Extracción HOG
│   └── extract_lbp_features(): Extracción LBP
├── Módulo de modelos
│   ├── PlateCNN: Arquitectura CNN (PyTorch)
│   ├── train_cnn_model(): Entrenamiento CNN
│   ├── train_svm_hog(): Entrenamiento SVM+HOG
│   └── train_svm_lbp(): Entrenamiento SVM+LBP
└── Interfaz de usuario (Streamlit)
    ├── Modo Teoría de Filtros
    ├── Modo Filtros Parte 1
    ├── Modo Descriptores y Clasificación Parte 2
    └── Modo Clasificar Imagen
```

### 2. Dataset

**Estructura:**
- **Total:** 1,080 imágenes
- **Clases:** 36 (dígitos 0-9 + letras A-Z)
- **Resolución:** 128×64 píxeles
- **División:**
  - Entrenamiento: 864 imágenes (80%)
  - Validación: 216 imágenes (20%)
- **Formato:** RGB, normalizado a escala de grises para descriptores

**Organización de carpetas:**
```
data/
├── train/
│   ├── class_0/ (24 imágenes)
│   ├── class_1/ (24 imágenes)
│   ├── ...
│   └── class_Z/ (24 imágenes)
└── val/
    ├── class_0/ (6 imágenes)
    ├── class_1/ (6 imágenes)
    ├── ...
    └── class_Z/ (6 imágenes)
```

---

## 📐 PARTE 1: FILTROS DE IMÁGENES (30%)

### 1.1 Filtros Implementados

Se implementaron 8 filtros digitales en el dominio espacial:

#### 🔹 1. Filtro de Media
- **Propósito:** Suavizado mediante promedio aritmético
- **Fórmula:** `g(x,y) = (1/mn) × Σ Σ f(x+i, y+j)`
- **Parámetro ajustable:** Tamaño de kernel (3×3, 5×5, 7×7, 9×9)
- **Aplicación:** Reducción de ruido gaussiano
- **Ventaja:** Simple y eficiente computacionalmente
- **Desventaja:** Difumina bordes

#### 🔹 2. Filtro de Mediana
- **Propósito:** Suavizado mediante valor mediano
- **Fórmula:** `g(x,y) = mediana{f(x+i, y+j)}`
- **Parámetro ajustable:** Tamaño de kernel (3×3, 5×5, 7×7)
- **Aplicación:** Excelente para ruido sal y pimienta
- **Ventaja:** Preserva bordes mejor que la media
- **Desventaja:** Mayor costo computacional

#### 🔹 3. Filtro Logarítmico
- **Propósito:** Compresión de rango dinámico
- **Fórmula:** `g(x,y) = c × log(1 + f(x,y))`
- **Parámetro ajustable:** Factor de escala c (1-100)
- **Aplicación:** Realce de detalles en zonas oscuras
- **Ventaja:** Mejora visualización de imágenes HDR
- **Desventaja:** Puede sobreexponer zonas claras

#### 🔹 4. Filtro Cuadro Normalizado
- **Propósito:** Suavizado uniforme con normalización
- **Implementación:** `cv2.boxFilter()` con `normalize=True`
- **Parámetro ajustable:** Tamaño de kernel (3×3, 5×5, 7×7, 9×9)
- **Aplicación:** Reducción de ruido con efecto de blur uniforme
- **Ventaja:** Rápido y predecible
- **Desventaja:** Pérdida de detalles finos

#### 🔹 5. Filtro Gaussiano
- **Propósito:** Suavizado ponderado gaussianamente
- **Fórmula:** `G(x,y) = (1/2πσ²) × exp(-(x²+y²)/2σ²)`
- **Parámetros ajustables:** 
  - Tamaño de kernel (3×3, 5×5, 7×7, 9×9)
  - Sigma σ (0.5 - 5.0)
- **Aplicación:** Preprocesamiento para detección de bordes
- **Ventaja:** Suavizado natural, preserva mejor los bordes que la media
- **Desventaja:** Más costoso que filtro de media

#### 🔹 6. Filtro Laplaciano
- **Propósito:** Detección de bordes mediante segunda derivada
- **Fórmula:** `∇²f = ∂²f/∂x² + ∂²f/∂y²`
- **Parámetro ajustable:** Tamaño de kernel (1, 3, 5)
- **Aplicación:** Realce de bordes y detalles
- **Ventaja:** Detecta bordes en todas direcciones
- **Desventaja:** Muy sensible al ruido

#### 🔹 7. Filtro Sobel
- **Propósito:** Detección de bordes mediante gradiente
- **Fórmulas:**
  - Gx (horizontal): `[-1 0 1; -2 0 2; -1 0 1]`
  - Gy (vertical): `[-1 -2 -1; 0 0 0; 1 2 1]`
  - Magnitud: `G = √(Gx² + Gy²)`
- **Parámetro ajustable:** Tamaño de kernel (3, 5, 7)
- **Aplicación:** Detección direccional de bordes
- **Ventaja:** Robustez al ruido, detección direccional
- **Desventaja:** Puede perderse información de bordes débiles

#### 🔹 8. Filtro Canny
- **Propósito:** Detección óptima de bordes multi-etapa
- **Etapas:**
  1. Suavizado gaussiano (reducción de ruido)
  2. Cálculo de gradiente (Sobel)
  3. Supresión no-máxima (adelgazamiento)
  4. Umbralización con histéresis
- **Parámetros ajustables:**
  - Umbral inferior (50-200)
  - Umbral superior (100-300)
- **Aplicación:** Detección precisa de contornos
- **Ventaja:** Mejor relación señal-ruido, bordes continuos
- **Desventaja:** Requiere ajuste cuidadoso de umbrales

### 1.2 Implementación Técnica

```python
def apply_filter(image_np, filter_type, **kwargs):
    """
    Aplica filtro seleccionado con parámetros configurables
    
    Args:
        image_np: Imagen en escala de grises (numpy array)
        filter_type: Tipo de filtro ('Media', 'Mediana', etc.)
        **kwargs: Parámetros específicos del filtro
    
    Returns:
        Imagen filtrada (numpy array)
    """
    if filter_type == "Media":
        ksize = kwargs.get('kernel_size', 5)
        return cv2.blur(image_np, (ksize, ksize))
    
    elif filter_type == "Mediana":
        ksize = kwargs.get('kernel_size', 5)
        return cv2.medianBlur(image_np, ksize)
    
    # ... [7 filtros más]
```

### 1.3 Interfaz de Usuario - Filtros

**Características de la interfaz:**
- Diseño de dos columnas (imagen original | imagen filtrada)
- Selectores de filtro con menú desplegable
- Sliders para ajuste de parámetros en tiempo real
- Visualización simultánea de resultados
- Información del filtro y parámetros aplicados

---

## 🧠 PARTE 2: DESCRIPTORES Y CLASIFICACIÓN (70%)

### 2.1 Descriptores de Características

#### 🔸 HOG (Histogram of Oriented Gradients)

**Concepto:**  
Descriptor que captura la distribución de gradientes de intensidad en regiones locales de la imagen.

**Parámetros de extracción:**
```python
hog_params = {
    'orientations': 9,           # Bins de orientación
    'pixels_per_cell': (8, 8),   # Tamaño de celda
    'cells_per_block': (2, 2),   # Celdas por bloque
    'block_norm': 'L2-Hys',      # Normalización L2-Hys
    'transform_sqrt': True,       # Raíz cuadrada de valores
    'feature_vector': True        # Vector 1D de salida
}
```

**Proceso:**
1. Conversión a escala de grises
2. Cálculo de gradientes (Sobel)
3. División en celdas de 8×8 píxeles
4. Cálculo de histograma de 9 bins por celda
5. Agrupación en bloques de 2×2 celdas
6. Normalización L2-Hys por bloque
7. Concatenación en vector de características

**Dimensión del vector:** ~3,780 características (128×64 imagen)

**Ventajas:**
- Robusto a cambios de iluminación
- Invariante a pequeñas deformaciones
- Captura información de forma/estructura

**Desventajas:**
- Sensible a rotación
- No captura información de textura fina

#### 🔸 LBP (Local Binary Patterns)

**Concepto:**  
Descriptor de textura que codifica la relación entre píxel central y vecinos.

**Parámetros de extracción:**
```python
lbp_params = {
    'radius': 3,        # Radio de vecindad
    'n_points': 24,     # Puntos de muestreo (8 × radius)
    'method': 'uniform' # Patrones uniformes
}
```

**Proceso:**
1. Conversión a escala de grises
2. Para cada píxel (x,y):
   - Muestrear 24 vecinos en radio 3
   - Comparar con valor central
   - Generar código binario
   - Convertir a valor decimal
3. Calcular histograma de patrones
4. Normalizar histograma

**Dimensión del vector:** 26 características (patrones uniformes)

**Ventajas:**
- Invariante a cambios monótonos de iluminación
- Muy eficiente computacionalmente
- Captura información de textura local

**Desventajas:**
- Sensible a ruido
- Pierde información de contraste

### 2.2 Modelos de Clasificación

Se implementaron 3 modelos diferentes para comparación:

#### 🤖 Modelo 1: SVM + HOG

**Arquitectura:**
```python
Pipeline(
    StandardScaler(),              # Normalización Z-score
    LinearSVC(                     # SVM lineal
        max_iter=5000,
        dual=True,
        random_state=42,
        class_weight='balanced'
    )
)
```

**Características:**
- Input: Vector HOG de ~3,780 dimensiones
- Escalado: Media 0, desviación estándar 1
- Clasificador: SVM con kernel lineal
- Clases balanceadas: Pesos inversamente proporcionales

**Entrenamiento:**
- Tiempo estimado: 2-5 minutos
- Memoria requerida: ~500 MB
- Convergencia: 5,000 iteraciones máximas

#### 🤖 Modelo 2: SVM + LBP

**Arquitectura:**
```python
Pipeline(
    StandardScaler(),              # Normalización Z-score
    LinearSVC(                     # SVM lineal
        max_iter=5000,
        dual=True,
        random_state=42,
        class_weight='balanced'
    )
)
```

**Características:**
- Input: Vector LBP de 26 dimensiones
- Escalado: Media 0, desviación estándar 1
- Clasificador: SVM con kernel lineal
- Clases balanceadas: Pesos inversamente proporcionales

**Entrenamiento:**
- Tiempo estimado: <1 minuto
- Memoria requerida: ~100 MB
- Convergencia: Rápida (pocas dimensiones)

#### 🤖 Modelo 3: CNN (Convolutional Neural Network)

**Arquitectura detallada:**

```python
PlateCNN(
    # Bloque convolucional 1
    Conv2d(3 → 32, kernel=3×3, padding=1)
    BatchNorm2d(32)
    ReLU()
    MaxPool2d(2×2)                  # 128×64 → 64×32
    
    # Bloque convolucional 2
    Conv2d(32 → 64, kernel=3×3, padding=1)
    BatchNorm2d(64)
    ReLU()
    MaxPool2d(2×2)                  # 64×32 → 32×16
    
    # Bloque convolucional 3
    Conv2d(64 → 128, kernel=3×3, padding=1)
    BatchNorm2d(128)
    ReLU()
    MaxPool2d(2×2)                  # 32×16 → 16×8
    
    # Bloque convolucional 4
    Conv2d(128 → 256, kernel=3×3, padding=1)
    BatchNorm2d(256)
    ReLU()
    AdaptiveAvgPool2d(1×1)          # 16×8 → 1×1
    
    # Clasificador fully-connected
    Flatten()
    Dropout(0.4)
    Linear(256 → 128)
    ReLU()
    Dropout(0.3)
    Linear(128 → 36)                # 36 clases
)
```

**Parámetros totales:** ~75,000

**Características:**
- Input: Imágenes RGB 128×64×3
- 4 bloques convolucionales con BatchNorm
- Global Average Pooling adaptativo
- 2 capas fully-connected
- Dropout para regularización (40% y 30%)

**Hiperparámetros de entrenamiento:**
```python
optimizer = Adam(
    lr=0.001,              # Learning rate
    weight_decay=1e-4      # Regularización L2
)
loss = CrossEntropyLoss() # Pérdida multi-clase
batch_size = 32
epochs = 20
```

**Proceso de entrenamiento:**
1. Carga de datos con DataLoader
2. Augmentación: normalización RGB
3. Forward pass en batches
4. Cálculo de pérdida (Cross-Entropy)
5. Backpropagation con Adam
6. Validación cada epoch
7. Early stopping si no mejora

**Tiempo estimado:** 10-15 minutos (CPU)  
**Memoria requerida:** ~2 GB

### 2.3 Métricas de Evaluación

Para cada modelo se calculan las siguientes métricas:

#### 📊 Métricas Globales
- **Accuracy:** Porcentaje de predicciones correctas
  ```
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  ```

- **Precision (macro-avg):** Promedio de precisión por clase
  ```
  Precision = TP / (TP + FP)
  ```

- **Recall (macro-avg):** Promedio de sensibilidad por clase
  ```
  Recall = TP / (TP + FN)
  ```

- **F1-Score (macro-avg):** Media armónica de precision y recall
  ```
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ```

#### 📊 Matriz de Confusión
- Visualización 36×36 de predicciones vs. verdad
- Diagonal: Predicciones correctas
- Fuera de diagonal: Confusiones entre clases

#### 📊 Reporte por Clase
- Precision, Recall, F1-Score para cada una de las 36 clases
- Identificación de clases problemáticas

---

## 💻 IMPLEMENTACIÓN TÉCNICA

### 3.1 Stack Tecnológico

**Lenguajes y Frameworks:**
- Python 3.12
- Streamlit 1.39.0 (interfaz web)
- OpenCV 4.10.0 (procesamiento de imágenes)
- scikit-learn 1.5.2 (SVM, métricas)
- PyTorch 2.4.1 (redes neuronales)
- NumPy 1.26.4 (operaciones numéricas)
- Pandas 2.2.3 (manejo de datos)
- Matplotlib 3.9.2 (visualización)

**Infraestructura:**
- Git/GitHub (control de versiones)
- Streamlit Cloud (deployment)
- Python venv (gestión de dependencias)

### 3.2 Estructura del Código

**Organización modular:**
```python
# 1. Configuración y constantes (líneas 1-70)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = Path('models')

# 2. Arquitectura CNN (líneas 71-117)
class PlateCNN(nn.Module):
    # ... definición de capas

# 3. Funciones de filtros (líneas 118-166)
def apply_filter(image_np, filter_type, **kwargs):
    # ... implementación de 8 filtros

# 4. Extracción de descriptores (líneas 167-244)
def extract_hog_features(image_gray, hog_params):
def extract_lbp_features(image_gray, radius, n_points):

# 5. Entrenamiento de modelos (líneas 245-380)
def train_cnn_model(data_root, epochs, lr, batch_size, weight_decay):
def train_svm_hog(data_root, hog_params):
def train_svm_lbp(data_root, radius, n_points):

# 6. Teoría de filtros (líneas 381-1127)
# 9 tabs con explicaciones matemáticas completas

# 7. Interfaz principal (líneas 1128-1580)
def main():
    # Modo Teoría, Filtros, Entrenamiento, Clasificación
```

### 3.3 Optimizaciones Implementadas

#### ⚡ Rendimiento
- Progress bars con ETA cada 10 imágenes durante entrenamiento
- Carga lazy de modelos (solo cuando se necesitan)
- Cache de descriptores para evitar recálculo
- Batch processing en CNN para eficiencia

#### 🔒 Robustez
- Validación de existencia de archivos
- Manejo de excepciones en carga de modelos
- Verificación de dimensiones de entrada
- Normalización automática de imágenes

#### 🎨 Interfaz de Usuario
- Diseño responsive con columnas
- Feedback visual con spinners y progress bars
- Mensajes informativos con st.info/success/warning
- Visualización de resultados en tablas y gráficos

---

## 📈 RESULTADOS

### 4.1 Desempeño de Modelos

Los resultados esperados (basados en arquitectura y dataset):

| Modelo | Accuracy Esperada | Tiempo Entrenamiento | Tamaño Modelo |
|--------|-------------------|---------------------|---------------|
| **SVM + HOG** | 85-92% | 2-5 min | ~15 MB |
| **SVM + LBP** | 70-80% | <1 min | ~1 MB |
| **CNN** | 90-96% | 10-15 min | ~300 KB |

**Análisis comparativo:**

**SVM + HOG:**
- ✅ Buen balance entre precisión y velocidad
- ✅ Interpetable (vectores de soporte)
- ✅ Funciona bien con datos limitados
- ❌ Requiere ingeniería de características manual
- ❌ Escalado lineal con tamaño de dataset

**SVM + LBP:**
- ✅ Muy rápido (26 características)
- ✅ Eficiente en memoria
- ✅ Bueno para texturas
- ❌ Menor precisión que HOG
- ❌ Pierde información espacial global

**CNN:**
- ✅ Mayor precisión potencial
- ✅ Aprendizaje automático de características
- ✅ Escalable a datasets grandes
- ❌ Requiere más datos de entrenamiento
- ❌ Mayor tiempo de entrenamiento
- ❌ Menos interpretable

### 4.2 Despliegue en Producción

**URL de la aplicación:**  
https://clasificaci-n-de-imagenes-6je33ygv8xeoekkn2dl8qd.streamlit.app/

**Configuración de despliegue:**

**packages.txt** (dependencias del sistema):
```bash
libgl1-mesa-glx    # OpenGL para OpenCV
libglib2.0-0       # GLib para procesamiento
```

**requirements.txt** (dependencias de Python):
```
opencv-python-headless==4.10.0.84
numpy==1.26.4
matplotlib==3.9.2
scikit-image==0.24.0
Pillow==10.4.0
scikit-learn==1.5.2
torch==2.4.1
torchvision==0.19.1
pandas==2.2.3
seaborn==0.13.2
streamlit==1.39.0
tqdm==4.66.5
```

**Problemas resueltos durante deployment:**
1. ❌ Python 3.13 incompatible con PyTorch → ✅ Forzar Python 3.12
2. ❌ opencv-python requiere libGL → ✅ Usar opencv-python-headless
3. ❌ width='stretch' deprecado → ✅ use_column_width=True
4. ❌ uv instalando versiones incorrectas → ✅ Remover runtime.txt

---

## 🎓 CONCLUSIONES

### 5.1 Logros del Proyecto

1. **Implementación Completa:**
   - ✅ 8 filtros digitales funcionales con parámetros ajustables
   - ✅ 2 descriptores de características (HOG, LBP) implementados
   - ✅ 3 modelos de clasificación entrenables (SVM×2, CNN)
   - ✅ Interfaz web interactiva y responsive
   - ✅ Despliegue exitoso en Streamlit Cloud

2. **Aprendizajes Técnicos:**
   - Comprensión profunda de filtros en dominio espacial
   - Dominio de descriptores de características tradicionales
   - Experiencia práctica con SVM y redes neuronales
   - Desarrollo de aplicaciones web con Streamlit
   - Gestión de deployment y dependencias en la nube

3. **Resultados Académicos:**
   - Documentación exhaustiva con fundamentos matemáticos
   - Implementación modular y bien estructurada
   - Código reproducible y versionado en Git
   - Aplicación funcional accesible públicamente

### 5.2 Comparación de Enfoques

**Métodos Tradicionales (SVM + Descriptores):**
- Requieren ingeniería de características manual
- Mayor interpretabilidad
- Funcionan bien con datasets pequeños
- Entrenamiento rápido
- Buen desempeño en problemas bien definidos

**Deep Learning (CNN):**
- Aprendizaje automático de características
- Mayor capacidad de generalización
- Requieren más datos y cómputo
- Mejor desempeño en problemas complejos
- Menos interpretables pero más flexibles

### 5.3 Limitaciones y Trabajo Futuro

**Limitaciones actuales:**
- Dataset limitado (1,080 imágenes)
- Entrenamiento solo en CPU (no GPU en Streamlit Cloud)
- Sin data augmentation extensiva
- Modelos no optimizados para producción (sin cuantización)

**Mejoras propuestas:**
1. **Dataset:**
   - Aumentar a 10,000+ imágenes
   - Data augmentation (rotación, escalado, ruido)
   - Incluir imágenes de diferentes fuentes

2. **Modelos:**
   - Transfer learning (ResNet, EfficientNet pre-entrenados)
   - Ensemble de modelos (voting)
   - Optimización de hiperparámetros (Grid Search, Bayesian Opt)
   - Cuantización post-entrenamiento para inferencia rápida

3. **Aplicación:**
   - API REST para integración
   - Modo batch para procesamiento masivo
   - Caché de modelos para reducir latencia
   - Soporte para GPU en deployment

4. **Filtros:**
   - Más filtros (bilateral, anisotropic diffusion)
   - Filtros en dominio de frecuencia (FFT, DCT)
   - Procesamiento en color (espacios HSV, LAB)

### 5.4 Reflexión Personal

Este proyecto demostró la importancia de entender tanto los fundamentos teóricos (filtros digitales, descriptores de características) como las herramientas modernas (deep learning, cloud deployment). La implementación práctica reveló que:

- **No existe un modelo perfecto:** Cada enfoque tiene trade-offs
- **La ingeniería de datos es crucial:** Preprocesamiento y extracción de características impactan significativamente
- **La visualización es poderosa:** Una interfaz intuitiva facilita la comprensión
- **El deployment tiene desafíos únicos:** Compatibilidad de dependencias, limitaciones de recursos

La experiencia de llevar un proyecto desde la teoría hasta una aplicación web funcional proporciona una visión integral del ciclo de vida del desarrollo de software en machine learning.

---

## 📚 REFERENCIAS

### Teoría de Filtros
1. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
2. Pratt, W. K. (2007). *Digital Image Processing* (4th ed.). Wiley-Interscience.

### Descriptores de Características
3. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR*.
4. Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *TPAMI*, 24(7), 971-987.

### Machine Learning
5. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.
6. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.

### Herramientas
7. Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal*.
8. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
9. Paszke et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*.

---

## 📎 ANEXOS

### Anexo A: Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/Emma-Ok/Clasificaci-n-de-imagenes.git
cd Clasificaci-n-de-imagenes

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app_streamlit_completa.py
```

### Anexo B: Uso de la Aplicación

**Modo 1: Teoría de Filtros**
1. Seleccionar pestaña deseada (Resumen, Media, Mediana, etc.)
2. Leer explicación teórica con fórmulas matemáticas
3. Ver ejemplos de aplicación

**Modo 2: Filtros Parte 1**
1. Subir imagen (JPG, PNG, JPEG)
2. Seleccionar filtro del menú desplegable
3. Ajustar parámetros con sliders
4. Ver resultado en tiempo real
5. Comparar con imagen original

**Modo 3: Entrenamiento**
1. Seleccionar modelo (CNN, SVM+HOG, SVM+LBP)
2. Ajustar hiperparámetros
3. Click en "Entrenar Modelo"
4. Esperar a que termine (ver progress bar)
5. Revisar métricas y matriz de confusión

**Modo 4: Clasificación**
1. Subir imagen a clasificar
2. Seleccionar modelo entrenado
3. Click en "Clasificar"
4. Ver predicción con probabilidades
5. Revisar top-5 predicciones

### Anexo C: Estructura de Archivos

```
Tarea2Imagenes/
├── app_streamlit_completa.py    # Aplicación principal (1,580 líneas)
├── requirements.txt              # Dependencias de Python
├── packages.txt                  # Dependencias del sistema
├── .streamlit/
│   ├── config.toml              # Configuración de Streamlit
│   └── secrets.toml             # Secretos (vacío)
├── data/
│   ├── train/                   # 864 imágenes de entrenamiento
│   │   ├── class_0/ ... class_Z/
│   └── val/                     # 216 imágenes de validación
│       ├── class_0/ ... class_Z/
├── models/                      # Modelos entrenados (no en repo)
│   ├── cnn_plate_classifier.pth
│   ├── svm_hog_classifier.pkl
│   ├── svm_lbp_classifier.pkl
│   ├── classes.npy
│   └── descriptor_config.pkl
├── TEORIA.md                    # Documentación teórica
├── INFORME.md                   # Este informe
├── README.md                    # Instrucciones del proyecto
└── .gitignore                   # Archivos ignorados por Git
```

### Anexo D: Comandos Git Útiles

```bash
# Ver estado
git status

# Añadir cambios
git add .

# Commit
git commit -m "Descripción del cambio"

# Push a GitHub
git push origin main

# Ver historial
git log --oneline

# Ver diferencias
git diff
```

---

## ✅ CHECKLIST DE COMPLETITUD

- [x] **PARTE 1 (30%): Filtros**
  - [x] 8 filtros implementados
  - [x] Parámetros ajustables
  - [x] Visualización interactiva
  - [x] Documentación teórica completa

- [x] **PARTE 2 (70%): Clasificación**
  - [x] Descriptores HOG implementados
  - [x] Descriptores LBP implementados
  - [x] SVM + HOG funcional
  - [x] SVM + LBP funcional
  - [x] CNN implementada y entrenable
  - [x] Métricas de evaluación completas
  - [x] Interfaz de clasificación

- [x] **Infraestructura**
  - [x] Código versionado en Git
  - [x] Repositorio público en GitHub
  - [x] Aplicación desplegada en Streamlit Cloud
  - [x] Documentación exhaustiva
  - [x] Informe técnico completo

---

**Firma Digital:**  
Emmanuel Bustamante  
Universidad de Antioquia  
Noviembre 2025

---

*Este informe fue generado como parte de la Tarea 2 del curso de Procesamiento Digital de Imágenes. Todo el código es original y está disponible públicamente en GitHub bajo licencia MIT.*
