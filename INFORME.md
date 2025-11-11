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

### 1. Fundamentos Teóricos

El procesamiento digital de imágenes se basa en dos pilares fundamentales:

**1.1 Procesamiento en el Dominio Espacial**  
Operaciones que se aplican directamente sobre los píxeles de la imagen. Los filtros espaciales modifican los valores de intensidad mediante operaciones matemáticas sobre vecindades locales.

**1.2 Análisis y Extracción de Características**  
Transformación de la información visual en representaciones numéricas que capturan propiedades relevantes de la imagen (forma, textura, bordes).

### 2. Enfoque Experimental

El proyecto aborda dos problemas fundamentales del procesamiento de imágenes:

**PARTE 1 - Filtrado Espacial (30%)**
- Implementación de 8 filtros clásicos
- Análisis comparativo de efectos
- Estudio de parámetros óptimos

**PARTE 2 - Clasificación de Patrones (70%)**
- Extracción de descriptores de características
- Entrenamiento de modelos supervisados
- Evaluación cuantitativa del desempeño

### 3. Dataset

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

### 1.2 Fundamento Matemático de Filtros

Los filtros espaciales se pueden clasificar en dos categorías principales:

**Filtros de Suavizado (Pasa-Bajas)**
- Reducen variaciones abruptas de intensidad
- Aplicaciones: reducción de ruido, preprocesamiento
- Trade-off: pérdida de detalles vs. reducción de ruido

**Filtros de Realce (Pasa-Altas)**
- Enfatizan transiciones rápidas de intensidad
- Aplicaciones: detección de bordes, sharpening
- Trade-off: sensibilidad al ruido vs. detección de detalles

**Operación de Convolución**  
Base matemática de los filtros lineales:

```
g(x,y) = Σ Σ f(x+i, y+j) × h(i,j)
```

Donde:
- `f(x,y)`: imagen original
- `h(i,j)`: kernel del filtro
- `g(x,y)`: imagen resultante

### 1.3 Análisis Comparativo de Filtros

**Selección según tipo de ruido:**
- **Ruido gaussiano** → Filtro de Media o Gaussiano
- **Ruido sal y pimienta** → Filtro de Mediana
- **Ruido en imágenes HDR** → Filtro Logarítmico

**Selección según aplicación:**
- **Preprocesamiento general** → Gaussiano
- **Detección de bordes** → Sobel, Canny
- **Realce de detalles** → Laplaciano

---

## 🧠 PARTE 2: DESCRIPTORES Y CLASIFICACIÓN (70%)

### 2.1 Descriptores de Características

#### 🔸 HOG (Histogram of Oriented Gradients)

**Fundamento Teórico:**  
El descriptor HOG se basa en el principio de que la forma y apariencia de objetos locales pueden ser caracterizadas por la distribución de gradientes de intensidad o direcciones de bordes, incluso sin conocimiento preciso de las ubicaciones de los bordes.

**Base Matemática:**

1. **Cálculo del Gradiente:**
   ```
   Gx = I(x+1,y) - I(x-1,y)
   Gy = I(x,y+1) - I(x,y-1)
   Magnitud: G = √(Gx² + Gy²)
   Orientación: θ = arctan(Gy/Gx)
   ```

2. **Histograma de Orientaciones:**
   - División del espacio angular (0°-180°) en 9 bins
   - Cada gradiente vota en bins según su orientación
   - Peso del voto proporcional a la magnitud

3. **Normalización por Bloques:**
   - Agrupa celdas en bloques de 2×2
   - Normalización L2-Hys para robustez a iluminación
   ```
   v_norm = v / √(||v||² + ε²)
   ```

**Propiedades Fundamentales:**
- **Invariancia a iluminación:** Normalización por bloques
- **Invariancia a traslación:** Uso de gradientes locales
- **Sensibilidad a forma:** Captura estructura geométrica

**Ventajas:**
- Robusto a cambios de iluminación
- Invariante a pequeñas deformaciones
- Captura información de forma/estructura

**Desventajas:**
- Sensible a rotación
- No captura información de textura fina

#### 🔸 LBP (Local Binary Patterns)

**Fundamento Teórico:**  
LBP es un operador de textura que caracteriza la estructura espacial de texturas locales mediante comparaciones binarias entre un píxel central y su vecindad circular.

**Base Matemática:**

1. **Codificación Binaria:**
   ```
   LBP(xc,yc) = Σ(i=0 to P-1) s(gi - gc) × 2^i
   
   donde:
   s(x) = 1 si x ≥ 0
   s(x) = 0 si x < 0
   ```

2. **Muestreo Circular:**
   - P puntos en círculo de radio R
   - Coordenadas: `(xc + R×cos(2πi/P), yc + R×sin(2πi/P))`
   - Interpolación bilineal para posiciones no enteras

3. **Patrones Uniformes:**
   - Patrón uniforme: máximo 2 transiciones 0→1 o 1→0
   - Reduce dimensionalidad: 256 patrones → 59 uniformes
   - Captura ~90% de texturas naturales

**Propiedades Fundamentales:**
- **Invariancia monotónica:** Robusto a cambios de iluminación
- **Invariancia rotacional:** Versión extendida (LBP^riu2)
- **Eficiencia computacional:** Operaciones binarias simples

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

**Fundamento Teórico:**  
Las Máquinas de Vectores de Soporte (SVM) son clasificadores que buscan el hiperplano óptimo que maximiza el margen entre clases en un espacio de alta dimensionalidad.

**Formulación Matemática:**

**Problema de optimización:**
```
minimizar: ½||w||² + C Σ ξi
sujeto a: yi(w·xi + b) ≥ 1 - ξi
```

Donde:
- `w`: vector normal al hiperplano
- `b`: término de sesgo
- `C`: parámetro de regularización
- `ξi`: variables de holgura (slack)

**Función de decisión:**
```
f(x) = sign(w·x + b)
```

**Características del enfoque SVM+HOG:**
- **Espacio de características:** ~3,780 dimensiones (HOG)
- **Kernel lineal:** Eficiente en alta dimensionalidad
- **Normalización:** Z-score para escala uniforme
- **Pesos balanceados:** Compensa desbalance de clases

#### 🤖 Modelo 2: SVM + LBP

**Fundamento Teórico:**  
Este modelo combina la capacidad de LBP para capturar micro-texturas con la robustez del clasificador SVM.

**Diferencias con SVM+HOG:**

**Espacio de características:**
- **Dimensionalidad:** 26 vs. 3,780 (HOG)
- **Tipo de información:** Textura vs. Forma
- **Complejidad:** Baja vs. Alta

**Ventajas del espacio reducido:**
- Convergencia más rápida
- Menor riesgo de overfitting
- Eficiencia computacional

**Trade-offs:**
- ⬆️ Velocidad de entrenamiento
- ⬇️ Capacidad de representación
- ⬇️ Precisión en patrones complejos

#### 🤖 Modelo 3: CNN (Convolutional Neural Network)

**Fundamento Teórico:**  
Las Redes Neuronales Convolucionales aprenden jerarquías de características directamente de los datos, desde bordes simples hasta patrones complejos.

**Principios Fundamentales:**

**1. Operación de Convolución:**
```
S(i,j) = (I * K)(i,j) = Σ Σ I(m,n)K(i-m, j-n)
                        m  n
```

**2. Campos Receptivos:**
- Cada neurona "ve" una región local de la entrada
- Campos receptivos crecen con la profundidad
- Captura patrones de complejidad creciente

**3. Arquitectura Jerárquica:**

**Nivel 1 (Baja complejidad):**
- Detectores de bordes (horizontal, vertical, diagonal)
- Filtros Gabor aprendidos
- Patrones locales simples

**Nivel 2 (Media complejidad):**
- Combinaciones de bordes
- Formas básicas (curvas, esquinas)
- Texturas simples

**Nivel 3 (Alta complejidad):**
- Partes de objetos
- Patrones recurrentes
- Características discriminativas

**Nivel 4 (Muy alta complejidad):**
- Representaciones globales
- Características de clase
- Patrones abstractos

**4. Componentes Clave:**

**Convolución:**
- Extracción de características locales
- Compartir pesos reduce parámetros
- Invariancia a traslación

**Pooling:**
- Reducción de dimensionalidad espacial
- Invariancia a pequeñas deformaciones
- Reduce overfitting

**Batch Normalization:**
- Estabiliza el entrenamiento
- Permite learning rates mayores
- Regularización implícita

**Dropout:**
- Regularización explícita
- Previene co-adaptación de neuronas
- Simula ensemble de redes

**5. Función de Pérdida:**

**Cross-Entropy multi-clase:**
```
L = -Σ yi × log(ŷi)
     i

donde:
yi: etiqueta verdadera (one-hot)
ŷi: probabilidad predicha (softmax)
```

**6. Optimización:**

**Adam Optimizer:**
- Combina momentum + RMSprop
- Tasas de aprendizaje adaptativas
- Convergencia rápida y estable

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

## 💻 MARCO TEÓRICO DEL PROCESAMIENTO

### 3.1 Teoría de Señales e Imágenes

**Representación Digital:**
Una imagen digital es una función bidimensional `f(x,y)` donde:
- `x, y`: coordenadas espaciales discretas
- `f`: intensidad o nivel de gris en ese punto

**Teorema de Muestreo (Nyquist-Shannon):**
```
fs ≥ 2 × fmax
```
La frecuencia de muestreo debe ser al menos el doble de la frecuencia máxima para evitar aliasing.

### 3.2 Espacios de Color

**RGB (Red, Green, Blue):**
- Modelo aditivo basado en percepción humana
- Cada píxel: (R, G, B) ∈ [0, 255]³

**Escala de Grises:**
```
Gray = 0.299×R + 0.587×G + 0.114×B
```
Ponderación basada en sensibilidad del ojo humano.

### 3.3 Transformaciones Fundamentales

**1. Transformaciones Puntuales:**
Operan píxel por píxel independientemente:
```
g(x,y) = T[f(x,y)]
```

**2. Transformaciones Locales:**
Usan vecindades (convolución):
```
g(x,y) = Σ Σ f(x+i,y+j) × h(i,j)
         i j
```

**3. Transformaciones Globales:**
Consideran toda la imagen (FFT, histograma)

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

### 5.2 Análisis Teórico Comparativo

**Paradigmas de Aprendizaje:**

**Enfoque Tradicional (Descriptores Manuales + SVM):**

**Ventajas teóricas:**
- **Base matemática sólida:** HOG y LBP tienen interpretación geométrica clara
- **Garantías teóricas:** SVM maximiza margen con fundamento estadístico
- **Eficiencia en datos:** Funciona con datasets limitados (teoría VC)
- **Interpretabilidad:** Vectores de soporte son ejemplares representativos

**Limitaciones teóricas:**
- **Sesgo inductivo fijo:** Características diseñadas a priori
- **Pérdida de información:** Compresión manual puede descartar patrones relevantes
- **Escalabilidad:** Complejidad O(n²) en SVM estándar

**Enfoque Moderno (Deep Learning - CNN):**

**Ventajas teóricas:**
- **Teorema de aproximación universal:** Puede aproximar cualquier función continua
- **Aprendizaje jerárquico:** Descubre representaciones óptimas automáticamente
- **Invariancia aprendida:** Adquiere invariancias relevantes del problema
- **Composicionalidad:** Combina características simples en complejas

**Limitaciones teóricas:**
- **Caja negra:** Difícil interpretación de características aprendidas
- **Mínimos locales:** Optimización no convexa
- **Requisitos de datos:** Necesita ejemplos suficientes para generalizar
- **Overfitting:** Alto riesgo con modelos sobreparametrizados

**Teoría del Aprendizaje Estadístico:**

Ambos enfoques buscan minimizar el riesgo esperado:
```
R(f) = E[L(Y, f(X))]
```

Pero difieren en cómo:
- **SVM:** Minimiza riesgo estructural (margen + error)
- **CNN:** Minimiza riesgo empírico con regularización

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
