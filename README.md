# 🖼️ Tarea 2: Filtros y Descriptores de Imágenes

Proyecto académico completo de procesamiento de imágenes que implementa filtros clásicos, extracción de descriptores (HOG, LBP) y clasificación con SVM y CNN.

## 📋 Contenido del Proyecto

### Parte 1: Filtros (30%)
Implementación y demostración de 8 filtros con sus fundamentos matemáticos:
- ✅ Filtro de Media
- ✅ Filtro de Mediana  
- ✅ Filtro Logarítmico
- ✅ Filtro de Cuadro Normalizado
- ✅ Filtro Gaussiano
- ✅ Filtro Laplace
- ✅ Filtro Sobel
- ✅ Filtro Canny

Cada filtro incluye:
- Formulación matemática
- Ejemplo explicado
- Ventajas y desventajas
- Implementación en OpenCV

### Parte 2: Descriptores y Clasificación (70%)

#### Descriptores Implementados
1. **HOG (Histogram of Oriented Gradients)**
   - Orientaciones: 9
   - Píxeles por celda: 8x8
   - Celdas por bloque: 2x2

2. **LBP (Local Binary Patterns)**
   - Radio: 3
   - Puntos: 24
   - Método: uniform

#### Clasificadores
1. **SVM + HOG**: Support Vector Machine con características HOG
2. **SVM + LBP**: Support Vector Machine con características LBP
3. **CNN (PyTorch)**: Red neuronal convolucional de 4 capas

#### Métricas Evaluadas
- ✅ Accuracy (Exactitud)
- ✅ Precision (Precisión)
- ✅ Recall (Sensibilidad)
- ✅ F1-Score
- ✅ Matriz de Confusión
- ✅ Falsos Positivos/Negativos

## 🚀 Instalación

### 1. Instalar dependencias

```powershell
pip install opencv-python numpy matplotlib scikit-image scikit-learn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas seaborn pillow streamlit plotly
```

### 2. Estructura del proyecto

```
Tarea2Imagenes/
├── main.ipynb                      # Notebook con teoría e implementación
├── app_streamlit_completa.py       # Interfaz gráfica completa
├── README.md                       # Este archivo
├── data/                          # Dataset de imágenes
│   ├── train/                     # Imágenes de entrenamiento
│   │   ├── class_0/
│   │   ├── class_A/
│   │   └── ...
│   └── val/                       # Imágenes de validación
│       ├── class_0/
│       └── ...
└── models/                        # Modelos entrenados (se genera)
    ├── cnn_plate_classifier.pth
    ├── svm_hog_classifier.pkl
    ├── svm_lbp_classifier.pkl
    ├── classes.npy
    └── descriptor_config.pkl
```

## 📊 Uso

### Opción 1: Jupyter Notebook (Trabajo académico completo)

```powershell
jupyter notebook main.ipynb
```

Ejecutar las celdas en orden para:
1. Ver la teoría de cada filtro con fórmulas
2. Aplicar filtros a imágenes de ejemplo
3. Extraer características HOG y LBP
4. Entrenar SVM y CNN
5. Evaluar métricas y comparar modelos

### Opción 2: Interfaz Streamlit (Aplicación interactiva)

```powershell
streamlit run app_streamlit_completa.py
```

La interfaz tiene 3 modos:

#### 🔍 Modo 1: Filtros (Parte 1)
- Cargar cualquier imagen
- Seleccionar filtro a aplicar
- Ajustar parámetros interactivamente
- Visualizar resultados en tiempo real
- Descargar imagen filtrada

#### 🤖 Modo 2: Descriptores y Clasificación (Parte 2)
- Configurar hiperparámetros de entrenamiento
- Seleccionar modelos a entrenar (CNN, SVM+HOG, SVM+LBP)
- Entrenar con barra de progreso
- Visualizar curvas de entrenamiento
- Guardar modelos automáticamente

#### 🎯 Modo 3: Clasificar Imagen
- Cargar modelos pre-entrenados
- Subir nueva imagen
- Seleccionar clasificador (CNN, SVM+HOG, SVM+LBP)
- Ver predicción con nivel de confianza
- Visualizar distribución de probabilidades

## 🔬 Ejemplos de Uso

### Entrenar todos los modelos

1. Abrir Streamlit: `streamlit run app_streamlit_completa.py`
2. Ir a modo "🤖 Descriptores y Clasificación (Parte 2)"
3. Configurar:
   - Ruta del dataset: `data`
   - Épocas CNN: 10-15
   - Batch size: 32
   - Learning rate: 0.001
4. Marcar todos los modelos (CNN, SVM+HOG, SVM+LBP)
5. Clic en "🚀 Iniciar Entrenamiento"
6. Esperar a que termine y se guarden los modelos

### Clasificar una imagen nueva

1. Ir a modo "🎯 Clasificar Imagen"
2. Los modelos se cargan automáticamente
3. Subir imagen de un carácter de placa
4. Seleccionar modelo (CNN recomendado)
5. Clic en "🔍 Clasificar"
6. Ver predicción y confianza

### Aplicar filtros

1. Ir a modo "🔍 Filtros (Parte 1)"
2. Subir imagen
3. Seleccionar filtro (ej: Canny)
4. Ajustar parámetros con los sliders
5. Clic en "🔄 Aplicar Filtro"
6. Descargar resultado si es necesario

## 📈 Dataset

El proyecto usa un dataset de caracteres de placas vehiculares:
- **36 clases**: 0-9, A-Z
- **864 imágenes de entrenamiento**
- **216 imágenes de validación**
- **Tamaño estandarizado**: 128x64 píxeles

Estructura esperada:
```
data/
├── train/
│   ├── class_0/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── class_A/
│   └── ...
└── val/
    ├── class_0/
    └── ...
```

## 🎯 Resultados Esperados

### Métricas típicas (según dataset)

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| SVM+HOG | ~75-85% | ~0.74-0.84 | ~0.73-0.83 | ~0.73-0.83 |
| SVM+LBP | ~70-80% | ~0.69-0.79 | ~0.68-0.78 | ~0.68-0.78 |
| CNN | ~85-95% | ~0.85-0.95 | ~0.84-0.94 | ~0.85-0.95 |

**Nota**: La CNN generalmente supera a los SVMs por su capacidad de aprender características automáticamente.

## 🛠️ Requisitos Técnicos

- **Python**: 3.8+
- **RAM**: 4GB mínimo (8GB recomendado)
- **GPU**: Opcional (CPU funciona bien para este dataset pequeño)
- **Almacenamiento**: ~500MB para dataset + modelos

### Dependencias principales

```
opencv-python >= 4.8.0
numpy >= 1.24.0
matplotlib >= 3.7.0
scikit-image >= 0.21.0
scikit-learn >= 1.3.0
torch >= 2.0.0
torchvision >= 0.15.0
streamlit >= 1.28.0
pandas >= 2.0.0
seaborn >= 0.12.0
```

## 📝 Notas Académicas

### Cumplimiento de requisitos

**PARTE 1 (30%)**:
- ✅ Investigación de 8 filtros con fórmulas matemáticas
- ✅ Ejemplos explicados para cada filtro
- ✅ Ventajas y desventajas documentadas
- ✅ Implementación en OpenCV

**PARTE 2 (70%)**:
- ✅ Banco de imágenes generado (36 clases)
- ✅ Preprocesamiento (resize a 128x64)
- ✅ Características HOG extraídas y definidas
- ✅ Características LBP extraídas (descriptor adicional)
- ✅ SVM entrenado para HOG
- ✅ SVM entrenado para LBP
- ✅ CNN (red neuronal) entrenada
- ✅ Métricas investigadas e implementadas:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Matriz de Confusión
  - Falsos Positivos/Negativos
- ✅ Interfaz gráfica con Streamlit
- ✅ Clasificación de nuevas imágenes

### Documentación

- **Notebook**: Contiene toda la teoría, fórmulas y experimentos
- **Streamlit**: Aplicación práctica e interactiva
- **README**: Instrucciones de uso y referencia

## 🐛 Troubleshooting

### Error: "No se encontraron modelos"
**Solución**: Entrenar modelos primero en el modo 2 de Streamlit o ejecutar el notebook completo.

### Error: "Ruta del dataset no existe"
**Solución**: Verificar que existe la carpeta `data/` con subcarpetas `train/` y `val/`.

### Entrenamiento muy lento
**Solución**: 
- Reducir épocas (usar 5-10 en lugar de 15-30)
- Aumentar batch size a 64
- Si disponible, usar GPU cambiando en el código

### Errores de importación
**Solución**: Reinstalar dependencias con:
```powershell
pip install --upgrade -r requirements.txt
```

## 📧 Contacto

Este es un proyecto académico. Para consultas técnicas, revisar:
- Código fuente en `main.ipynb`
- Implementación en `app_streamlit_completa.py`
- Documentación en línea de cada librería

## 📄 Licencia

Proyecto académico - Tarea 2 de Procesamiento de Imágenes

---

**Desarrollado con**: Python 🐍 | OpenCV 📸 | PyTorch 🔥 | Streamlit ⚡
