# 📚 Resumen Teórico - Tarea 2

## PARTE 1: FILTROS DE IMÁGENES (30%)

### 1. Filtro de Media

**Definición**: Promedio aritmético de píxeles vecinos en una ventana.

**Fórmula**:
```
g(x,y) = (1/mn) × Σ Σ f(x+i, y+j)
```
donde `m×n` es el tamaño de la máscara.

**Ejemplo**: Ventana 5×5 → promedia 25 píxeles

**Ventajas**:
- ✅ Simple y eficiente
- ✅ Reduce ruido gaussiano
- ✅ Preserva la media global

**Desventajas**:
- ❌ Difumina bordes
- ❌ Sensible a valores atípicos

---

### 2. Filtro de Mediana

**Definición**: Reemplaza cada píxel por la mediana de su vecindad.

**Fórmula**:
```
g(x,y) = mediana{f(x+i, y+j) : (i,j) ∈ W}
```

**Ejemplo**: Kernel 3×3 → ordena 9 valores, toma el del medio

**Ventajas**:
- ✅ Excelente para ruido sal y pimienta
- ✅ Preserva bordes mejor que la media

**Desventajas**:
- ❌ Más costoso computacionalmente
- ❌ Puede distorsionar texturas finas

---

### 3. Filtro Logarítmico

**Definición**: Transformación punto a punto que comprime rango dinámico.

**Fórmula**:
```
g(x,y) = c × log(1 + f(x,y))
```
donde `c` es constante de escala.

**Ejemplo**: `c = 255/log(2)` para imágenes 8-bit

**Ventajas**:
- ✅ Realza detalles en sombras
- ✅ Útil para HDR (High Dynamic Range)

**Desventajas**:
- ❌ Amplifica ruido en bajas intensidades
- ❌ Requiere normalización

---

### 4. Filtro de Cuadro Normalizado

**Definición**: Variante del filtro de media con coeficientes uniformes que suman 1.

**Fórmula**:
```
g(x,y) = Σ Σ (1/k²) × f(x+i, y+j)
```
donde `k` es el lado del kernel cuadrado.

**Ejemplo**: Kernel 7×7 → cada coeficiente = 1/49

**Ventajas**:
- ✅ Suavizado controlado
- ✅ Implementación optimizada (integrales)

**Desventajas**:
- ❌ Similar a media: difumina bordes
- ❌ Artefactos de bloque si se aplica iterativamente

---

### 5. Filtro Gaussiano

**Definición**: Convolución con función gaussiana 2D.

**Fórmula**:
```
G(i,j) = (1/(2πσ²)) × exp(-(i² + j²)/(2σ²))
```

**Ejemplo**: σ=1.0, máscara 5×5 → mayor peso al centro

**Ventajas**:
- ✅ Reduce ruido gaussiano eficientemente
- ✅ Separable en 1D (más rápido)
- ✅ Preserva bordes mejor que media

**Desventajas**:
- ❌ Difumina detalles muy finos
- ❌ Requiere elegir σ apropiado

---

### 6. Filtro Laplace

**Definición**: Derivada de segundo orden, detecta cambios bruscos de intensidad.

**Fórmula**:
```
∇²f = ∂²f/∂x² + ∂²f/∂y²

Aproximación discreta:
∇²f ≈ -4f(x,y) + f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1)
```

**Kernel común**:
```
[ 0  1  0]
[ 1 -4  1]
[ 0  1  0]
```

**Ventajas**:
- ✅ Detecta bordes en todas direcciones
- ✅ Implementación simple

**Desventajas**:
- ❌ Muy sensible al ruido
- ❌ Resultado no directamente visualizable

---

### 7. Filtro Sobel

**Definición**: Derivada de primer orden con suavizado, calcula gradiente direccional.

**Fórmulas**:
```
Gx = [-1  0  1]      Gy = [-1 -2 -1]
     [-2  0  2]           [ 0  0  0]
     [-1  0  1]           [ 1  2  1]

Magnitud: |∇f| = √(Gx² + Gy²)
```

**Ventajas**:
- ✅ Detecta bordes con reducción de ruido
- ✅ Computacionalmente eficiente
- ✅ Proporciona dirección del gradiente

**Desventajas**:
- ❌ Sensible a ruido fuerte
- ❌ Requiere umbralización posterior

---

### 8. Filtro Canny

**Definición**: Detector de bordes multietapa optimizado.

**Pasos**:
1. Suavizado gaussiano
2. Cálculo de gradiente (Sobel)
3. Supresión no-máxima
4. Umbralización con histéresis (doble umbral)

**Parámetros típicos**: σ=1, umbral_bajo=50, umbral_alto=150

**Ventajas**:
- ✅ Mejor detector de bordes (localización precisa)
- ✅ Bordes continuos y delgados
- ✅ Control fino con umbrales

**Desventajas**:
- ❌ Más costoso computacionalmente
- ❌ Sensible a elección de parámetros

---

## PARTE 2: DESCRIPTORES Y CLASIFICACIÓN (70%)

### Descriptores de Características

#### HOG (Histogram of Oriented Gradients)

**Concepto**: Histograma de gradientes orientados en regiones locales.

**Parámetros**:
- Orientaciones: 9 bins (0-180°)
- Píxeles por celda: 8×8
- Celdas por bloque: 2×2
- Normalización: L2-Hys

**Proceso**:
1. Calcular gradientes (magnitud y dirección)
2. Dividir imagen en celdas
3. Crear histograma de orientaciones por celda
4. Normalizar bloques de celdas
5. Concatenar vectores

**Aplicaciones**: Detección de peatones, reconocimiento de objetos, OCR

**Ventajas**:
- ✅ Robusto a cambios de iluminación
- ✅ Captura información de forma y contorno
- ✅ Invariante a pequeñas deformaciones

**Desventajas**:
- ❌ Alto dimensional
- ❌ Sensible a rotación
- ❌ No captura información de color

---

#### LBP (Local Binary Patterns)

**Concepto**: Descriptor de textura que compara píxeles con sus vecinos.

**Parámetros**:
- Radio: 3 píxeles
- Puntos: 24 (8 × radio)
- Método: uniform (patrones uniformes)

**Proceso**:
1. Para cada píxel, comparar con vecinos circulares
2. Asignar 1 si vecino ≥ centro, 0 si <
3. Convertir patrón binario a número decimal
4. Generar histograma de patrones

**Fórmula**:
```
LBP(x,y) = Σ s(gp - gc) × 2^p

donde s(x) = 1 si x ≥ 0, 0 si x < 0
```

**Aplicaciones**: Reconocimiento facial, análisis de texturas, clasificación de materiales

**Ventajas**:
- ✅ Invariante a cambios monótonos de iluminación
- ✅ Computacionalmente eficiente
- ✅ Dimensión baja (histograma compacto)

**Desventajas**:
- ❌ Pierde información de contraste
- ❌ Sensible a rotación (sin extensiones)
- ❌ No captura información de forma global

---

### Clasificadores

#### SVM (Support Vector Machine)

**Concepto**: Encuentra el hiperplano óptimo que maximiza el margen entre clases.

**Función objetivo**:
```
min (1/2)||w||² + C × Σ ξi

sujeto a: yi(w·xi + b) ≥ 1 - ξi
```

**Kernel lineal** (usado aquí):
```
K(xi, xj) = xi · xj
```

**Ventajas**:
- ✅ Efectivo en espacios de alta dimensión
- ✅ Robusto al sobreajuste
- ✅ Funciona bien con datasets pequeños

**Desventajas**:
- ❌ Costoso para datasets grandes
- ❌ Sensible a desbalance de clases
- ❌ Requiere normalización de datos

---

#### CNN (Convolutional Neural Network)

**Arquitectura usada**:
```
Input (3×128×64)
    ↓
Conv2d(3→32) + BN + ReLU + MaxPool
    ↓
Conv2d(32→64) + BN + ReLU + MaxPool
    ↓
Conv2d(64→128) + BN + ReLU + MaxPool
    ↓
Conv2d(128→256) + BN + ReLU + AdaptiveAvgPool
    ↓
Flatten → Dropout(0.4) → Linear(256→128) → ReLU
    ↓
Dropout(0.3) → Linear(128→36)
    ↓
Output (36 clases)
```

**Componentes**:
- **Convolución**: Aprende filtros automáticamente
- **BatchNorm**: Estabiliza entrenamiento
- **MaxPool**: Reduce dimensionalidad
- **Dropout**: Previene sobreajuste
- **AdaptiveAvgPool**: Independiente de tamaño

**Optimización**:
- Función de pérdida: CrossEntropyLoss
- Optimizador: Adam
- Learning rate: 0.001
- Weight decay: 0.0001

**Ventajas**:
- ✅ Aprende características automáticamente
- ✅ Superior desempeño general
- ✅ Robusto a variaciones

**Desventajas**:
- ❌ Requiere más datos
- ❌ Computacionalmente intensivo
- ❌ "Caja negra" (difícil interpretación)

---

### Métricas de Evaluación

#### 1. Accuracy (Exactitud)
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Proporción de predicciones correctas.

#### 2. Precision (Precisión)
```
Precision = TP / (TP + FP)
```
De las predicciones positivas, cuántas son correctas.

#### 3. Recall (Sensibilidad)
```
Recall = TP / (TP + FN)
```
De los casos positivos reales, cuántos se detectaron.

#### 4. F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Media armónica entre precision y recall.

#### 5. Matriz de Confusión

```
                Predicho
              Pos    Neg
Real  Pos  [  TP  |  FN  ]
      Neg  [  FP  |  TN  ]
```

- **TP (True Positive)**: Positivos correctamente clasificados
- **TN (True Negative)**: Negativos correctamente clasificados
- **FP (False Positive)**: Negativos clasificados como positivos (Error Tipo I)
- **FN (False Negative)**: Positivos clasificados como negativos (Error Tipo II)

---

## Comparación de Enfoques

| Aspecto | SVM+HOG | SVM+LBP | CNN |
|---------|---------|---------|-----|
| **Accuracy típica** | 75-85% | 70-80% | 85-95% |
| **Tiempo entrenamiento** | Medio | Bajo | Alto |
| **Tiempo inferencia** | Bajo | Bajo | Medio |
| **Interpretabilidad** | Alta | Alta | Baja |
| **Requiere ingeniería** | Sí | Sí | No |
| **Datos necesarios** | Pocos | Pocos | Moderados |
| **Generalización** | Media | Media | Alta |

---

## Referencias Académicas

1. **Filtros**: Gonzalez & Woods - "Digital Image Processing"
2. **HOG**: Dalal & Triggs (2005) - "Histograms of Oriented Gradients for Human Detection"
3. **LBP**: Ojala et al. (2002) - "Multiresolution Gray-Scale and Rotation Invariant Texture Classification"
4. **SVM**: Cortes & Vapnik (1995) - "Support-Vector Networks"
5. **CNN**: LeCun et al. (1998) - "Gradient-Based Learning Applied to Document Recognition"
6. **Canny**: Canny (1986) - "A Computational Approach to Edge Detection"

---

**Documento preparado para**: Tarea 2 - Filtros y Descriptores de Imágenes  
**Última actualización**: Noviembre 2025
