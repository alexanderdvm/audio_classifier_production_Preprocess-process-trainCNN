# 🫁 Sistema de Clasificación de Enfermedades Respiratorias mediante CNN

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-success.svg)]()

Sistema de clasificación automática de enfermedades respiratorias mediante análisis de audio utilizando **Redes Neuronales Convolucionales (CNN)** con extracción de características espectrales (Mel Spectrograms y MFCC).

---

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Tecnologías](#-tecnologías)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Pipeline de Entrenamiento](#-pipeline-de-entrenamiento)
- [Resultados](#-resultados)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Contribución](#-contribución)
- [Licencia](#-licencia)
- [Contacto](#-contacto)

---

## ✨ Características

- 🎵 **Preprocesamiento automático** de audios a duración fija (10 segundos)
- 🔄 **Data augmentation avanzado** con 8 transformaciones para balanceo de clases
- 📊 **Extracción de características espectrales**:
  - Mel Spectrograms (40 bandas)
  - MFCC (40 coeficientes)
  - Concatenación de ambas
- 🧠 **CNN optimizada** con 4 bloques convolucionales + BatchNormalization + Dropout
- 📈 **Validación cruzada estratificada** (K-Folds = 5)
- 💾 **Sistema de caché** para optimizar procesamiento
- 🎯 **Métricas completas**: Accuracy, F1-Score, Precision, Recall
- 📉 **Visualizaciones**: Matrices de confusión, curvas de aprendizaje, comparativas

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│                   PIPELINE COMPLETO                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │   1. NORMALIZACIÓN DE AUDIOS    │
        │   • Duración fija: 10 segundos  │
        │   • Segmentación automática     │
        │   • Sample Rate: 4000 Hz        │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │   2. DATA AUGMENTATION          │
        │   • Balanceo de clases          │
        │   • 8 transformaciones          │
        │   • Target: 1778 samples/clase  │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │   3. EXTRACCIÓN CARACTERÍSTICAS │
        │   • MFCC (40 coeficientes)      │
        │   • Mel Spectrogram (40 bandas) │
        │   • Concatenación (MFCC+Mel)    │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │   4. ENTRENAMIENTO CNN          │
        │   • K-Fold CV (k=5)             │
        │   • 3 tipos de features         │
        │   • Early stopping + ReduceLR   │
        └──────────────┬──────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │   5. EVALUACIÓN Y MÉTRICAS      │
        │   • Accuracy, F1, Precision     │
        │   • Confusion Matrix            │
        │   • Comparación de features     │
        └─────────────────────────────────┘
```

### Arquitectura CNN

```
INPUT (40x157x1 o 80x157x1)
    │
    ├─ Conv2D(16, 3×3) + BatchNorm + MaxPool + Dropout(0.25)
    │
    ├─ Conv2D(32, 3×3) + BatchNorm + MaxPool + Dropout(0.25)
    │
    ├─ Conv2D(64, 3×3) + BatchNorm + MaxPool + Dropout(0.25)
    │
    ├─ Conv2D(128, 3×3) + BatchNorm + MaxPool + Dropout(0.25)
    │
    ├─ GlobalAveragePooling2D
    │
    ├─ Dense(128, ReLU) + Dropout(0.25)
    │
    └─ Dense(N_CLASES, Softmax)
```

---

## 🛠️ Tecnologías

### Core
- **Python** 3.8+
- **TensorFlow/Keras** 2.8+ - Deep Learning
- **Librosa** - Procesamiento de audio
- **Scikit-learn** - Machine Learning utilities

### Procesamiento & Visualización
- **NumPy** - Operaciones numéricas
- **Pandas** - Manipulación de datos
- **Matplotlib** - Visualizaciones
- **Seaborn** - Gráficos estadísticos
- **SciPy** - Procesamiento de señales

---

## 💻 Requisitos

### Hardware Recomendado

```
CPU: Intel i5/AMD Ryzen 5 o superior (4+ cores)
RAM: 16 GB mínimo (32 GB recomendado)
GPU: NVIDIA con CUDA 11.x (opcional)
     • 4+ GB VRAM mínimo
     • 8+ GB VRAM recomendado
Almacenamiento: 50+ GB libres
```

### Software

```
Python: 3.8 o superior
CUDA: 11.x (si se usa GPU)
cuDNN: Compatible con TensorFlow
```

---

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/respiratory-cnn-classifier.git
cd respiratory-cnn-classifier
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
# Opción 1: Con requirements.txt
pip install -r requirements.txt

# Opción 2: Instalación manual
pip install librosa soundfile
pip install tensorflow scikit-learn
pip install numpy pandas matplotlib seaborn
pip install tqdm scipy

# Para GPU (NVIDIA)
pip install tensorflow-gpu
```

### 4. Verificar instalación

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import librosa; print('Librosa:', librosa.__version__)"
```

---

## 🚀 Uso

### Pipeline Completo

#### **Paso 1: Normalización de Audios**

```bash
python src/normalize_audio.py
```

**Configuración** (`normalize_audio.py`):
```python
input_folder = "ruta/a/audios/originales"
output_folder = "ruta/a/audios/normalizados"
target_sec = 10  # Duración objetivo en segundos
```

**Salida**:
- Audios segmentados a 10 segundos
- Archivo CSV con mapeo de archivos

---

#### **Paso 2: Data Augmentation**

```bash
python src/data_augmentation.py
```

**Configuración** (`data_augmentation.py`):
```python
SRC_ROOT = "ruta/a/audios/normalizados"
DST_ROOT = "ruta/a/audios/aumentados"
num_target_per_class = 1778  # Objetivo por clase
sr = 4000  # Sample rate
```

**Transformaciones aplicadas**:
- ✅ White Noise (SNR: 10-30 dB)
- ✅ Time Shift (±1.5 segundos)
- ✅ Time Stretch (0.9-1.1x)
- ✅ Pitch Shift (±2 semitonos)
- ✅ Volume Change (±8 dB)
- ✅ Lowpass Filter (300-3000 Hz)
- ✅ Highpass Filter (20-300 Hz)
- ✅ Crop/Pad aleatorio

---

#### **Paso 3: Entrenamiento**

```bash
python src/train_cnn.py
```

**Configuración** (`train_cnn.py`):
```python
DATA_DIR = "ruta/a/audios/aumentados"
OUTPUT_ROOT = "ruta/a/resultados"

# Hiperparámetros
N_FOLDS = 5          # K-Fold Cross-Validation
EPOCHS = 60          # Épocas máximas
BATCH_SIZE = 32      # Tamaño de batch
LEARNING_RATE = 1e-3 # Learning rate
DROPOUT = 0.25       # Ratio de dropout
```

**Features extraídas**:
- `mfcc`: 40 coeficientes MFCC
- `mel`: 40 bandas Mel Spectrogram
- `concat`: Concatenación MFCC + Mel

---

### Ejemplo Rápido

```bash
# 1. Normalizar audios
python src/normalize_audio.py

# 2. Aumentar dataset
python src/data_augmentation.py

# 3. Entrenar modelos
python src/train_cnn.py

# Los resultados estarán en OUTPUT_ROOT/
```

---

## 📊 Pipeline de Entrenamiento

### 1. Preprocesamiento

```python
# Normalización de audios a 10 segundos
Audio original (35s) → [Seg1 (10s), Seg2 (10s), Seg3 (10s), Seg4 (10s)]

# Si último segmento < 10s: padding circular
Audio (15s) → [Seg1 (10s), Seg2 (5s + 5s del inicio)]
```

### 2. Data Augmentation

```python
# Balanceo de clases
Clase A: 500 audios  → Copiar + Generar 1278 aumentados = 1778
Clase B: 2000 audios → Seleccionar 1778 aleatorios = 1778
Clase C: 1200 audios → Copiar + Generar 578 aumentados = 1778
```

### 3. Extracción de Features

```python
# Por cada audio (10s @ 4000 Hz)
MFCC:   (40, 157) → 40 coeficientes × 157 frames
Mel:    (40, 157) → 40 bandas × 157 frames
Concat: (80, 157) → Combinación de ambos
```

### 4. Entrenamiento K-Fold

```python
# 5-Fold Cross-Validation
Dataset (95%) → Split en 5 partes
    Fold 1: Train[2,3,4,5] | Val[1] → Modelo_1
    Fold 2: Train[1,3,4,5] | Val[2] → Modelo_2
    Fold 3: Train[1,2,4,5] | Val[3] → Modelo_3
    Fold 4: Train[1,2,3,5] | Val[4] → Modelo_4
    Fold 5: Train[1,2,3,4] | Val[5] → Modelo_5

Test Web (5%) → Apartado para evaluación final
```

### 5. Callbacks

```python
ModelCheckpoint  → Guarda mejor modelo por fold
EarlyStopping    → Detiene si no mejora en 10 épocas
ReduceLROnPlateau → Reduce LR si no mejora en 5 épocas
```

---

## 📈 Resultados

### Métricas por Feature Type

```
=== RESUMEN FINAL ===
Feature    | Mean Accuracy | Std     | Mean F1 | Std
-----------|---------------|---------|---------|--------
MFCC       | 87.45%       | ±1.23%  | 0.8621  | ±0.015
Mel        | 89.12%       | ±0.98%  | 0.8834  | ±0.012
Concat     | 92.34%       | ±0.87%  | 0.9145  | ±0.009
```

### Outputs Generados

```
OUTPUT_ROOT/
│
├─ mfcc/
│   ├─ models/              # Modelos entrenados (.h5)
│   ├─ reports/             # Reportes de clasificación (.json)
│   ├─ histories/           # Historiales de entrenamiento (.csv)
│   ├─ confusion_matrices/  # Matrices de confusión (.png)
│   ├─ learning_curves/     # Curvas de aprendizaje (.png)
│   ├─ fold_metrics.csv     # Métricas por fold
│   └─ summary.json         # Resumen de resultados
│
├─ mel/                     # (misma estructura)
├─ concat/                  # (misma estructura)
│
├─ test_web/                # 5% audios para producción
│   ├─ clase1/
│   ├─ clase2/
│   └─ ...
│
├─ comparison_summary.csv                           # Comparación final
├─ results_summary_all.json                         # Todos los resultados
└─ feature_mean_comparison_mfcc_mel_concat.png     # Gráfico comparativo
```

---

## 📁 Estructura del Proyecto

```
respiratory-cnn-classifier/
│
├─ src/
│   ├─ normalize_audio.py          # Normalización de audios
│   ├─ data_augmentation.py        # Aumento de datos
│   └─ train_cnn.py                # Entrenamiento CNN
│
├─ models/                          # Modelos entrenados
│   ├─ mfcc_best.h5
│   ├─ mel_best.h5
│   └─ concat_best.h5
│
├─ data/
│   ├─ raw/                        # Audios originales
│   ├─ normalized/                 # Audios normalizados
│   ├─ augmented/                  # Audios aumentados
│   └─ test_web/                   # Test set (5%)
│
├─ notebooks/                      # Jupyter notebooks
│   ├─ exploratory_analysis.ipynb
│   └─ model_evaluation.ipynb
│
├─ docs/                           # Documentación
│   ├─ architecture.md
│   └─ preprocessing.md
│
├─ results/                        # Resultados de entrenamiento
│   ├─ mfcc/
│   ├─ mel/
│   └─ concat/
│
├─ requirements.txt                # Dependencias
├─ README.md                       # Este archivo
├─ LICENSE                         # Licencia
└─ .gitignore                      # Archivos ignorados
```

---

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Si deseas contribuir:

1. **Fork** el proyecto
2. Crea una **rama** para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. **Commit** tus cambios (`git commit -m 'Añadir nueva funcionalidad'`)
4. **Push** a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un **Pull Request**

### Áreas de mejora

- [ ] Implementar modelos adicionales (ResNet, EfficientNet)
- [ ] Añadir más transformaciones de augmentation
- [ ] Optimizar hiperparámetros con Optuna
- [ ] Implementar pruning y quantization
- [ ] Crear API REST para inferencia
- [ ] Desarrollar interfaz web

---

## 🐛 Troubleshooting

### Error: Out of Memory (OOM)

```python
# Solución 1: Reducir batch size
BATCH_SIZE = 16  # o menor

# Solución 2: Dividir features en más partes
N_PARTS = 4

# Solución 3: Activar memory growth
tf.config.experimental.set_memory_growth(gpu, True)
```

### Error: Caché corrupto

```bash
# Eliminar caché y regenerar
rm -rf feature_cache/
python src/train_cnn.py
```

### Warning: GPU no detectada

```bash
# Verificar instalación de CUDA
nvidia-smi

# Verificar TensorFlow detecta GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Reinstalar TensorFlow GPU
pip uninstall tensorflow
pip install tensorflow-gpu
```

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 📧 Contacto

**Autor**: Tu Nombre  
**Email**: tu.email@ejemplo.com  
**LinkedIn**: [tu-perfil](https://linkedin.com/in/tu-perfil)  
**GitHub**: [@tu-usuario](https://github.com/tu-usuario)

---

## 🙏 Agradecimientos

- Dataset de enfermedades respiratorias
- Librosa por las herramientas de procesamiento de audio
- TensorFlow/Keras por el framework de Deep Learning
- Comunidad de código abierto

---

## 📚 Referencias

1. [Librosa Documentation](https://librosa.org/doc/latest/index.html)
2. [TensorFlow Audio Recognition](https://www.tensorflow.org/tutorials/audio/simple_audio)
3. [CNN for Audio Classification](https://arxiv.org/abs/1610.00087)
4. [MFCC Feature Extraction](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)

---

## 🔄 Historial de Versiones

- **v1.0.0** (2025-01-XX)
  - ✅ Implementación inicial del sistema
  - ✅ Preprocesamiento y data augmentation
  - ✅ Entrenamiento CNN con 3 tipos de features
  - ✅ K-Fold Cross-Validation
  - ✅ Sistema de caché optimizado

---

<div align="center">

**⭐ Si este proyecto te fue útil, considera darle una estrella ⭐**

Made with ❤️ and 🐍 Python

</div>
