# 🫁 Clasificación de Neumonía con CNN — Semana Santa

---

## 📁 Estructura del proyecto

```
Pneumonia_EasterWeek/
├── Pneumonia_Clasificacion_CNN.ipynb       ← Modelo CON Data Augmentation
├── GUIA_SUSTENTACION.md                    ← Preguntas y respuestas para sustentar
├── WithoutDataAugmentation/
│   └── Pneumonia_SinAugmentation.ipynb    ← Modelo SIN Data Augmentation (comparación)
└── README.md
```

---

## 📦 Dataset (NO incluido en el repositorio)

El dataset **Chest X-Ray Pneumonia** no está incluido por su tamaño (~1.2 GB).

**Descárgalo desde Kaggle:**  
👉 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Una vez descargado, descomprime dentro de esta carpeta con la siguiente estructura:

```
Pneumonia_EasterWeek/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

---

## 🚀 Orden de ejecución

1. Ejecuta primero `Pneumonia_Clasificacion_CNN.ipynb` (CON augmentation)
2. Luego ejecuta `WithoutDataAugmentation/Pneumonia_SinAugmentation.ipynb`
3. El segundo notebook carga automáticamente el historial del primero para comparar

---

## 🎯 Objetivo

Comparar el impacto del **Data Augmentation** en la clasificación binaria:
- **NORMAL** → pulmón sano
- **PNEUMONIA** → neumonía bacteriana o viral

| | CON Augmentation | SIN Augmentation |
|---|---|---|
| Overfitting | Reducido | Mayor |
| Generalización | Mejor | Peor |
| Val/Test Accuracy | Mayor | Menor |

---

## 🛠️ Dependencias

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn plotly pillow
```
