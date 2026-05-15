# Guía de Sustentación — Introducción a PyTorch
**Semana 6 | Redes Neuronales**

---

## Contexto del Taller

Este taller es una introducción práctica al framework **PyTorch** usando el dataset **Dog Breeds** (8 razas de perros en imágenes JPG a color). El flujo completo cubre:

1. Fundamentos de tensores
2. Carga y preprocesamiento de datos
3. Definición de una red neuronal convolucional (CNN)
4. Entrenamiento y evaluación
5. Guardado y despliegue del modelo

El resultado final es una CNN (**DogNet**) que clasifica 8 razas de perros entrenada desde cero con solo 541 imágenes.

> **Notebook:** `Introduccion_PyTorch_DogBreeds.ipynb`

---

## Dataset Dog Breeds
[text](../../../../dog-breeds)
| Raza | Imágenes |
|------|----------|
| beagle | 83 |
| bulldog | 76 |
| dalmatian | 51 |
| german-shepherd | 76 |
| husky | 71 |
| labrador-retriever | 50 |
| poodle | 56 |
| rottweiler | 78 |
| **Total** | **541** |

**Diferencias clave vs MNIST:**
- Color (3 canales RGB) vs escala de grises (1 canal)
- Sin split pre-hecho → dividimos con `random_split` 80/20
- Muchas menos imágenes (541 vs 70,000) → necesita data augmentation
- Más difícil de clasificar → accuracy esperado 40–70% (no 99%)

---

## Conceptos Clave

### 1. ¿Qué es un Tensor?
Un tensor es la estructura de datos fundamental en PyTorch. Es una generalización de matrices a N dimensiones:

| Concepto | Ejemplo | Dimensiones |
|----------|---------|-------------|
| Escalar  | `5`     | 0D          |
| Vector   | `[1, 2, 3]` | 1D      |
| Matriz   | `[[1,2],[3,4]]` | 2D  |
| Imagen   | `(canales, alto, ancho)` | 3D |
| Batch de imágenes | `(batch, canales, alto, ancho)` | 4D |

**Diferencia clave con NumPy:** los tensores pueden vivir en la GPU para acelerar cómputo.

---

### 2. CUDA y GPU
CUDA es el framework de NVIDIA que permite ejecutar cómputo en GPU. PyTorch lo usa automáticamente si está disponible.

```python
torch.cuda.is_available()  # True si hay GPU disponible
device = torch.device("cuda:0")
tensor.to(device)  # mueve el tensor a GPU
```

**Por qué importa:** el entrenamiento en GPU puede ser 10–100x más rápido que en CPU para redes grandes.

**GPU usada en este taller:** NVIDIA GeForce RTX 5060 Ti (16 GB VRAM, arquitectura Blackwell, Driver 596.49, CUDA 13.2)

**Instalación correcta de PyTorch con CUDA:**
`pip install torch` instala por defecto la versión **CPU-only** (`+cpu`). Para habilitar la GPU hay que indicar explícitamente el índice de CUDA:

```bash
# Para RTX 5060 Ti (Blackwell) — requiere CUDA 12.8 mínimo
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

| Versión instalada | CUDA disponible |
|-------------------|----------------|
| `torch 2.12.0+cpu` | `False` (solo CPU) |
| `torch 2.11.0+cu128` | `True` (RTX 5060 Ti detectada) |

**Patrón correcto para código portable (GPU o CPU):**
```python
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
```
Nunca escribir `torch.device("cuda:0")` directamente sin el guard, porque lanza `RuntimeError` en máquinas sin GPU.

---

### 3. Pipeline de Datos

```
dog-breeds/ (carpetas) → ImageFolder → Subset(train/val) → DataLoader → Batches
```

| Componente | Función |
|------------|---------|
| `datasets.ImageFolder` | Lee imágenes desde carpetas; el nombre de carpeta = clase |
| `transforms.Compose` | Aplica transformaciones en cadena |
| `random_split` / `Subset` | Divide el dataset en train (80%) y val (20%) |
| `DataLoader` | Divide en batches y baraja automáticamente |

**`ImageFolder`** asigna índices automáticamente en orden alfabético:
`beagle=0, bulldog=1, dalmatian=2, german-shepherd=3, husky=4, labrador-retriever=5, poodle=6, rottweiler=7`

---

### 4. Arquitectura CNN (clase `DogNet`)

```
Entrada (3×64×64)  ← imágenes color 64×64
    ↓ Conv2d(3→32, k=3, padding=1) + ReLU + MaxPool(2)  → 32×32×32
    ↓ Conv2d(32→64, k=3, padding=1) + ReLU + MaxPool(2) → 64×16×16
    ↓ Conv2d(64→128, k=3, padding=1) + ReLU + MaxPool(2)→ 128×8×8
    ↓ Flatten → 8192
    ↓ Dropout(0.5) + Linear(8192→512) + ReLU → 512
    ↓ Dropout(0.3) + Linear(512→8) → 8 clases (logits)
```

**¿Por qué `padding=1` con `kernel=3`?** Mantiene las dimensiones espaciales iguales antes del MaxPool, haciendo el cálculo predecible: 64→32→16→8 (cada MaxPool divide por 2 exactamente).

**¿Por qué Dropout?** Con solo 541 imágenes, el modelo tiende a memorizar (sobreajuste). Dropout desactiva neuronas aleatoriamente durante el entrenamiento, forzando a la red a aprender representaciones más robustas.

**¿Por qué 3 canales en conv1?** Las imágenes son RGB (3 canales). MNIST era escala de grises (1 canal). El primer número en `Conv2d(3, 32, ...)` debe coincidir con los canales de entrada.

---

### 5. Función de Pérdida: CrossEntropyLoss

`CrossEntropyLoss` = `log_softmax` + `NLLLoss` en un solo paso. Es la función estándar para clasificación multi-clase.

```
logits → softmax → probabilidades
       → log     → log-probabilidades
       → NLLLoss → pérdida
```

**Ventaja sobre NLLLoss separada:** no hay que agregar `log_softmax` al final del modelo. El `forward` retorna logits crudos y `CrossEntropyLoss` se encarga del resto.

**Intuición:** si el modelo asigna probabilidad 0.01 a la clase correcta, la pérdida es alta (−log(0.01) ≈ 4.6). Si asigna 0.99, la pérdida es casi cero (−log(0.99) ≈ 0.01).

---

### 6. Optimizador Adam

```python
opt = optim.Adam(model.parameters(), lr=1e-4)
```

Adam es una variante de descenso por gradiente que adapta la tasa de aprendizaje por parámetro. Es más robusto que SGD puro para empezar.

---

### 7. Ciclo de Entrenamiento

El ciclo tiene **3 pasos fijos** que se repiten por cada batch:

```python
loss.backward()   # 1. Calcular gradientes (retropropagación)
opt.step()        # 2. Actualizar pesos
opt.zero_grad()   # 3. Limpiar gradientes (evitar acumulación)
```

**Error común:** olvidar `zero_grad()` hace que los gradientes se acumulen y el entrenamiento diverge.

---

## Líneas de Código Más Importantes

### Detectar GPU y definir device
```python
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
```
**Por qué:** el código debe funcionar tanto con GPU como sin ella. Todos los tensores y el modelo se mueven a `device` para coherencia.

---

### Cargar dataset desde carpetas con ImageFolder
```python
full_dataset = datasets.ImageFolder("./dog-breeds", transform=transforms)
```
**Por qué:** `ImageFolder` lee la estructura de carpetas y asigna automáticamente etiquetas numéricas. No hay que crear ningún CSV ni mapeo manual.

---

### Split 80/20 con índices compartidos
```python
indices = torch.randperm(n_total).tolist()
train_ds = Subset(full_dataset_train, indices[:n_train])
val_ds   = Subset(full_dataset_val,   indices[n_train:])
```
**Por qué:** se cargan dos instancias del dataset con diferentes transforms (augmentation para train, solo resize para val) pero se usan los mismos índices para garantizar que la misma imagen no aparezca en los dos sets.

---

### Conv2d con padding=1
```python
self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
```
**Por qué:** `padding=1` con `kernel=3` mantiene las dimensiones espaciales iguales antes del MaxPool. Esto hace que el cálculo de dimensiones sea predecible: 64→32→16→8.

---

### Definir DogNet con nn.Module
```python
class DogNet(nn.Module):
    def __init__(self, num_classes=8):
        super(DogNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 canales RGB
        ...
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        ...
        return x  # logits crudos — sin log_softmax
```
**Por qué:** heredar de `nn.Module` activa tracking de parámetros y gradientes. Sin `log_softmax` final porque `CrossEntropyLoss` lo incluye internamente.

---

### Dropout para regularización
```python
self.dropout1 = nn.Dropout(0.5)
x = self.dropout1(x)
```
**Por qué:** con solo 541 imágenes el riesgo de sobreajuste es alto. Dropout desactiva el 50% de las neuronas aleatoriamente durante el entrenamiento, forzando representaciones más distribuidas.

---

### Normalización ImageNet
```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```
**Por qué:** estos son la media y desviación estándar del dataset ImageNet (millones de imágenes naturales). Normalizar con estos valores centra la distribución de pixeles y hace que el gradiente fluya mejor.

---

### Modo eval y no_grad en validación
```python
model.eval()
with torch.no_grad():
    val_loss, val_metric = loss_epoch(model, loss_func, val_dl)
```
**Por qué:** `model.eval()` desactiva el Dropout (en validación queremos usar todas las neuronas). `torch.no_grad()` evita calcular gradientes, ahorrando ~50% de memoria y tiempo.

---

### Guardar y cargar pesos
```python
torch.save(model.state_dict(), "dognet_weights.pt")
_model.load_state_dict(torch.load("dognet_weights.pt", map_location=device))
```
**Por qué:** `state_dict` guarda solo los pesos. `map_location=device` permite cargar pesos entrenados en GPU en una máquina sin GPU (y viceversa).

---

### Predicción y visualización de probabilidades
```python
x = img_tensor.unsqueeze(0).to(device)   # (3,64,64) → (1,3,64,64)
output = _model(x)                        # logits: (1, 8)
pred_idx = output.argmax(dim=1).item()
probs = torch.softmax(output, dim=1)[0]  # convertir logits a probabilidades
```
**Por qué:** el modelo espera un batch. `argmax` da el índice de la clase predicha. `softmax` convierte logits a probabilidades interpretables (suman 1).

---

## Resultados Obtenidos

Entrenamiento real ejecutado sobre **NVIDIA RTX 5060 Ti**, 20 épocas, 432 imágenes de train / 109 de validación:

| Época | Train Loss | Val Loss | Val Accuracy |
|-------|-----------|----------|-------------|
| 1  | 2.0901 | 2.0481 | 12.8% |
| 5  | 1.4227 | 1.6300 | 40.4% |
| 10 | 0.9194 | 1.1992 | 63.3% |
| 15 | 0.5448 | 1.0966 | 70.6% |
| 17 | 0.4454 | 0.9467 | **76.1%** |
| 20 | 0.3597 | 1.1543 | 73.4% |

**Mejor accuracy alcanzado: 76.1%** (época 17)

Con 541 imágenes entrenadas desde cero esto es un buen resultado. La brecha entre train loss (0.36) y val loss (1.15) al final indica algo de sobreajuste, normal con tan pocas imágenes.

La curva de pérdida muestra descenso claro en train. Si val loss sube mientras train loss baja → sobreajuste. El Dropout ayuda a mitigarlo pero no lo elimina del todo con un dataset tan pequeño.

**Inferencia:** el modelo predijo correctamente "rottweiler" con 100% de confianza en la imagen de prueba.

---

## Parámetros del Modelo DogNet

```
Conv2d-1 (3→32):    3×32×3×3 + 32 bias =    896 parámetros
Conv2d-2 (32→64):  32×64×3×3 + 64 bias =  18,496 parámetros
Conv2d-3 (64→128): 64×128×3×3 + 128 bias = 73,856 parámetros
Linear-1 (8192→512):          8192×512+512 = 4,194,816 parámetros
Linear-2 (512→8):                  512×8+8 =     4,104 parámetros
────────────────────────────────────────────────────────────────
Total:                                      ~4,292,168 parámetros
```

La mayoría de los parámetros están en `fc1` (Linear 8192→512), porque conecta el mapa de características aplanado con la primera capa densa. Esto es típico en CNNs clásicas.

---

## Preguntas Frecuentes en Sustentación

**¿Por qué usar PyTorch y no Keras/TensorFlow?**
PyTorch es más Pythónico, facilita el debugging (grafo dinámico) y es preferido en investigación. Keras es más alto nivel pero menos flexible.

**¿Qué diferencia hay entre `nn.Sequential` y `nn.Module`?**
`Sequential` apila capas en línea recta. `Module` permite flujos personalizados: bifurcaciones, conexiones residuales, reutilización de capas. `DogNet` usa `Module` porque necesita Dropout en posiciones específicas.

**¿Por qué `CrossEntropyLoss` en vez de `NLLLoss`?**
`CrossEntropyLoss = log_softmax + NLLLoss`. Usar CE es más simple porque no hay que agregar `log_softmax` al final del modelo. El modelo retorna logits crudos y la función de pérdida los procesa.

**¿Qué hace exactamente `backward()`?**
Recorre el grafo computacional en sentido inverso aplicando la regla de la cadena para calcular `∂Loss/∂w` para cada parámetro del modelo.

**¿Qué ocurre si no muevo el modelo a la GPU?**
Los datos en GPU y el modelo en CPU son incompatibles. PyTorch lanza `RuntimeError`. Por eso el patrón es: `model.to(device)` y `tensor.to(device)` antes de cada forward pass.

**¿Por qué `torch.cuda.current_device()` lanza error si hay GPU?**
Porque `pip install torch` instala la versión `+cpu` por defecto. Aunque la máquina tenga GPU, PyTorch no la detecta sin la build CUDA. La solución es instalar con `--index-url https://download.pytorch.org/whl/cu128`. Además, llamar funciones CUDA sin el guard `if torch.cuda.is_available()` lanza `RuntimeError` en máquinas sin GPU.

**¿Por qué el accuracy es bajo comparado con MNIST?**
Porque: (1) el dataset es mucho más pequeño (541 vs 70,000 imágenes), (2) las razas de perros son más similares entre sí que los dígitos, (3) entrenamos desde cero sin pesos pre-entrenados. Con transfer learning (ResNet, VGG) el accuracy subiría a 85–95%.

**¿Cómo se calcula que fc1 tiene 8192 entradas?**
- Entrada: 3×64×64
- Conv2d(3→32, p=1) + MaxPool(2): 32×32×32
- Conv2d(32→64, p=1) + MaxPool(2): 64×16×16
- Conv2d(64→128, p=1) + MaxPool(2): 128×8×8
- Flatten: 128 × 8 × 8 = **8192**

Con `padding=1` y `kernel=3`, las dimensiones espaciales se mantienen antes de cada MaxPool (que las divide por 2). Por eso el cálculo es limpio.

**¿Qué es el `state_dict`?**
Es un diccionario Python que mapea cada capa a sus tensores de parámetros (pesos y biases). Es la forma estándar de serializar modelos en PyTorch.

**¿Qué hace `ImageFolder`?**
Lee una carpeta donde cada subdirectorio es una clase, asigna índices automáticamente en orden alfabético, y carga las imágenes como PIL Images aplicando las transforms definidas.

**¿Por qué usar transforms distintos para train y val?**
El aumento de datos (RandomFlip, Rotation, ColorJitter) solo se aplica en entrenamiento para generar variedad artificial. En validación queremos medir el rendimiento real, sin distorsiones artificiales.

---

## Flujo Completo del Taller (Resumen Visual)

```
[dog-breeds/ carpetas → 8 razas, 541 imágenes JPG]
      ↓
[ImageFolder → asigna etiquetas por carpeta]
      ↓
[transforms: Resize(64)+ Flip + Rotation + ColorJitter + Normalize]
      ↓
[random_split 80/20 → 432 train / 109 val]
      ↓
[DataLoader → batches de 16]
      ↓
[DogNet: Conv(3→32)→Pool→Conv(32→64)→Pool→Conv(64→128)→Pool→FC→FC]
      ↓
[Logits crudos: 8 valores por imagen]
      ↓
[CrossEntropyLoss]  ←  compara con etiqueta real
      ↓
[loss.backward()]  →  gradientes
      ↓
[Adam.step()]  →  actualiza pesos
      ↓
[Repite por 20 épocas]
      ↓
[76.1% accuracy en validación — RTX 5060 Ti]
      ↓
[torch.save() → dognet_weights.pt]
      ↓
[Inferencia: predicción + probabilidades por clase]
```

---

## Comparativa MNIST vs Dog Breeds

| Aspecto | MNIST | Dog Breeds |
|---------|-------|------------|
| Imágenes | 70,000 | 541 |
| Clases | 10 (dígitos) | 8 (razas) |
| Canales | 1 (gris) | 3 (RGB) |
| Tamaño | 28×28 | 64×64 (redimensionado) |
| Split pre-hecho | Sí (60k/10k) | No → `random_split` |
| Data augmentation | Opcional | Necesaria |
| Pérdida | NLLLoss + log_softmax | CrossEntropyLoss |
| Accuracy obtenido | ~98.95% | **76.1%** (época 17) |
| Dificultad real | Baja | Media-Alta |
| GPU usada | Tesla P100 (Kaggle) | RTX 5060 Ti (local) |
| PyTorch build | `+cpu` (sin GPU local) | `2.11.0+cu128` |

---

## Ventaja Real de la GPU — Medición y Sustentación

### ¿Dónde se ve el beneficio en el código?

Hay tres líneas clave donde el código "cambia de carril" hacia la GPU:

```python
# 1. El modelo pasa a GPU — todos sus parámetros viven ahí
model = model.to(device)

# 2. Cada batch de imágenes y etiquetas sube a GPU antes del forward pass
xb = xb.to(device)
yb = yb.to(device)

# 3. El forward pass ocurre completamente en GPU (multiplicaciones matriciales en paralelo)
yb_h = model(xb)
```

Sin estas tres líneas, todo ocurre en CPU aunque haya GPU disponible. PyTorch no mueve datos automáticamente.

---

### Benchmark Real: RTX 5060 Ti vs CPU

Se midió en la misma máquina con el mismo modelo DogNet y las mismas imágenes 64×64:

#### Forward pass puro (inferencia, sin carga de datos)

| Batch size | CPU | GPU (RTX 5060 Ti) | Aceleración |
|-----------|-----|-------------------|-------------|
| 1 imagen  | 0.92 ms | 0.33 ms | **2.8x** |
| 16 imágenes | 6.77 ms | 0.36 ms | **18.9x** |
| 64 imágenes | 26.54 ms | 1.54 ms | **17.3x** |
| 256 imágenes | 119.22 ms | 6.02 ms | **19.8x** |

#### Paso completo de entrenamiento (forward + backward + actualización de pesos)

| Batch size | CPU | GPU (RTX 5060 Ti) | Aceleración |
|-----------|-----|-------------------|-------------|
| 16  | 22.24 ms | 2.26 ms | **9.9x** |
| 64  | 66.62 ms | 5.24 ms | **12.7x** |
| 256 | 276.98 ms | 17.89 ms | **15.5x** |

---

### ¿Por qué en las épocas completas solo fue 1.3x?

Durante el entrenamiento del notebook se midió 4.45s/época en CPU y 3.53s/época en GPU — solo 1.3x. La razón es el **cuello de botella de carga de datos**:

```
Tiempo total de una época =
    [carga de imágenes desde disco → CPU]   ← esto no lo acelera la GPU
  + [transferencia CPU → GPU]               ← overhead adicional
  + [cómputo en GPU]                        ← esto sí se acelera ~10-20x
```

Con 541 imágenes pequeñas (64×64), el cómputo GPU dura apenas ~2ms por batch, pero leer las imágenes del disco y aplicar transforms tarda mucho más. El tiempo total está dominado por la carga de datos, no por el cómputo.

**Regla general:** la GPU muestra su mayor ventaja cuando el cómputo domina el tiempo total (datasets grandes, imágenes grandes, modelos profundos). Con datasets pequeños el cuello de botella suele ser el disco o la CPU.

---

### ¿Cuándo la GPU marca una diferencia enorme?

| Escenario | CPU | GPU | Speedup típico |
|-----------|-----|-----|----------------|
| DogNet, 541 imgs, 64×64 | ~4.5s/época | ~3.5s/época | 1.3x |
| ResNet-50, ImageNet (1.2M imgs, 224×224) | ~8 horas/época | ~20 min/época | ~24x |
| GPT-2, entrenamiento de texto | semanas | días | 30–100x |
| Inferencia en producción (batch=256) | 119ms | 6ms | **19.8x** |

La ventaja real del taller es didáctica: el **mismo código** funciona en CPU o GPU cambiando solo la variable `device`. En producción con modelos reales (ResNet, BERT, etc.) la diferencia sería de horas vs días.

---

### Cómo verificarlo tú mismo en el notebook

En la **Celda 6**, al imprimir:
```
GPU disponible: NVIDIA GeForce RTX 5060 Ti
```
confirma que `device = cuda:0`.

En la **Celda 27**, al imprimir:
```
Dispositivo antes: cpu
Dispositivo después: cuda:0
```
confirma que el modelo está en GPU.

En la **Celda 30**, al hacer el primer forward pass, el tiempo de respuesta es notablemente menor que si se comenta `model.to(device)` y `xb.to(device)`.

Para medir tiempos manualmente se puede agregar esta celda al notebook:
```python
import time
# En GPU
t0 = time.perf_counter()
with torch.no_grad():
    _ = model(torch.randn(64, 3, 64, 64).to(device))
torch.cuda.synchronize()  # esperar que la GPU termine
print(f"GPU: {(time.perf_counter()-t0)*1000:.2f} ms")

# En CPU
model_cpu = model.to('cpu')
t0 = time.perf_counter()
with torch.no_grad():
    _ = model_cpu(torch.randn(64, 3, 64, 64))
print(f"CPU: {(time.perf_counter()-t0)*1000:.2f} ms")
```
Resultado esperado: CPU ~26ms, GPU ~1.5ms → **17x más rápido**.

---

## Entorno de Ejecución

| Componente | Detalle |
|------------|---------|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| VRAM | 16 GB |
| Driver | 596.49 |
| CUDA máx. soportado | 13.2 |
| PyTorch instalado | `2.11.0+cu128` |
| Torchvision | `0.26.0+cu128` |
| Python | 3.12.10 |
| Comando de instalación | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |

**Por qué `cu128` y no `cu121` u otro:**
La RTX 5060 Ti es arquitectura Blackwell (2025). Requiere CUDA 12.8 mínimo para funcionar. Builds más antiguas (`cu121`, `cu118`) no incluyen soporte para Blackwell y el entrenamiento fallaría o no reconocería la GPU.

---

---

## Explicación Línea a Línea de Cada Celda

---

### CELDA 5 — Importaciones y versiones

```python
import torch
```
Importa el framework PyTorch completo. Sin esto no existe nada: tensores, capas, optimizadores. Es la librería base de todo el taller.

```python
import torchvision
```
Extensión de PyTorch especializada en visión computacional. Provee `datasets.ImageFolder`, `transforms` y utilidades como `make_grid`. No viene incluida en `torch`, hay que importarla por separado.

```python
import numpy as np
```
Importa NumPy con el alias estándar `np`. Se usa para convertir tensores a arrays (necesario para mostrar imágenes con matplotlib, que no entiende tensores PyTorch).

```python
import matplotlib.pyplot as plt
```
Librería de visualización. Se usa para mostrar imágenes y graficar las curvas de pérdida y accuracy. `pyplot` es el módulo que contiene `imshow`, `plot`, `figure`, etc.

```python
%matplotlib inline
```
Directiva especial de Jupyter (no es Python puro, es un "magic command"). Le dice al notebook que muestre las gráficas directamente en la celda, sin abrir una ventana externa. Solo funciona en Jupyter.

```python
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
```
Verifica qué versiones están instaladas. Importante para reproducibilidad: la misma versión de PyTorch debe dar los mismos resultados. `__version__` es un atributo estándar de cualquier paquete Python.

---

### CELDA 6 — Detección de GPU

```python
if torch.cuda.is_available():
```
Pregunta a PyTorch si detecta una GPU compatible con CUDA. Retorna `True` o `False`. Si es `False`, puede ser porque: (1) no hay GPU, (2) el driver no está instalado, o (3) PyTorch fue instalado en versión CPU-only (`+cpu`).

```python
    device = torch.device("cuda:0")
```
Crea un objeto `device` que representa la GPU número 0 (la primera). `"cuda:0"` es la notación estándar: `cuda` = GPU NVIDIA, `:0` = índice (si tuvieras 2 GPUs serían `cuda:0` y `cuda:1`). Este objeto se usará luego para mover tensores y el modelo a la GPU.

```python
    print("GPU disponible:", torch.cuda.get_device_name(0))
    print("Número de GPUs:", torch.cuda.device_count())
```
`get_device_name(0)` retorna el nombre comercial de la GPU (en este caso "NVIDIA GeForce RTX 5060 Ti"). `device_count()` retorna cuántas GPUs hay disponibles.

```python
else:
    device = torch.device("cpu")
    print("GPU no disponible — usando CPU")
```
Si no hay GPU, `device` apunta a la CPU. El código es idéntico para ambos casos porque siempre usamos la variable `device`. Esto hace el notebook portable: funciona con o sin GPU sin cambiar ninguna otra línea.

---

### CELDA 8 — Creación de tensores

```python
x_rand = torch.rand(2, 3)
```
Crea un tensor de forma 2×3 (2 filas, 3 columnas) con valores aleatorios uniformes entre 0 y 1. `rand` = distribución uniforme U(0,1).

```python
print("Tensor aleatorio:\n", x_rand)
```
Imprime el tensor. `\n` es salto de línea para que el tensor se muestre en la siguiente línea del texto.

```python
x_ones = torch.ones(2, 3)
```
Crea un tensor 2×3 lleno de unos. El tipo por defecto es `torch.float32`.

```python
print("\nTensor de unos:\n", x_ones, "\nTipo:", x_ones.dtype)
```
`x_ones.dtype` muestra el tipo de dato del tensor. Por defecto `torch.float32` (número de coma flotante de 32 bits, 4 bytes por elemento).

---

### CELDA 9 — Tipos de datos en tensores

```python
x_int = torch.ones(2, 3, dtype=torch.int16)
```
Crea tensor de unos especificando el tipo: `int16` = entero de 16 bits (valores de -32768 a 32767). Se usa `dtype=` como argumento keyword.

```python
x_float = x_int.type(torch.float32)
```
Convierte el tensor a `float32`. `.type()` cambia el dtype sin modificar los valores. Es equivalente a un cast. Las capas de PyTorch esperan `float32` por defecto, por eso esta conversión es necesaria antes de pasar datos al modelo.

---

### CELDA 10 — Distribuciones de tensores

```python
r_val = torch.randn(3, 4)
```
Crea tensor 3×4 con distribución normal estándar: media=0, desviación estándar=1. `randn` (n = normal). Se usa mucho para inicializar pesos de prueba. Diferencia con `rand`: `rand` es U(0,1), `randn` es N(0,1) y puede tener valores negativos y mayores a 1.

```python
rng_val = torch.randint(0, 10, (3, 4))
```
Crea tensor 3×4 con enteros aleatorios. Primer argumento: límite inferior (0, incluido). Segundo: límite superior (10, **excluido**, o sea valores de 0 a 9). Tercero: la forma como tupla `(3, 4)`.

---

### CELDA 11 — Conversión Tensor ↔ NumPy

```python
x = torch.rand(2, 3)
y_np = x.numpy()
```
`.numpy()` convierte un tensor PyTorch en un array NumPy. **Condición:** el tensor debe estar en CPU. Si está en GPU hay que hacer `.cpu().numpy()`. La conversión comparte memoria: cambiar `y_np` cambia `x` también.

```python
arr = np.ones((2, 3), dtype=np.float32)
y_t = torch.from_numpy(arr)
```
`torch.from_numpy()` hace la conversión inversa: NumPy → PyTorch. También comparte memoria. El tipo `float32` en NumPy equivale a `torch.float32`.

---

### CELDA 12 — Mover tensores entre dispositivos

```python
x = torch.tensor([2.3, 5.8])
print("Dispositivo inicial:", x.device)
```
Por defecto todo tensor creado sin especificar dispositivo vive en CPU. `.device` muestra dónde está: `cpu` o `cuda:0`.

```python
x = x.to(device)
print("Después de .to(device):", x.device)
```
`.to(device)` mueve el tensor al dispositivo definido en la celda 6. Si `device = cuda:0`, el tensor pasa a la GPU. **Regla crítica:** todos los tensores que interactúen entre sí deben estar en el mismo dispositivo, o PyTorch lanza `RuntimeError`.

---

### CELDA 15 — Transformaciones e imports de datos

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
```
Importa las herramientas necesarias para el pipeline de datos. Se importan explícitamente (no `import torchvision` completo) para claridad.

```python
data_path = "./dog-breeds"
```
Ruta relativa al dataset. El `.` significa "carpeta actual", o sea la misma carpeta donde está el notebook. `ImageFolder` buscará aquí las subcarpetas de clases.

```python
train_transforms = transforms.Compose([
```
`Compose` encadena transformaciones: aplica la primera, pasa el resultado a la segunda, y así sucesivamente. Es como una tubería (pipeline) de procesamiento de imagen.

```python
    transforms.Resize((64, 64)),
```
Redimensiona la imagen a 64×64 píxeles. Las imágenes del dataset tienen tamaños distintos; la red neuronal requiere entrada de tamaño fijo. Se eligió 64×64 como balance entre calidad y velocidad de entrenamiento.

```python
    transforms.RandomHorizontalFlip(),
```
Con probabilidad 0.5 (por defecto) voltea la imagen horizontalmente. Un bulldog mirando a la derecha es el mismo bulldog mirando a la izquierda. Esto "duplica" la variedad del dataset artificialmente. Solo en entrenamiento.

```python
    transforms.RandomRotation(15),
```
Rota la imagen aleatoriamente entre -15° y +15°. El ángulo se elige al azar en ese rango cada vez que se carga la imagen. Ayuda al modelo a ser invariante a pequeñas rotaciones.

```python
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
```
Altera aleatoriamente el brillo y contraste de la imagen en ±20%. Simula variaciones de iluminación. Hace el modelo más robusto a condiciones de fotografía distintas.

```python
    transforms.ToTensor(),
```
Convierte la imagen PIL (Python Imaging Library) a tensor PyTorch. Hace dos cosas simultáneamente: (1) cambia el orden de dimensiones de HWC (Alto, Ancho, Canal) a CHW (Canal, Alto, Ancho), que es lo que espera PyTorch; (2) normaliza los valores de píxeles de [0, 255] a [0.0, 1.0].

```python
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
```
Normaliza cada canal (R, G, B) restando la media y dividiendo por la desviación estándar. Los valores `[0.485, 0.456, 0.406]` y `[0.229, 0.224, 0.225]` son la media y std calculadas sobre ImageNet (más de 1 millón de imágenes naturales). Esto hace que los valores de entrada al modelo estén centrados alrededor de 0, lo que estabiliza el entrenamiento y acelera la convergencia.

```python
val_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```
Para validación se omiten los augmentations (Flip, Rotation, ColorJitter). La razón: queremos medir el rendimiento real del modelo sobre imágenes "como vienen", no sobre versiones artificialmente modificadas. La normalización sí se mantiene porque el modelo la espera siempre.

---

### CELDA 16 — Cargar dataset con ImageFolder

```python
full_dataset_train = datasets.ImageFolder(data_path, transform=train_transforms)
full_dataset_val   = datasets.ImageFolder(data_path, transform=val_transforms)
```
Carga el **mismo dataset dos veces**, pero con transforms diferentes. `ImageFolder` lee la estructura de carpetas:
- Cada subcarpeta = una clase
- El nombre de la carpeta = nombre de la clase
- Las asigna en orden alfabético: beagle=0, bulldog=1, ..., rottweiler=7

Se necesitan dos instancias porque más adelante los índices de train usarán `full_dataset_train` (con augmentation) y los de val usarán `full_dataset_val` (sin augmentation).

```python
class_names = full_dataset_train.classes
```
`.classes` es una lista con los nombres de las clases en el orden que `ImageFolder` las asignó. Resultado: `['beagle', 'bulldog', 'dalmatian', 'german-shepherd', 'husky', 'labrador-retriever', 'poodle', 'rottweiler']`.

```python
num_classes = len(class_names)
```
Cuenta cuántas clases hay. Resultado: 8. Este número se usará luego para definir la capa de salida del modelo (`nn.Linear(512, num_classes)`).

```python
print("Total de imágenes:", len(full_dataset_train))
```
`len()` sobre un dataset de PyTorch retorna el número total de muestras: 541.

```python
print("\nÍndices:", full_dataset_train.class_to_idx)
```
`.class_to_idx` es un diccionario que muestra el mapeo nombre→índice: `{'beagle': 0, 'bulldog': 1, ...}`. Útil para saber qué número corresponde a qué raza al hacer predicciones.

---

### CELDA 17 — División train / validación

```python
n_total = len(full_dataset_train)   # 541
n_train = int(0.8 * n_total)        # 432
n_val   = n_total - n_train         # 109
```
Calcula tamaños del split 80/20. `int()` redondea hacia abajo para garantizar que sea entero. El resto va a validación. 80/20 es la división estándar cuando no hay conjunto de test separado.

```python
torch.manual_seed(42)
```
Fija la semilla del generador de números aleatorios de PyTorch. Con la misma semilla, `randperm` siempre produce la misma permutación. El número 42 es convención (irrelevante, cualquier entero funciona). Sin esto, cada ejecución daría una división diferente y los resultados no serían reproducibles.

```python
indices = torch.randperm(n_total).tolist()
```
`randperm(541)` genera una permutación aleatoria de los enteros 0–540 (como barajar una baraja de 541 cartas). `.tolist()` la convierte a lista Python, más fácil de usar para indexar.

```python
train_idx = indices[:n_train]   # primeros 432 índices
val_idx   = indices[n_train:]   # últimos 109 índices
```
Divide la lista de índices barajados en dos partes. Como `randperm` ya barajó, los índices son aleatorios y no hay sesgo de clase.

```python
train_ds = Subset(full_dataset_train, train_idx)
val_ds   = Subset(full_dataset_val,   val_idx)
```
`Subset` crea una "vista" del dataset original usando solo los índices indicados. Train usa `full_dataset_train` (con augmentation) y val usa `full_dataset_val` (sin augmentation), pero los índices son los mismos, así que nunca hay solapamiento entre conjuntos.

---

### CELDA 18 — Visualización del dataset

```python
viz_transforms = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
viz_dataset = datasets.ImageFolder(data_path, transform=viz_transforms)
```
Carga el dataset una tercera vez, solo con resize y ToTensor, sin normalización. Se hace así porque la normalización altera los colores para que matplotlib los muestre raros (valores fuera de [0,1]).

```python
seen = {}
for img, label in viz_dataset:
    if label not in seen:
        seen[label] = True
        samples.append(img)
```
Recorre el dataset y toma **una imagen por clase**. El diccionario `seen` actúa como un conjunto: si ya vimos esa etiqueta, la saltamos. Así obtenemos exactamente 8 imágenes (una por raza).

```python
grid = utils.make_grid(samples, nrow=4, padding=4)
```
`make_grid` toma una lista de tensores de imagen y los organiza en una cuadrícula. `nrow=4` = 4 imágenes por fila (2 filas en total para 8 imágenes). `padding=4` = 4 píxeles de espacio entre imágenes.

```python
npimg = grid.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
```
`make_grid` retorna tensor en formato CHW (canal, alto, ancho). `matplotlib.imshow` espera HWC (alto, ancho, canal). `np.transpose(npimg, (1, 2, 0))` reordena los ejes: eje-0 (C) pasa a posición 2, eje-1 (H) a posición 0, eje-2 (W) a posición 1.

---

### CELDA 19 — Comparar original vs augmentation

```python
viz_no_transform = datasets.ImageFolder(data_path)
img_pil, label = viz_no_transform[0]
```
Carga el dataset sin ninguna transformación. El `[0]` accede al primer elemento: una tupla `(imagen_PIL, etiqueta_int)`. La imagen es un objeto PIL (Python Imaging Library), no un tensor.

```python
img_augmented = train_transforms(img_pil)
```
Aplica todas las transformaciones de entrenamiento manualmente sobre la imagen PIL. Cada vez que se ejecuta esta línea el resultado puede ser diferente (porque las transformaciones son aleatorias).

```python
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_aug_show = (img_augmented * std + mean).clamp(0, 1)
```
**Denormalización:** deshace la operación `Normalize` para poder mostrar la imagen con colores reales. La fórmula es la inversa: `pixel_original = pixel_normalizado × std + mean`. `.view(3, 1, 1)` da forma (3,1,1) al tensor para que la multiplicación funcione por broadcasting (se aplica a cada canal C×H×W). `.clamp(0, 1)` recorta valores fuera del rango válido [0,1] que pueden aparecer por errores de punto flotante.

```python
plt.imshow(img_resized.permute(1, 2, 0))
```
`.permute(1, 2, 0)` es equivalente a `np.transpose` pero para tensores PyTorch. Convierte CHW → HWC para matplotlib.

---

### CELDA 21 — DataLoaders

```python
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=16, shuffle=False)
```
`DataLoader` envuelve el dataset y entrega los datos en batches durante el entrenamiento.
- `batch_size=16`: cada iteración entrega 16 imágenes. Más grande = gradientes más estables pero más memoria. Se eligió 16 por el tamaño del dataset.
- `shuffle=True` en train: baraja el orden de los datos al inicio de cada época, evitando que el modelo memorice el orden de llegada.
- `shuffle=False` en val: no importa el orden para evaluación, y mantenerlo fijo hace la validación reproducible.

```python
for xb, yb in train_dl:
    print("Forma del batch de imágenes:", xb.shape)  # torch.Size([16, 3, 64, 64])
    print("Forma del batch de etiquetas:", yb.shape) # torch.Size([16])
    break
```
Itera **un solo batch** para verificar las formas. `break` detiene el loop tras el primer batch. `xb.shape = [16, 3, 64, 64]`: 16 imágenes, 3 canales, 64×64 píxeles. `yb.shape = [16]`: 16 etiquetas enteras (una por imagen).

```python
print("Ejemplo de etiquetas:", [class_names[y] for y in yb[:4]])
```
List comprehension que convierte los 4 primeros índices numéricos (`yb[:4]`) a nombres de clase usando `class_names`.

---

### CELDA 23 — Demo: capa lineal

```python
input_tensor = torch.randn(16, 8192)
linear_layer = nn.Linear(8192, 512)
output = linear_layer(input_tensor)
print("Entrada:", input_tensor.shape, "→ Salida:", output.shape)
```
Demuestra cómo funciona una capa `Linear`. `nn.Linear(8192, 512)` crea una transformación matricial: `salida = entrada × W^T + b`, donde W tiene forma (512, 8192) y b tiene forma (512). Resultado: cualquier entrada de 8192 features se comprime a 512. Esto replica lo que hace `fc1` en `DogNet`.

---

### CELDA 24 — Demo: nn.Sequential

```python
modelo_simple = nn.Sequential(
    nn.Linear(8192, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 8)
)
```
`Sequential` encadena capas en orden. La salida de cada capa es la entrada de la siguiente. Más simple que `nn.Module`, pero no permite bifurcaciones ni lógica condicional. Se muestra aquí como comparación antes de definir `DogNet` con `Module`.

---

### CELDA 25 — Clase DogNet (el modelo principal)

```python
import torch.nn.functional as F
```
Importa las funciones de activación y pooling como funciones puras (sin estado). `F.relu`, `F.max_pool2d` no tienen parámetros entrenables, por eso se usan como funciones y no como capas (`nn.ReLU` sería equivalente pero como capa).

```python
class DogNet(nn.Module):
```
Define la clase del modelo heredando de `nn.Module`. Esta herencia activa automáticamente: (1) tracking de parámetros para el optimizador, (2) los métodos `.train()` y `.eval()`, (3) `.to(device)` para mover todo el modelo, (4) `state_dict()` para guardar/cargar pesos.

```python
    def __init__(self, num_classes=8):
        super(DogNet, self).__init__()
```
`__init__` es el constructor: se ejecuta al hacer `DogNet()`. `super().__init__()` llama al constructor de `nn.Module` y es **obligatorio**: sin esto el tracking de parámetros no funciona.

```python
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
```
Primera capa convolucional:
- `3`: canales de entrada (R, G, B)
- `32`: número de filtros (mapas de características) a aprender
- `kernel_size=3`: cada filtro es una ventana de 3×3 píxeles
- `padding=1`: agrega 1 píxel de borde en cada lado → la imagen de salida mantiene el mismo tamaño espacial (64×64) antes del MaxPool

```python
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
```
Cada capa duplica el número de filtros (32→64→128). Patrón estándar: más filtros en capas profundas porque detectan patrones más complejos (texturas, formas completas) que requieren más representaciones.

```python
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
```
Primera capa densa. La entrada `128 * 8 * 8 = 8192` viene del cálculo: imagen 64×64 después de 3 MaxPool(2) queda 8×8, con 128 filtros. `512` neuronas de salida.

```python
        self.fc2 = nn.Linear(512, num_classes)
```
Capa de salida. `num_classes=8`: produce 8 números (uno por raza). Estos son **logits** crudos, no probabilidades. `CrossEntropyLoss` los convierte a probabilidades internamente.

```python
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
```
`Dropout(p)`: durante el entrenamiento, desactiva aleatoriamente el `p*100%` de las neuronas en cada forward pass. `0.5` = 50% de neuronas apagadas. Esto fuerza a la red a no depender de neuronas específicas y aprende representaciones más distribuidas. **Solo activo en modo `.train()`, se desactiva automáticamente en `.eval()`**.

```python
    def forward(self, x):
```
Define el camino que sigue un batch de imágenes `x` a través del modelo. PyTorch llama a este método automáticamente cuando haces `model(x)`.

```python
        x = F.relu(self.conv1(x))
```
Dos operaciones en una línea:
1. `self.conv1(x)`: aplica la convolución → detecta patrones de bajo nivel (bordes, gradientes)
2. `F.relu(...)`: aplica la función de activación ReLU: `max(0, x)`. Convierte negativos en cero. Sin activación, las capas serían solo transformaciones lineales y toda la red equivaldría a una sola capa lineal.

```python
        x = F.max_pool2d(x, 2, 2)
```
Max pooling con ventana 2×2 y stride 2: toma el valor máximo de cada bloque de 2×2 píxeles. Reduce las dimensiones espaciales a la mitad (64→32). Efecto: hace la red invariante a pequeñas traslaciones y reduce la cantidad de cómputo.

```python
        x = x.view(-1, 128 * 8 * 8)
```
"Aplana" el tensor de 3D (128×8×8) a 1D (8192) por imagen. El `-1` le dice a PyTorch que calcule automáticamente esa dimensión (el batch size). Sin esto no se puede conectar a la capa lineal, que espera entrada 1D por muestra.

```python
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
```
Dropout antes de fc1 (50%), capa densa con ReLU, dropout antes de fc2 (30%), capa de salida. Se retornan logits crudos sin activación final: `CrossEntropyLoss` aplica softmax internamente y es más estable numéricamente que hacerlo manualmente.

---

### CELDA 26 — Instanciar el modelo

```python
model = DogNet(num_classes=8)
print(model)
```
Crea una instancia del modelo con pesos inicializados aleatoriamente (PyTorch usa inicialización de Kaiming por defecto para conv y Xavier para linear). `print(model)` muestra la arquitectura completa con configuraciones de cada capa.

---

### CELDA 27 — Mover modelo a GPU

```python
print("Dispositivo antes:", next(model.parameters()).device)
```
`model.parameters()` es un generador de todos los tensores de parámetros del modelo. `next()` toma el primero. `.device` muestra dónde está. Antes de moverlo: `cpu`.

```python
model = model.to(device)
print("Dispositivo después:", next(model.parameters()).device)
```
`.to(device)` mueve **todos los parámetros** del modelo al dispositivo. Después: `cuda:0`. Es crucial que modelo y datos estén en el mismo dispositivo.

---

### CELDA 28 — Resumen con torchsummary

```python
from torchsummary import summary
summary(model, input_size=(3, 64, 64))
```
`torchsummary` pasa un tensor falso de forma `(3, 64, 64)` por el modelo y registra la forma de salida de cada capa. Muestra: nombre de la capa, forma de salida, número de parámetros. Muy útil para verificar que las dimensiones son correctas antes de entrenar.

```python
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
```
Alternativa manual si torchsummary no está instalado. `p.numel()` = número de elementos del tensor de parámetros. `p.requires_grad` filtra solo los parámetros entrenables (excluye los congelados). Generator expression dentro de `sum()`.

---

### CELDA 30 — Función de pérdida y prueba

```python
loss_func = nn.CrossEntropyLoss(reduction="sum")
```
`CrossEntropyLoss` calcula la pérdida para clasificación multi-clase. `reduction="sum"`: suma las pérdidas de todas las muestras del batch (en vez de promediar). Se divide manualmente entre el total de muestras en `loss_epoch` para tener control explícito.

```python
for xb, yb in train_dl:
    xb = xb.to(device)
    yb = yb.to(device)
```
Mueve el batch de imágenes y etiquetas al mismo dispositivo que el modelo. Si no se hace, PyTorch lanza error.

```python
    out = model(xb)
```
Pasa el batch por el modelo: llama a `forward(xb)` internamente. `out` tiene forma `[16, 8]`: 16 imágenes, 8 logits por imagen.

```python
    loss = loss_func(out, yb)
    print(f"Pérdida inicial (primer batch): {loss.item():.4f}")
```
Calcula la pérdida comparando las predicciones `out` con las etiquetas reales `yb`. Al inicio (pesos aleatorios) la pérdida es alta. `.item()` extrae el valor escalar del tensor de pérdida a un float Python.

```python
    break
```
Sale del loop tras el primer batch. Solo queríamos verificar que todo funciona.

---

### CELDA 31 — Optimizador

```python
opt = optim.Adam(model.parameters(), lr=1e-3)
```
Crea el optimizador Adam.
- `model.parameters()`: le pasa todos los parámetros entrenables del modelo para que Adam sepa qué actualizar
- `lr=1e-3`: tasa de aprendizaje = 0.001. Controla el tamaño del paso en cada actualización. Muy grande → diverge. Muy pequeña → aprende lento. 0.001 es el default recomendado para Adam.

Adam (Adaptive Moment Estimation) mantiene una tasa de aprendizaje diferente por parámetro, ajustada según el historial de gradientes. Es más eficiente que SGD puro.

---

### CELDA 33 — Funciones auxiliares de entrenamiento

```python
def metrics_batch(target, output):
    pred = output.argmax(dim=1, keepdim=True)
```
`argmax(dim=1)` encuentra el índice del valor máximo a lo largo de la dimensión de clases (dim=1). Para un output de forma `[16, 8]`, retorna `[16, 1]` con el índice de la clase predicha por cada imagen. `keepdim=True` mantiene la dimensión para que la comparación funcione.

```python
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects
```
`target.view_as(pred)`: da a `target` la misma forma que `pred` para poder comparar elemento a elemento. `.eq()`: comparación elemento a elemento, retorna tensor de `True`/`False`. `.sum()`: cuenta los `True` (predicciones correctas). `.item()`: convierte a int Python.

```python
def loss_batch(loss_func, xb, yb, yb_h, opt=None):
    loss = loss_func(yb_h, yb)
    metric_b = metrics_batch(yb, yb_h)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), metric_b
```
`opt=None` como valor por defecto permite reutilizar esta función tanto para entrenamiento (pasando el optimizador) como para validación (sin optimizador). Si `opt is not None`:
- `loss.backward()`: calcula `∂Loss/∂w` para cada parámetro usando la regla de la cadena (retropropagación)
- `opt.step()`: actualiza los pesos: `w = w - lr × gradiente`
- `opt.zero_grad()`: limpia los gradientes acumulados. Si no se hace, el próximo backward **suma** gradientes a los anteriores → divergencia

```python
def loss_epoch(model, loss_func, dataset_dl, opt=None):
    loss = 0.0
    metric = 0.0
    len_data = len(dataset_dl.dataset)
```
`dataset_dl.dataset` accede al dataset subyacente del DataLoader. `len()` retorna el número total de muestras (432 para train, 109 para val). Se usa para promediar al final.

```python
    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        yb_h = model(xb)
        loss_b, metric_b = loss_batch(loss_func, xb, yb, yb_h, opt)
        loss   += loss_b
        metric += metric_b
    return loss / len_data, metric / len_data
```
Itera sobre todos los batches del DataLoader. Acumula pérdida y métrica. Al final divide entre el total de muestras para obtener el promedio por imagen (no por batch, que variaría con el batch size).

---

### CELDA 34 — Función de entrenamiento principal

```python
def train_val(epochs, model, loss_func, opt, train_dl, val_dl):
    history = {"train_loss": [], "val_loss": [], "accuracy": []}
```
Diccionario para guardar el historial de métricas por época. Se usa después para graficar las curvas de aprendizaje.

```python
    for epoch in range(epochs):
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)
```
`model.train()`: activa el modo entrenamiento. Activa Dropout (desactiva neuronas aleatoriamente) y cualquier BatchNorm en modo entrenamiento. `loss_epoch` con `opt` activa las actualizaciones de pesos.

```python
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)
```
`model.eval()`: cambia al modo evaluación. Desactiva Dropout (todas las neuronas activas). `torch.no_grad()`: desactiva el cálculo del grafo computacional para gradientes. Esto ahorra ~50% de memoria y hace la inferencia más rápida. No necesitamos gradientes en validación porque no actualizamos pesos.

```python
        accuracy = 100 * val_metric
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(accuracy)
        print(f"Epoch {epoch+1:2d}/{epochs} | ...")
    return history
```
`val_metric` es la fracción de aciertos (0.0 a 1.0). Se multiplica por 100 para expresarlo en porcentaje. `{epoch+1:2d}` formatea el número con ancho mínimo de 2 dígitos (para alinear la salida). `{train_loss:.4f}` muestra 4 decimales.

---

### CELDA 35 — Ejecutar entrenamiento

```python
num_epochs = 20
history = train_val(num_epochs, model, loss_func, opt, train_dl, val_dl)
```
Llama a la función de entrenamiento con 20 épocas. Una **época** = el modelo ve todas las imágenes de entrenamiento una vez (432 imágenes / 16 batch = 27 batches por época). 20 épocas × 27 batches = 540 actualizaciones de pesos totales.

---

### CELDA 36 — Gráficas de entrenamiento

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
```
Crea una figura con 2 subgráficas en una fila. `figsize=(12, 4)` = 12 pulgadas de ancho, 4 de alto.

```python
epochs_range = range(1, num_epochs + 1)
axes[0].plot(epochs_range, history["train_loss"], label="Train Loss")
axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss")
```
Grafica train loss y val loss en el mismo eje. Si `val_loss` sube mientras `train_loss` baja → el modelo está sobreajustando (memorizando en vez de aprender).

```python
axes[0].legend()
axes[0].grid(True)
```
`.legend()` muestra la leyenda con los labels asignados. `.grid(True)` dibuja la cuadrícula de fondo para facilitar la lectura.

```python
plt.tight_layout()
plt.show()
```
`tight_layout()` ajusta automáticamente los márgenes para que los títulos y etiquetas no se solapen. `show()` renderiza y muestra la figura.

---

### CELDA 38 — Guardar pesos

```python
weights_path = "dognet_weights.pt"
torch.save(model.state_dict(), weights_path)
```
`model.state_dict()`: retorna un diccionario Python con todos los tensores de parámetros del modelo (pesos y biases de cada capa). `torch.save()` lo serializa en el archivo `.pt` usando `pickle`. Solo guarda los **pesos**, no la arquitectura (la clase `DogNet` debe existir para cargarlos).

---

### CELDA 39 — Cargar pesos

```python
_model = DogNet(num_classes=8)
```
Crea un modelo nuevo con pesos aleatorios. La arquitectura debe ser idéntica a la que se guardó.

```python
_model.load_state_dict(torch.load(weights_path, map_location=device))
```
`torch.load()` deserializa el archivo `.pt`. `map_location=device` es crucial: si los pesos se guardaron en GPU pero la máquina actual no tiene GPU, `map_location=cpu` los carga en CPU sin error. Sin este argumento lanzaría error en esa situación.

`load_state_dict()` copia los tensores del archivo al modelo. Los nombres de las capas deben coincidir exactamente.

```python
_model = _model.to(device)
_model.eval()
```
Mueve el modelo cargado al dispositivo correcto y cambia a modo evaluación. Siempre hacer esto antes de usar el modelo para predicciones.

---

### CELDA 41 — Preparar imagen para predicción

```python
sample_idx = 5
img_tensor, true_label = val_ds[sample_idx]
```
Accede al elemento 5 del conjunto de validación. Retorna tupla `(tensor_imagen, etiqueta_int)`. El tensor ya tiene las transformaciones de val_transforms aplicadas (normalizado).

```python
mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std_t  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_show = (img_tensor * std_t + mean_t).clamp(0, 1)
```
Denormaliza la imagen para mostrarla con colores reales. La imagen en el dataset está normalizada (valores aproximadamente entre -2 y 2). Para matplotlib necesita estar entre 0 y 1. `.view(3, 1, 1)` da forma de columna para broadcasting correcto.

```python
plt.imshow(img_show.permute(1, 2, 0))
```
`.permute(1, 2, 0)` convierte CHW → HWC (PyTorch → matplotlib).

---

### CELDA 42 — Hacer la predicción

```python
x = img_tensor.unsqueeze(0).to(device)
```
`unsqueeze(0)` agrega una dimensión en la posición 0: `(3, 64, 64)` → `(1, 3, 64, 64)`. El modelo espera siempre un batch, aunque sea de tamaño 1. Sin este paso el modelo lanzaría error de dimensiones.

```python
with torch.no_grad():
    output = _model(x)
```
`torch.no_grad()` desactiva el cálculo de gradientes. En inferencia no se necesitan y ahorran memoria. `output` tiene forma `[1, 8]`: 1 imagen, 8 logits.

```python
pred_idx = output.argmax(dim=1).item()
```
`argmax(dim=1)`: índice del logit más alto entre las 8 clases → la clase predicha. `.item()` convierte el tensor a entero Python.

```python
probs = torch.softmax(output, dim=1)[0]
```
`softmax` convierte los 8 logits a probabilidades que suman 1. `[0]` selecciona la primera (y única) imagen del batch. Ahora `probs` es un tensor de 8 probabilidades.

```python
for i, (name, prob) in enumerate(zip(class_names, probs)):
    bar = '█' * int(prob.item() * 30)
    print(f"  {name:<22} {prob.item()*100:5.1f}% {bar}")
```
`zip(class_names, probs)` empareja cada nombre de clase con su probabilidad. `prob.item() * 30` convierte la probabilidad a un número de caracteres `█` para la barra visual. `{name:<22}` alinea el nombre a la izquierda en un campo de 22 caracteres. `{prob.item()*100:5.1f}` muestra la probabilidad en % con 1 decimal.

---

*Documento generado para preparación de sustentación — Semana 6*
