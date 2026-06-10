# Pipeline Modular de Visión Artificial para el Análisis Biomecánico y Estimación del Rendimiento (RIR)

Este repositorio contiene el flujo de trabajo computacional desarrollado para la monitorización biomecánica en tiempo real, la segmentación cinemática y la estimación de la fatiga intra-serie (RIR - *Reps in Reserve*) en diferentes ejercicios de gimnasio.

Debido a la alta demanda de memoria gráfica ($VRAM$) asociada al uso simultáneo de modelos de *Deep Learning* de última generación, el proyecto se ha desacoplado modularmente en dos fases secuenciales optimizadas para su ejecución eficiente tanto en entornos interactivos cloud (**Google Colab**) como en infraestructuras de computación de alto rendimiento (**HPC / Clúster SNOW**).

Para una descripción completa de la metodología, marco teórico, experimentación y resultados, puedes consultar la **memoria completa del TFG** adjunta en este repositorio (`docs/Memoria_TFG_Eric_Matas_Perez.pdf`).

---

## 🚀 Arquitectura del Sistema

El sistema tiene como objetivo realizar un análisis biomecánico y una estimación del rendimiento deportivo a partir de vídeos convencionales de ejercicios de gimnasio, eliminando la necesidad de sensores externos o dispositivos específicos de medición. Para ello se ha diseñado una **arquitectura modular y secuencial** basada en técnicas de visión por computador y estimación de pose humana en tres dimensiones, donde la salida de cada etapa constituye la entrada de la siguiente.

El flujo de procesamiento comienza con la adquisición del vídeo del ejercicio y finaliza con la obtención de métricas biomecánicas, variables cinemáticas y una estimación de la proximidad al fallo muscular mediante el cálculo del **RIR** (*Repetitions In Reserve*). El pipeline se articula en dos notebooks secuenciales:

1. **Módulo de Segmentación** (`Pruebas_SAM3_GPU_issues.ipynb`): Aplica una máscara de segmentación sobre el vídeo raw para aislar al atleta del fondo, generando los frames limpios que alimentarán el módulo siguiente.

2. **Módulo de Análisis** (`Pruebas_SAM3D_GPU_issues_[ejercicio].ipynb`): A partir de los frames segmentados, estima la pose 3D del sujeto (127 keypoints corporales) y ejecuta dos submódulos en paralelo:
   - **Submódulo Biomecánico:** calcula ángulos articulares, ROM y métricas de técnica, determinando si la ejecución es correcta o presenta desviaciones respecto a los umbrales establecidos.
   - **Submódulo de Rendimiento:** extrae variables cinemáticas repetición a repetición (VMC, tiempos de fase, desplazamiento) para estimar el RIR restante hasta el fallo muscular.

> 💡 **¿Qué ejercicio quieres analizar?** Este repositorio incluye notebooks independientes y preconfigurados para cada ejercicio disponible (Press de Banca, Jalón al Pecho, etc.). Puedes encontrarlos listados directamente en la raíz del repositorio — simplemente elige el `.ipynb` correspondiente al movimiento que vayas a registrar.

---

## 🛠️ Requisitos de Software y Dependencias

El entorno requiere **Python 3.10+** junto con soporte para aceleración por hardware mediante **NVIDIA CUDA**.

### Stack Tecnológico Principal:

* **Framework:** `PyTorch` (compatible con CUDA)
* **Visión Artificial:** `OpenCV`, `Ultralytics` (Modelos YOLO / SAM), `SAM 3` (segmentación cinemática), `SAM-3D-Body` (estimación de pose 3D)
* **Modelos 3D y Geometría:** `MoGe`, `detectron2`, `roma`, `pyrender`
* **Entrenamiento y Configuración:** `pytorch-lightning`, `yacs`
* **Procesamiento de Datos:** `NumPy`, `SciPy`, `Pandas`, `decord`
* **Acceso a Modelos Preentrenados:** `huggingface_hub` *(requiere cuenta y token de acceso en [huggingface.co](https://huggingface.co))*
* **Visualización:** `Matplotlib`

> ⚠️ **Nota sobre dependencias:** La lista completa de instalación está integrada celda a celda dentro de cada notebook. Las dependencias listadas aquí son las principales; ejecuta los bloques `pip install` del notebook en orden para garantizar la compatibilidad.

---

## 💻 Configuración del Entorno de Ejecución

### Opción A: Ejecución Local o Cloud (Google Colab)

Cada Notebook incluye las directivas necesarias para instalar las dependencias directamente en la máquina virtual asignada mediante bloques independientes con comandos `pip`. No obstante, para entornos locales con **Conda**, puedes inicializar el entorno base ejecutando:

```bash
conda create -n pipeline_biomecanica python=3.10 -y
conda activate pipeline_biomecanica
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python pandas scipy matplotlib decord
pip install huggingface_hub roma pytorch-lightning yacs pyrender braceexpand
pip install 'git+https://github.com/facebookresearch/sam3.git'
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install git+https://github.com/microsoft/MoGe.git
```

> 🔑 **Autenticación en Hugging Face (obligatorio para el Módulo 2):** El notebook de estimación de pose descarga automáticamente el checkpoint `facebook/sam-3d-body-dinov3` desde el Hub. Para ello necesitas un token de acceso:
> 1. Crea una cuenta en [huggingface.co](https://huggingface.co) si no tienes una.
> 2. Genera un token en *Settings → Access Tokens*.
> 3. Cuando el notebook ejecute `login()`, introduce tu token en el campo que aparece.

> 📁 **Nota sobre rutas de Google Drive:** Las rutas de entrada/salida en los notebooks apuntan a una estructura dentro de tu Drive (`MyDrive/TFG/...`). Ajusta la variable `ruta_base` en las primeras celdas de configuración para que coincida con tu propia organización de carpetas antes de ejecutar.

---

### Opción B: Ejecución en Clúster de Alto Rendimiento (HPC SNOW - DTIC UPF)

Para procesar series de vídeo completas, sets de datos voluminosos o tareas de cálculo intensivo pesado que requieran nodos dedicados, se debe utilizar la infraestructura del clúster mediante el gestor de trabajos **Slurm**.

#### Conexión al clúster y transferencia de datos:

```bash
# Conectarse vía SSH (sustituye por tu usuario)
ssh tu_usuario_upf@hpc.s.upf.edu

# Enviar el vídeo que quieres analizar desde tu máquina local al clúster
scp "C:\Ruta\A\Tu\Video\mi_video_entrenamiento.mp4" tu_usuario_upf@hpc.s.upf.edu:~/tfg/data/
```

#### Carga de módulos del entorno e inicio de sesión interactiva:

Para depurar o realizar pruebas rápidas en un nodo con GPU dedicada de alta capacidad (como las **NVIDIA L40S** de 200GB asignadas):

```bash
# Solicitar un nodo interactivo con GPU en la partición corta
srun --partition=short --gres=gpu:l40s:1 --mem=200G --pty bash

# Cargar las herramientas del clúster necesarias
module load Miniconda3/4.9.2
module load CUDA/11.4.3
```

#### Activación de Conda y despliegue automatizado en segundo plano (Batch):

Para ejecuciones largas se recomienda no usar el modo interactivo, sino lanzar un script job (`.sh`) a la cola utilizando `sbatch`:

```bash
# Inicializar la shell para Conda (si es la primera vez)
eval "$(conda shell.bash hook)"

# Activar el entorno virtual preconfigurado del clúster
conda activate sam3_env

# Lanzar el job de segmentación (Módulo SAM3)
sbatch run_sam3.sh

# Monitorizar los logs de error y salida en tiempo real
tail -f slurm.FINAL.XXXXXXX.err
```

#### Descarga de los resultados procesados:

Una vez completado el procesamiento del esqueleto 3D (Módulo SAM3D Body), puedes empaquetar los resultados en un archivo comprimido y descargarlos a tu ordenador local para su visualización:

```bash
# En el clúster: Comprimir la carpeta de resultados generada
cd ~/tfg/data
zip -r resultados_ejercicio.zip keypoints_salida/ pose_frames_salida/

# En tu máquina local: Descargar el vídeo final analizado y el .zip
scp tu_usuario_upf@hpc.s.upf.edu:~/tfg/data/video_analizado_final.mp4 "C:\Users\TuUsuario\Desktop\"
scp tu_usuario_upf@hpc.s.upf.edu:~/tfg/data/resultados_ejercicio.zip "C:\Users\TuUsuario\Desktop\"
```

---

## 📖 Guía de Uso del Pipeline Modular

> ⚠️ **Nota Importante sobre Rutas:** Antes de ejecutar el código, localice las variables de entrada/salida (paths) en las primeras celdas de configuración de los notebooks y sustituya las rutas de ejemplo por sus archivos correspondientes (`tu_video.mp4`, `directorio_salida/`, etc.).

### Paso 1: Segmentación del Atleta

Abra y ejecute el notebook `Pruebas_SAM3_GPU_issues.ipynb`.

* **Propósito:** Lee el vídeo raw del ejercicio y aplica un modelo de segmentación para generar una máscara sobre el atleta, aislándolo del fondo del gimnasio. No se realiza ningún análisis biomecánico en esta etapa — el objetivo exclusivo es producir frames limpios para el módulo siguiente.
* **Entrada:** Vídeo en formato `.mp4` / `.avi`.
* **Salida:** Frames segmentados frame a frame y vídeo preprocesado, listos para ser consumidos por el Módulo de Análisis.

### Paso 2: Estimación de Pose 3D, Análisis Biomecánico y Estimación del RIR

Abra y ejecute el notebook del ejercicio correspondiente (ej. `Pruebas_SAM3D_GPU_issues_Press.ipynb` para Press de Banca). *(El flujo es análogo para el resto de ejercicios disponibles en el repositorio.)*

* **Propósito:** A partir de los frames segmentados del paso anterior, el modelo estima la pose tridimensional del sujeto extrayendo **127 keypoints corporales**. Sobre esta representación 3D se ejecutan en paralelo dos submódulos:

**Submódulo Biomecánico:**
* Calcula ángulos articulares clave (flexión de codo, hombro, extensión de cadera, etc.), ratio de agarre e inclinación del torso, según los umbrales específicos de cada ejercicio.
* Determina si el rango de movimiento (ROM) es completo y si la técnica se ejecuta dentro de los parámetros de seguridad establecidos.
* Genera alertas visuales dinámicas en el vídeo de salida cuando se detecta una desviación técnica.

**Submódulo de Estimación del Rendimiento:**
* Calcula la velocidad media concéntrica (VMC) repetición a repetición, junto con los tiempos de fase excéntrica y concéntrica y otras métricas ejecución y fatiga intra-serie.
* Compara las diferentes métricas de cada repetición con la de la repetición más veloz de la serie. Con estas, se estima matemáticamente las **repeticiones en reserva (RIR)** restantes hasta el fallo muscular.

---

## 📊 Estructura de Resultados Generados

El pipeline produce dos tipos de salida:

**Vídeo analizado** con overlay gráfico unificado que contiene:
* **Esqueleto 3D:** renderizado en verde si la técnica es correcta, o en rojo si se detecta alguna desviación respecto a los umbrales biomecánicos.
* **Panel de métricas en tiempo real** (esquina superior), con los siguientes valores por repetición:
  * Número de repetición actual.
  * **VMC** ($m/s$): Velocidad Media Concéntrica.
  * **ROM** ($m$): Rango de Movimiento de la barra o segmento analizado.
  * **T.Ecc / T.Conc** ($s$): Duración de las fases excéntrica y concéntrica.
  * **RIR ESTIMADO:** predicción numérica de la proximidad al fallo muscular.

**Archivo de keypoints** (`.npy`) con las coordenadas 3D de los 127 landmarks corporales para cada frame, disponible para análisis posteriores o exportación.

---

## 📌 Solución de Problemas Comunes (GPU Issues)

Ambos notebooks están diseñados específicamente para mitigar los errores de desbordamiento de memoria de vídeo (**OOM** - *Out of Memory*) de PyTorch. Si experimenta cuelgues durante el procesamiento de secuencias largas:

* Asegúrese de que no hay otra sesión activa ocupando el núcleo de la GPU (Buda / Cuda).
* En el clúster HPC, verifique la asignación de su nodo y asegúrese de limpiar la caché de memoria intermedia ejecutando `torch.cuda.empty_cache()` (instrucción ya integrada al final de los bucles de lectura de frames).

---

## 👤 Autoría y Contacto

Este proyecto ha sido desarrollado como **Trabajo de Final de Grado (TFG)** en el marco del Grado en **Ingeniería en Sistemas Audiovisuales** de la Universitat Pompeu Fabra (UPF).

| | |
|---|---|
| **Estudiante** | Eric Matas Pérez |
| **Supervisión** | Antonio Agudo — *Professor & Research Scientist, UPF* |
| **Titulación** | Grado en Ingeniería en Sistemas Audiovisuales |
| **Institución** | Universitat Pompeu Fabra (UPF) · Barcelona |
| **Contacto** | [eric.matas01@estudiant.upf.edu](mailto:eric.matas01@estudiant.upf.edu) |
| **Memoria TFG** | [`📄 docs/Memoria_TFG_Eric_Matas.pdf`](docs/Memoria_TFG_Eric_Matas.pdf) |

> Para cualquier duda técnica sobre el pipeline, sugerencias de mejora o solicitudes de colaboración, no dudes en abrir un [Issue](../../issues) en este repositorio o contactar directamente por correo.
