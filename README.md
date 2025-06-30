### Estimación de Dirección de Arribo (DOA) con Arrays de Micrófonos

Este proyecto implementa una simulación completa para la estimación de dirección de arribo (DOA) de una fuente acústica utilizando técnicas basadas en TDOA (Time Difference of Arrival) con arrays lineales de micrófonos. Se utilizan RIRs (Respuestas al Impulso de Sala) simuladas mediante pyroomacoustics, estimaciones con distintos métodos de correlación cruzada y un análisis detallado del error angular.

### Estructura General

El flujo del proyecto está dividido en 4 módulos principales, diseñados para una ejecución secuencial y modular:

simulation.py: Generación de RIRs y metadatos.

main.py: Simulación de señales, estimación de TDOA/DOA y exportación de resultados.

load_signal.py: Utilidades para cargar señales WAV.

generate_plots.py: Análisis y visualización del error de DOA.

1. simulation.py
Este script es el punto de partida del proyecto. Se encarga de generar conjuntos de RIRs (Room Impulse Responses) para distintas configuraciones geométricas y de reverberación de una sala, así como de almacenar los metadatos asociados.

Características Principales:
Definición de Sala: Modela salas cúbicas de 10x10x10 metros.

Parámetros Variables: Permite la variación de los siguientes parámetros para una exploración exhaustiva:

RT60: Tiempos de reverberación de 0.05 s (anecoica), 0.7 s (reverberación media) y 2.8 s (alta reverberación).

Ángulo de Llegada (Azimut): Desde 0° a 180° con incrementos de 10°.

Ángulo de Elevación: 0°, 15°, 30° y 60°.

Cantidad de Micrófonos: Configuración de arrays con 4 y 8 micrófonos.

Validación Espacial: Incorpora validaciones para asegurar que la fuente acústica y los micrófonos no estén ubicados demasiado cerca entre sí o de las paredes de la sala, previniendo artefactos en las RIRs.

Cálculo de Ángulo Real: Calcula el ángulo real de llegada (azimut y elevación) respecto al centro del array de micrófonos, sirviendo como verdad fundamental para la evaluación del rendimiento.

Salida de Datos:

RIRs: Las RIRs generadas se guardan individualmente en archivos .wav para cada micrófono y configuración.

Metadatos: Todos los metadatos de la simulación, incluyendo las configuraciones de la sala, las posiciones de la fuente y los micrófonos, y los ángulos reales, se consolidan en el archivo rir_dataset_user_defined/simulation_metadata.csv.

2. main.py
Este script es el corazón del proceso de estimación de DOA. Toma las RIRs y metadatos generados por simulation.py para simular señales reverberantes, estimar los TDOAs y DOAs, y finalmente registrar los errores para un análisis posterior.

Requisitos:
Una señal anecoica (ejemplo: p336_007.wav) que será convolucionada con las RIRs. Asegúrate de colocar este archivo en la ubicación esperada por el script o ajustar la ruta en el código.

El archivo simulation_metadata.csv generado previamente por simulation.py.

Funcionalidad Detallada:
Convolución de Señales: Convoluciona la señal anecoica de entrada con cada RIR generada, creando señales acústicas simuladas que emulan la propagación del sonido en el entorno reverberante.

Adición de Ruido: Añade ruido blanco a las señales resultantes para alcanzar el SNR (Signal-to-Noise Ratio) especificado (por defecto, 90 dB), permitiendo evaluar la robustez de los algoritmos en entornos ruidosos.

Estimación de TDOAs: Calcula las diferencias de tiempo de llegada entre pares de micrófonos utilizando una variedad de métodos de correlación cruzada, lo que permite una comparación exhaustiva de su rendimiento:

CC (Correlación Cruzada Clásica)

GCC-PHAT (Generalized Cross-Correlation with Phase Transform)

GCC-SCOT (Generalized Cross-Correlation with Smoothed Coherence Transform)

ML (Maximum Likelihood)

GCC-ROTH (Generalized Cross-Correlation with Roth weighting)

Cálculo de DOAs: A partir de los TDOAs estimados, se calculan las Direcciones de Arribo para pares de micrófonos adyacentes. Posteriormente, se estima un DOA promedio para cada array completo.

Salida de Datos:

Resultados Individuales: Se guardan los resultados detallados por cada par de micrófonos.

Resultados Promediados: Los resultados promediados por array, incluyendo los DOAs estimados y los errores, se almacenan en el archivo full_experiment_results.csv, el cual será utilizado para el análisis y visualización.

3. load_signal.py
Este módulo auxiliar proporciona utilidades esenciales para la carga y preprocesamiento de señales de audio en formato WAV, asegurando la consistencia de los datos en todo el pipeline. Es un módulo de soporte que será invocado por otros scripts (principalmente main.py).

Funcionalidad:
load_signal_from_wav(filepath, target_fs):

Carga de Señal: Carga la señal de audio desde la ruta del archivo especificada.

Validación de Frecuencia de Muestreo: Comprueba que la frecuencia de muestreo de la señal cargada coincida con la target_fs esperada, lanzando una advertencia o error si hay una discrepancia.

Conversión a Mono: Si la señal es estéreo, se convierte a mono para simplificar el procesamiento.

Normalización: Detecta y normaliza la señal si se encuentra clipping, previniendo distorsiones.

4. generate_plots.py
Este script es el módulo final del proyecto, dedicado al análisis y visualización de los resultados. Genera una serie de gráficos que permiten comprender el impacto de los diferentes parámetros de la simulación en el error promedio de la estimación de DOA.

Requisitos:
El archivo full_experiment_results.csv generado por main.py, que contiene todos los resultados de los experimentos.

Gráficos Generados (ejemplos visuales):
Cada gráfico presenta líneas de error promedio con su desviación estándar (errorbar='sd') para mostrar la variabilidad de los resultados, y utiliza hue='tdoa_method_for_avg_doa' para comparar el rendimiento de los distintos métodos de TDOA.

Error promedio DOA vs Ángulo de elevación

Error promedio DOA vs Distancia fuente-array

Error promedio DOA vs Separación entre micrófonos

Error promedio DOA vs Cantidad de micrófonos

Error promedio DOA vs Tiempo de reverberación (RT60)

Error promedio DOA vs Frecuencia de muestreo

Error promedio DOA vs SNR para distintas frecuencias de muestreo

Requisitos del Sistema
Para ejecutar este proyecto, asegúrese de tener instaladas las siguientes dependencias de Python:

Python 
ge 3.8

Paquetes:

numpy

pandas

scipy

soundfile

pyroomacoustics

matplotlib

seaborn

Instalación Recomendada:
La forma más sencilla de instalar todas las dependencias es utilizando pip y el archivo requirements.txt (si está disponible en el repositorio del proyecto):

Bash

pip install -r requirements.txt
Si no tienes un archivo requirements.txt, puedes instalarlos individualmente:

Bash

pip install numpy pandas scipy soundfile pyroomacoustics matplotlib seaborn
Instrucciones para Correr el Pipeline Completo
Para ejecutar el proyecto de principio a fin y generar todos los datos y gráficos, siga los siguientes pasos en el orden especificado, ya que cada script depende de los resultados del anterior:

1. Generar RIRs y Metadatos (simulation.py)
Este es el primer paso y el más fundamental. Ejecute el script simulation.py desde su terminal. Este proceso creará:

Los archivos .wav de las RIRs para cada configuración de micrófono dentro del directorio rir_dataset_user_defined/.

El archivo rir_dataset_user_defined/simulation_metadata.csv, que contiene todos los metadatos esenciales de la simulación.

Bash

python simulation.py
2. Simular Señales y Estimar DOA (main.py)
Una vez que simulation.py ha finalizado, puede ejecutar main.py. Este script leerá los metadatos y las RIRs generadas, simulará las señales con reverberación y ruido, y realizará las estimaciones de TDOA y DOA.

Asegúrese de que el archivo de señal anecoica (ej. p336_007.wav) esté disponible en la ubicación que el script main.py espera.

Este paso generará el archivo full_experiment_results.csv, que contiene todos los errores y resultados para el análisis posterior.

Bash

python main.py
3. Generar Gráficos de Análisis (generate_plots.py)
Finalmente, con el archivo full_experiment_results.csv ya creado por main.py, ejecute el script generate_plots.py. Este script cargará los resultados y generará automáticamente los gráficos descritos en la sección 4, visualizando el rendimiento de los métodos de DOA bajo diversas condiciones.

Bash

python generate_plots.py
Nota importante: Los módulos como load_signal.py (y cualquier otro módulo auxiliar como tdoa.py o doa.py si forman parte de la implementación interna) no deben ser ejecutados directamente. Sus funciones son llamadas automáticamente por main.py (o por otros scripts del pipeline) cuando son necesarias.