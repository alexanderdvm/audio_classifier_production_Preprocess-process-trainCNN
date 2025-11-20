import os
import wave
import contextlib
import numpy as np
from scipy.signal import butter, filtfilt, spectrogram
import matplotlib.pyplot as plt

# Carpeta raíz de audios
input_folder = r"D:\dataset pf\COPIA\organized_by_type\FILTERED_AUDIO_10s"
# Carpeta donde guardar imágenes
output_img_folder = r"D:\dataset pf\COPIA\organized_by_type\SPECTRO_IMAGES"

# Parámetros
fs_target = 4000               # frecuencia de muestreo
cutoff = int(0.95 * (fs_target/2))  # filtro paso bajo seguro (95% de Nyquist)
nfft = 256                      # tamaño de ventana para espectrograma
noverlap = 128                  # solapamiento

# Crear carpeta de salida
if not os.path.exists(output_img_folder):
    os.makedirs(output_img_folder)

# Funciones auxiliares
def read_wav_resampled(file_path, fs_target=4000):
    with contextlib.closing(wave.open(file_path,'r')) as f:
        n_channels = f.getnchannels()
        rate = f.getframerate()
        frames = f.readframes(f.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        if n_channels > 1:
            audio = audio[::n_channels]  # convertir a mono
    audio = audio.astype(np.float32)/32768.0
    if rate != fs_target:
        num_samples = int(len(audio) * fs_target / rate)
        audio = np.interp(np.linspace(0, len(audio), num_samples), np.arange(len(audio)), audio)
    return audio

def lowpass_filter(audio, fs, cutoff=cutoff, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, audio)

def audio_to_spectrogram(audio, fs, nfft=256, noverlap=128):
    f, t, Sxx = spectrogram(audio, fs, nperseg=nfft, noverlap=noverlap)
    Sxx = 10 * np.log10(Sxx + 1e-10)
    return Sxx

# Listas para guardar datos y etiquetas
X = []
y = []
labels_dict = {}
label_counter = 0

# Recorrer todas las subcarpetas
for subdir, dirs, files in os.walk(input_folder):
    if subdir == input_folder:
        continue

    subfolder_name = os.path.basename(subdir)
    if subfolder_name not in labels_dict:
        labels_dict[subfolder_name] = label_counter
        label_counter += 1
    label_idx = labels_dict[subfolder_name]

    # Crear carpeta de salida para esta subcarpeta
    out_subdir = os.path.join(output_img_folder, subfolder_name)
    if not os.path.exists(out_subdir):
        os.makedirs(out_subdir)

    for file in files:
        if file.lower().endswith(".wav"):
            file_path = os.path.join(subdir, file)
            try:
                # Leer y filtrar
                audio = read_wav_resampled(file_path, fs_target)
                filtered_audio = lowpass_filter(audio, fs_target, cutoff)
                
                # Espectrograma
                Sxx = audio_to_spectrogram(filtered_audio, fs_target, nfft, noverlap)
                Sxx_norm = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min())
                X.append(Sxx_norm)
                y.append(label_idx)

                # Guardar imagen
                plt.figure(figsize=(6,4))
                plt.pcolormesh(Sxx_norm, cmap='viridis')
                plt.axis('off')
                img_name = os.path.splitext(file)[0] + "_SPEC.png"
                plt.savefig(os.path.join(out_subdir, img_name), bbox_inches='tight', pad_inches=0)
                plt.close()
            except Exception as e:
                print(f"No se pudo procesar {file_path}: {e}")

# Convertir a arrays numpy
X = np.array(X)
y = np.array(y)

print("Número de muestras:", X.shape[0])
print("Dimensión de cada espectrograma:", X.shape[1:])
print("Número de clases:", len(labels_dict))
print("Diccionario de etiquetas:", labels_dict)
print("Espectrogramas guardados en:", output_img_folder)
