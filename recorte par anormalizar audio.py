import os
import wave
import contextlib
import numpy as np
import csv

# Carpeta raíz con audios originales
input_folder = r"D:\dataset pf\COPIA\organized_by_type\FILTERED_AUDIO"
# Carpeta donde guardaremos los audios normalizados
output_folder = r"D:\dataset pf\COPIA\organized_by_type\FILTERED_AUDIO_10s"
# Archivo CSV de mapeo
csv_file = os.path.join(output_folder, "mapping_normalized_audio.csv")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Duración fija en segundos
target_sec = 10

# Lista para guardar información del CSV
csv_rows = []

# Función para normalizar y dividir un audio
def process_wav(file_path, output_dir, target_sec=10):
    with contextlib.closing(wave.open(file_path,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        audio = f.readframes(frames)
        audio = np.frombuffer(audio, dtype=np.int16)
    
    target_len = target_sec * rate
    total_len = len(audio)
    segments = []

    # Dividir en segmentos de target_len
    start = 0
    segment_index = 1
    while start < total_len:
        end = start + target_len
        segment = audio[start:end]
        # Si el segmento es menor a target_len, rellenar con inicio del mismo audio
        if len(segment) < target_len:
            remaining = target_len - len(segment)
            segment = np.concatenate([segment, audio[:remaining]])
        segments.append(segment)

        # Guardar cada segmento como WAV
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_seg{segment_index}.wav")
        with wave.open(output_path, 'w') as wf:
            wf.setnchannels(1)           # Mono
            wf.setsampwidth(2)           # 16 bits = 2 bytes
            wf.setframerate(rate)
            wf.writeframes(segment.tobytes())

        # Guardar info para CSV
        rel_subfolder = os.path.relpath(output_dir, output_folder)
        csv_rows.append([file_path, output_path, rel_subfolder, segment_index])

        start += target_len
        segment_index += 1

# Recorrer todas las subcarpetas
for subdir, dirs, files in os.walk(input_folder):
    rel_path = os.path.relpath(subdir, input_folder)
    out_subdir = os.path.join(output_folder, rel_path)
    if not os.path.exists(out_subdir):
        os.makedirs(out_subdir)
    
    for file in files:
        if file.lower().endswith(".wav"):
            file_path = os.path.join(subdir, file)
            try:
                process_wav(file_path, out_subdir, target_sec)
            except Exception as e:
                print(f"No se pudo procesar {file_path}: {e}")

# Guardar CSV
with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["original_file", "normalized_file", "subfolder", "segment_index"])
    writer.writerows(csv_rows)

print("Normalización completada y CSV generado en:", csv_file)
