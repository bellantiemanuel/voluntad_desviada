import numpy as np
import matplotlib.pyplot as plt
import random
import time
from datetime import datetime
import os
import sounddevice as sd
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARÁMETROS GLOBALES
# ============================================================
N = 20
direcciones = ['arriba', 'abajo', 'izquierda', 'derecha']

SAMPLE_RATE = 44100
DURACION_SONIDO = 5.0   # segundos por evento
VOLUMEN = 0.3

# distancia máxima para normalizar
max_dist = np.sqrt((N/2)**2 + (N/2)**2)

# ============================================================
# EFECTOS AUDIBLES
# ============================================================
def aplicar_distorsion(signal, drive):
    drive = float(np.clip(drive, 0.0, 1.0))
    # Soft clipping con ganancia dependiente de drive
    ganancia = 1.0 + drive * 20.0
    out = np.tanh(signal * ganancia)
    # normalizar al rango original RMS
    m_in = np.sqrt(np.mean(signal**2) + 1e-12)
    m_out = np.sqrt(np.mean(out**2) + 1e-12)
    if m_out > 0:
        out = out * (m_in / m_out)
    return out

def aplicar_delay(signal, wet, delay_ms=180):
    if wet <= 0:
        return signal
    delay_samples = int(SAMPLE_RATE * delay_ms / 1000)
    out = np.copy(signal)
    fb = 0.25 + wet * 0.45  # feedback aumentado para mayor presencia
    # usar buffer con feedback
    buf = np.zeros(len(signal) + delay_samples + 10, dtype=signal.dtype)
    for n in range(len(signal)):
        read_idx = n
        write_idx = n + delay_samples
        buf[write_idx] += buf[read_idx] * fb + signal[n]
        out[n] = (1 - wet) * signal[n] + wet * buf[read_idx]
    return out

def aplicar_flanger(signal, wet, depth_ms=4.0, rate_hz=0.5):
    if wet <= 0:
        return signal
    depth_samples = int(depth_ms * SAMPLE_RATE / 1000)
    out = np.zeros_like(signal)
    t = np.arange(len(signal)) / SAMPLE_RATE
    lfo = (np.sin(2 * np.pi * rate_hz * t) + 1) / 2  # 0..1
    # indices dinámicos de delay
    for i in range(len(signal)):
        d = int(lfo[i] * depth_samples)
        if i - d >= 0:
            out[i] = signal[i] + 0.7 * signal[i - d]
        else:
            out[i] = signal[i]
    return (1 - wet) * signal + wet * out

# ============================================================
# FIGURA Y DIRECTORIOS
# ============================================================
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim([-N/2, N/2])
ax.set_ylim([-N/2, N/2])
ax.set_aspect('equal')
ax.set_title('Caminatas Aleatorias en 2D - Arte Generativo con Sonido Fourier')
plt.ion()

script_dir = os.path.dirname(os.path.abspath(__file__))
capturas_dir = os.path.join(script_dir, 'capturas')
if not os.path.exists(capturas_dir):
    os.makedirs(capturas_dir)

# ============================================================
# MAPA DE NOTAS (36 notas → 3 octavas) - desde C2 para más graves
# ============================================================
NOTAS_36 = []
frecuencia_base = 65.4064  # C2 ~ 65.4064 Hz (más graves que C3)
for i in range(36):
    freq = frecuencia_base * (2 ** (i / 12))
    NOTAS_36.append(freq)

def nota_desde_angulo(x, y):
    ang = np.degrees(np.arctan2(y, x))
    if ang < 0:
        ang += 360.0
    idx = int((ang / 360.0) * 36.0)
    idx = np.clip(idx, 0, 35)
    return NOTAS_36[idx], idx

# ============================================================
# SÍNTESIS: construir espectro y convertir a temporal
# ============================================================
def generar_espectro_musical(pos_x, pos_y):
    n_f = 512
    espectro = np.zeros(n_f, dtype=complex)

    distancia = np.sqrt(pos_x**2 + pos_y**2)
    d_norm = np.clip(distancia / max_dist, 0.0, 1.0)

    freq_fundamental, nota_idx = nota_desde_angulo(pos_x, pos_y)

    # Número de armónicos depende de la distancia (más lejos = más armónicos)
    num_armonicos = int(4 + d_norm * 18)  # 4..22

    for h in range(1, num_armonicos + 1):
        freq = freq_fundamental * h
        if freq < SAMPLE_RATE / 2:
            idx = int((freq / (SAMPLE_RATE / 2)) * (n_f // 2))
            if 0 < idx < n_f // 2:
                # magnitud decreciente, con algo de variación aleatoria
                magn = (1.0 / h) * np.interp(d_norm, [0, 1], [1.0, 0.35]) * random.uniform(0.8, 1.2)
                fase = random.uniform(0, 2 * np.pi)
                espectro[idx] += magn * np.exp(1j * fase)
                espectro[n_f - idx] += magn * np.exp(-1j * fase)

    # ruido espectral controlado por distancia (más lejos = más ruido)
    ruido_strength = d_norm * 0.7
    for k in range(1, n_f // 4):
        if random.random() < ruido_strength * 0.35:
            magn = random.random() * 0.12 * ruido_strength
            fase = random.uniform(0, 2 * np.pi)
            espectro[k] += magn * np.exp(1j * fase)
            espectro[n_f - k] += magn * np.exp(-1j * fase)

    return espectro

def generar_envolvente_adsr(longitud):
    # ADSR que termina en 0, con release corto para evitar clicks si no se aplica fade externo
    attack = 0.01
    decay = 0.08
    sustain = 0.7
    release = 0.12

    iA = int(attack * longitud)
    iD = int((attack + decay) * longitud)
    iR = int((1 - release) * longitud)

    envol = np.zeros(longitud)
    if iA > 0:
        envol[:iA] = np.linspace(0.0, 1.0, iA)
    if iD > iA:
        envol[iA:iD] = np.linspace(1.0, sustain, iD - iA)
    if iR > iD:
        envol[iD:iR] = sustain
    if iR < longitud:
        envol[iR:] = np.linspace(sustain, 0.0, longitud - iR)
    return envol

def aplicar_fades(signal, fade_in_ms=5.0, fade_out_ms=15.0):
    # Fade in/out para evitar clicks (valores en ms)
    n = len(signal)
    fi = int(SAMPLE_RATE * (fade_in_ms / 1000.0))
    fo = int(SAMPLE_RATE * (fade_out_ms / 1000.0))
    if fi < 1:
        fi = 1
    if fo < 1:
        fo = 1
    if fi + fo >= n:
        # si la señal es muy corta, ajustar proporcionalmente
        fi = int(n * 0.1)
        fo = int(n * 0.1)

    # aplicar fade-in
    signal[:fi] *= np.linspace(0.0, 1.0, fi)
    # aplicar fade-out
    signal[-fo:] *= np.linspace(1.0, 0.0, fo)
    return signal

def sintetizar_sonido_fourier(espectro):
    señal = np.fft.irfft(espectro)
    muestras = int(SAMPLE_RATE * DURACION_SONIDO)
    if len(señal) < muestras:
        señal = np.pad(señal, (0, muestras - len(señal)))
    else:
        señal = señal[:muestras]

    # ADSR y fades
    envol = generar_envolvente_adsr(len(señal))
    señal *= envol

    # evitar clicks residuales
    señal = aplicar_fades(señal, fade_in_ms=5.0, fade_out_ms=15.0)

    # normalizar
    max_val = np.max(np.abs(señal)) + 1e-12
    señal = señal / max_val * VOLUMEN
    return señal

def reproducir_sonido_async(señal):
    try:
        sd.play(señal.astype(np.float32), samplerate=SAMPLE_RATE, blocking=False)
    except Exception as e:
        print(f"Error reproducción: {e}")

def reverb_simple(signal, wet):
    if wet <= 0:
        return signal
    # kernels determinísticos (no regenerar ruido en cada llamada) para coherencia
    # usar kernels fijos basados en semilla local para consistencia
    # np.random.seed(0)
    k1 = np.random.randn(2000) * 0.0007
    k2 = np.random.randn(1500) * 0.0009
    k3 = np.random.randn(1000) * 0.0011
    comb1 = np.convolve(signal, k1, mode='same')
    comb2 = np.convolve(signal, k2, mode='same')
    comb3 = np.convolve(signal, k3, mode='same')
    reverb = (comb1 + comb2 + comb3) / 3.0
    out = (1 - wet) * signal + wet * reverb
    # nivel relativo conservado
    mv = np.max(np.abs(out)) + 1e-12
    out = out / mv * (np.max(np.abs(signal)) + 1e-12)
    return out

# ============================================================
# UTILIDADES VISUALES / CAPTURAS
# ============================================================
def color_aleatorio():
    return (random.random(), random.random(), random.random())

def tomar_captura():
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arte_abstracto_{ts}.png"
        fp = os.path.join(capturas_dir, filename)
        plt.savefig(fp, dpi=150, bbox_inches='tight', facecolor='w', edgecolor='w', format='png')
        if os.path.exists(fp):
            print(f"✓ Captura guardada: {fp}")
            return True
        return False
    except Exception as e:
        print(f"Error captura: {e}")
        return False

# ============================================================
# BUCLE PRINCIPAL (siempre desde el centro, efectos por cuadrante)
# ============================================================
ultima_captura = time.time()
intervalo_captura = 1800

contador_pasos = 0
INTERVALO_SONIDO = 1  # generar sonido cada X pasos (aquí cada paso)

print("Iniciando programa con sonido geométrico + Fourier + efectos por cuadrante...")

try:
    while True:
        color_linea = color_aleatorio()
        color_punto = color_aleatorio()
        direccion = random.choice(direcciones)

        # SIEMPRE desde el centro
        x0, y0 = 0.0, 0.0

        x = np.zeros(N + 1)
        y = np.zeros(N + 1)
        x[0], y[0] = x0, y0

        hp, = ax.plot([], [], color=color_linea, linestyle='-', marker='o',
                      markersize=1, linewidth=2, alpha=0.9)
        hp2, = ax.plot([], [], 'o', color=color_punto, markersize=10, alpha=0.9)

        for i in range(N):
            tiempo = time.time()
            if tiempo - ultima_captura >= intervalo_captura:
                if tomar_captura():
                    ultima_captura = tiempo

            # Movimiento aleatorio (mantengo tu lógica)
            if direccion == 'arriba':
                x[i+1] = x[i] + (np.random.rand() * 2 - 1) * 2
                y[i+1] = y[i] - np.random.rand()
            elif direccion == 'abajo':
                x[i+1] = x[i] + (np.random.rand() * 2 - 1) * 2
                y[i+1] = y[i] + np.random.rand()
            elif direccion == 'izquierda':
                x[i+1] = x[i] - np.random.rand()
                y[i+1] = y[i] + (np.random.rand() * 2 - 1) * 2
            else:  # derecha
                x[i+1] = x[i] + np.random.rand()
                y[i+1] = y[i] + (np.random.rand() * 2 - 1) * 2

            contador_pasos += 1
            if contador_pasos >= INTERVALO_SONIDO:
                # calcular espectro y sintetizar
                espectro = generar_espectro_musical(x[i], y[i])
                sonido = sintetizar_sonido_fourier(espectro)

                # distancia normalizada 0..1
                distancia = np.sqrt(x[i]**2 + y[i]**2)
                d_norm = np.clip(distancia / max_dist, 0.0, 1.0)

                # intensidad por paso: 0 (paso 0) .. 1 (paso N-1)
                paso_norm = float(i) / max(1, (N - 1))

                # combinación: paso controla los 20 niveles; distancia modula sensiblemente
                wet = paso_norm * (d_norm ** 0.5)
                wet = float(np.clip(wet, 0.0, 0.98))

                # seleccionar cuadrante y aplicar efecto
                if x[i] >= 0 and y[i] >= 0:
                    # Q1: Reverb
                    sonido = reverb_simple(sonido, wet)
                    efecto_nombre = f"Reverb wet={wet:.5f}"
                elif x[i] < 0 and y[i] >= 0:
                    # Q2: Delay
                    # delay más corto si wet pequeño, más largo si wet grande
                    delay_ms = 80 + wet * 420
                    sonido = aplicar_delay(sonido, wet, delay_ms=int(delay_ms))
                    efecto_nombre = f"Delay {int(delay_ms)}ms wet={wet:.2f}"
                elif x[i] < 0 and y[i] < 0:
                    # Q3: Distorsión
                    drive = wet  # 0..1
                    sonido = aplicar_distorsion(sonido, drive)
                    efecto_nombre = f"Distortion drive={drive:.5f}"
                else:
                    # Q4: Flanger
                    depth_ms = 1.0 + wet * 8.0
                    rate_hz = 0.2 + wet * 1.0
                    sonido = aplicar_flanger(sonido, wet, depth_ms=depth_ms, rate_hz=rate_hz)
                    efecto_nombre = f"Flanger depth={depth_ms:.2f}ms rate={rate_hz:.3f}Hz wet={wet:.4f}"

                # normalizar después del efecto para mantener control de volumen
                max_val = np.max(np.abs(sonido)) + 1e-12
                sonido = sonido / max_val * VOLUMEN

                # reproducir
                reproducir_sonido_async(sonido)

                # log simple (solo algunos pasos para no saturar)
                if i == 0 or i == N-1 or i % 5 == 0:
                    nota_freq, nota_idx = nota_desde_angulo(x[i], y[i])
                    print(f"Paso {i}: pos=({x[i]:.2f},{y[i]:.2f}) dist={d_norm:.2f} nota_idx={nota_idx} freq={nota_freq:.1f}Hz -> {efecto_nombre}")

                contador_pasos = 0

            # actualizar gráfico
            hp.set_data(x[:i+1], y[:i+1])
            hp2.set_data([x[i]], [y[i]])
            plt.draw()
            plt.pause(0.08)

        hp2.remove()

except KeyboardInterrupt:
    print("\nInterrupción recibida. Tomando captura final...")
    tomar_captura()

finally:
    plt.ioff()
    plt.close()
    print("Programa finalizado.")
