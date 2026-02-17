# Voluntad Desviada — Sistema Generativo Audiovisual

**Voluntad Desviada** es un proyecto de arte generativo que traduce caminatas aleatorias en el plano bidimensional en estructuras sonoras y visuales en tiempo real. El sistema combina matemáticas, síntesis digital y visualización algorítmica para producir una obra donde el movimiento espacial se transforma directamente en sonido.

---

## Descripción General

El programa genera trayectorias pseudoaleatorias que parten siempre del origen (0,0). Cada posición alcanzada durante la caminata se interpreta como información musical, que se convierte en un espectro de frecuencias mediante síntesis basada en transformadas discretas inversas de Fourier. El resultado es un entorno audiovisual donde geometría y sonido están acoplados.

El sistema funciona como un motor generativo autónomo: no reproduce material preexistente, sino que produce continuamente nuevas configuraciones visuales y sonoras a partir de reglas matemáticas.

---

## Funcionamiento Conceptual

El flujo interno del programa puede resumirse de la siguiente manera:

    Posición → Nota → Espectro → IDFT → Envolvente → Efectos → Salida sonora

Cada etapa corresponde a una transformación matemática o de señal:

- **Posición:** coordenadas del punto actual en la caminata.  
- **Nota:** frecuencia fundamental calculada según el ángulo respecto del origen.  
- **Espectro:** conjunto de armónicos generados en función de la distancia al centro.  
- **IDFT:** reconstrucción temporal del sonido.  
- **Envolvente ADSR:** modelado dinámico del volumen.  
- **Efectos:** procesamiento según cuadrante y distancia.  
- **Salida:** reproducción en tiempo real.  

---

## Caminata Aleatoria

Las trayectorias se construyen mediante desplazamientos sucesivos:

- Cada caminata tiene 20 pasos.  
- Se selecciona una dirección dominante inicial.  
- Existe variación continua en la magnitud y desviación lateral.  
- No hay grilla discreta ni posiciones cuantizadas.  

Esto produce recorridos orgánicos, con coherencia direccional pero impredecibles en detalle.

---

## Sonificación

La síntesis sonora se basa en tres principios:

### 1. Altura tonal dependiente del ángulo
El ángulo polar del punto determina qué nota se activa dentro de una escala cromática de tres octavas.

### 2. Complejidad armónica dependiente de la distancia
Cuanto más lejos está el punto del origen:
- mayor cantidad de armónicos  
- mayor contenido espectral  
- mayor densidad sonora  

### 3. Intensidad de efectos dependiente del radio
La distancia normalizada controla el parámetro *wet* de los efectos, generando una progresión gradual desde señal limpia en el centro hasta máxima transformación en los bordes.

---

## Sistema de Cuadrantes

El plano cartesiano se divide en cuatro regiones. Cada cuadrante aplica un tipo distinto de procesamiento sonoro, permitiendo que la posición espacial no solo determine la nota sino también el carácter tímbrico.

Esto convierte el espacio visual en un mapa acústico.

---

## Características Técnicas

- Síntesis espectral procedural  
- Generación visual en tiempo real  
- Audio no pregrabado  
- Sistema determinista-estocástico  
- Arquitectura modular de efectos  
- Normalización automática de señal  
- Captura automática de imágenes  

---

## Tecnologías Utilizadas

- Python  
- NumPy  
- Matplotlib  
- SoundDevice  
- SciPy  

---

## Objetivo Artístico

El proyecto explora la relación entre:

- espacio y sonido  
- azar y estructura  
- matemática y estética  

Propone una práctica donde el código no es solo herramienta técnica sino medio compositivo, y donde los algoritmos funcionan como agentes creativos capaces de producir obra.

---

## Posibles Extensiones

El sistema está diseñado para poder ampliarse fácilmente. Algunas líneas futuras incluyen:

- síntesis polifónica simultánea  
- espacialización sonora  
- control en vivo  
- exportación multicanal  
- integración con sensores o datos externos  

---

## Licencia

Proyecto de uso artístico y educativo. Consultar licencia del repositorio para detalles.

---

## Autor

Proyecto desarrollado como investigación en arte generativo audiovisual y matemáticas aplicadas al sonido.

Emanuel Bellanti
Universidad Nacional de las Artes
Departamento de Artes Multimediales
Especialización en Sonido Aplicado a las Artes Digitales
2025
