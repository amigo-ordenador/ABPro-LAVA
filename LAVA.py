import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import meep as mp
import numpy as np
import os

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FFMpegWriter

# Resolución de las imágenes.
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Mapa de color personalizado.
plt.style.use('dark_background')
cmap_alpha = LinearSegmentedColormap.from_list(
    'custom_alpha', [[1, 1, 1, 0], [1, 1, 1, 1]])
cmap_blue = LinearSegmentedColormap.from_list(
    'custom_blue', [[0, 0, 0], [0, 0.66, 1], [1, 1, 1]])

# Etiquetas de entramado.
def label_plot(ax, title=None, xlabel=None, ylabel=None, elapsed=None):
    if title:
        ax.set_title(title)
    elif elapsed is not None:
        ax.set_title(f'{elapsed:0.1f} fs')
    if xlabel is not False:
        ax.set_xlabel('x (μm)'if xlabel is None else xlabel)
    if ylabel is not False:
        ax.set_ylabel('y (μm)'if ylabel is None else ylabel)

# Trama del dominio 2D.
def plot_eps_data(eps_data, domain, ax=None, **kwargs):
    ax = ax or plt.gca()
    ax.imshow(eps_data.T, cmap=cmap_alpha, extent=domain, origin='lower')
    label_plot(ax, **kwargs)

# Trama del campo electromagnético.
def plot_ez_data(ez_data, domain, ax=None, vmax=None, aspect=None, **kwargs):
    ax = ax or plt.gca()
    ax.imshow(
        np.abs(ez_data.T),
        interpolation='spline36',
        cmap=cmap_blue,
        extent=domain,
        vmax=vmax,
        aspect=aspect,
        origin='lower',
        )
    label_plot(ax, **kwargs)

# Trama PML.
def plot_pml(pml_thickness, domain, ax=None):
    ax = ax or plt.gca()
    x_start = domain[0] + pml_thickness
    x_end = domain[1] - pml_thickness
    y_start = domain[2] + pml_thickness
    y_end = domain[3] - pml_thickness
    rect = plt.Rectangle(
        (x_start, y_start),
        x_end - x_start,
        y_end - y_start,
        fill=None,
        color='#fff',
        linestyle='dashed',
        )
    ax.add_patch(rect)

# Velocidad de la luz (Speed of Light) representada en μm/fs.
SOL = 299792458e-9

# Dominio espacial 2D en μm.
domain = [0, 30, -10, 10]
center = mp.Vector3(
    (domain[1] + domain[0]) / 2,
    (domain[3] + domain[2]) / 2,
    )
cell_size = mp.Vector3(
    domain[1] - domain[0],
    domain[3] - domain[2],
    )

# Dimensiones de la celosía (muro con dos aperturas).
wall_position = 10
wall_thickness = 0.5
aperture_width = 1
inner_wall_len = 4  # Muro entre las aperturas.
outer_wall_len = (
    cell_size[1]
    - 2*aperture_width
    - inner_wall_len
    ) / 2

# Define un material para el muro con una constante dieléctrica alta, 
# bloqueando efectivamente la luz y reflejándola hacia el orígen.
material = mp.Medium(epsilon=1e6)

# Define el muro como una matriz de tres bloques verticales.
geometry = [
    mp.Block(
        mp.Vector3(wall_thickness, outer_wall_len, mp.inf),
        center=mp.Vector3(
            wall_position - center.x,
            domain[3] - outer_wall_len / 2),
        material=material),
    mp.Block(
        mp.Vector3(wall_thickness, outer_wall_len, mp.inf),
        center=mp.Vector3(
            wall_position - center.x,
            domain[2] + outer_wall_len / 2),
        material=material),
    mp.Block(
        mp.Vector3(wall_thickness, inner_wall_len, mp.inf),
        center=mp.Vector3(wall_position - center.x, 0),
        material=material),
    ]

# Capa PML.
pml_thickness = 1
pml_layers = [mp.PML(pml_thickness)]

# Extrae y exporta la data dieléctrica (acorde a la geometría del muro).
sim = mp.Simulation(cell_size=cell_size, geometry=geometry, resolution=10)
sim.init_sim()
eps_data = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Dielectric)
ax = plt.gca()
plot_pml(pml_thickness, domain, ax=ax)
plot_eps_data(eps_data, domain, ax=ax)
plt.savefig('wall_geometry.png')
plt.close() 

# Parametrización de la onda (Longitud de onda, frequencia y lóbulo de apertura).
source_lambda = 0.47  # (valor en μm)
source_frequency = 1 / source_lambda
source_beam_width = 6

# Método que devuelve una onda plana de valor complejo en el eje x.
def plane_wave(x):
    return np.exp(2j * np.pi / source_lambda * x)

# Trama de la onda plana.
xarr = np.linspace(0, 10*source_lambda, 1000)
wave = plane_wave(xarr)
plt.plot(xarr, wave.real)
plt.xlabel('x (μm)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.savefig('plane_wave.png')
plt.close()

# Método que computa el perfil de Gauss en el eje y.
def gaussian_profile(y):
    return np.exp(-y**2 / (2 * (source_beam_width / 2)**2))

# Trama del perfil de Gauss.
yarr = np.linspace(domain[2], domain[3], 200)
prof = gaussian_profile(yarr)
plt.plot(yarr, prof)
plt.xlabel('y (μm)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.savefig('gaussian_profile.png')
plt.close()

# Mátriz de puntos (Meshgrid).
X, Y = np.meshgrid(xarr, yarr)

# Superposición de tramas (términos combinados).
combined = plane_wave(X) * gaussian_profile(Y)
plt.imshow(
    np.real(combined),
    cmap='coolwarm',
    aspect='auto',
    extent=[xarr[0], xarr[-1], yarr[0], yarr[-1]],
    origin='lower',
    )
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.savefig('combined_terms.png')
plt.close()

# Función que define la amplitud de la onda.
def amp_func(pos):
    return plane_wave(pos[0]) * gaussian_profile(pos[1])

source = mp.Source(
    src=mp.ContinuousSource(
        frequency=source_frequency,
        is_integrated=True,
        ),
    component=mp.Ez,
    center= mp.Vector3(1, 0, 0) - center,  # Orígen posicionado en el extremo izquierdo, excluyendo PML.
    size=mp.Vector3(y=cell_size[1]),       # Extensión de la altura completa, incluyendo PML.
    amp_func=amp_func,
    )

# Resuelve en términos del componente más pequeño (método linear).
smallest_length = min(
    source_lambda,
    wall_thickness,
    aperture_width,
    inner_wall_len,
)
pixel_count = 10
resolution = int(np.ceil(pixel_count / smallest_length))
print('Simulation resolution:', resolution)

# Simulación de objeto. 
# Sé que esta función se repite un poco más abajo, 
# pero removerla causa comportamiento inusual en el código...
# ¯\_(ツ)_/¯
sim = mp.Simulation(
    cell_size=cell_size,
    sources=[source],
    boundary_layers=pml_layers,
    geometry=geometry,
    resolution=resolution,
    force_complex_fields=True,
    )

# Método para extraer la información Ez y data dieléctrica.
def get_data(sim, cell_size):
    ez_data = sim.get_array(
        center=mp.Vector3(), size=cell_size, component=mp.Ez)
    eps_data = sim.get_array(
        center=mp.Vector3(), size=cell_size, component=mp.Dielectric)
    return ez_data, eps_data

# Función para correr la simulación hasta que la luz viaje su recorrido completo (+1).
sim.run(until=cell_size[0] + 10)
ez_data, eps_data = get_data(sim, cell_size)

# Trama de la simulación.
ax = plt.gca()
plot_ez_data(ez_data, domain, ax=ax)
plot_eps_data(eps_data, domain, ax=ax)
plot_pml(pml_thickness, domain, ax=ax)

# Duración de la simulación y cantidad de fotogramas.
sim_time = 120  # Valor en fs.
n_frames = 30 # Es recomendable cambiar el número de fotogramas a 30 o 60 para exportar video.

# PATH del archivo H5.
sim_path = 'simulation.h5'

# Simulación de objeto.
# ¯\_(ツ)_/¯
sim = mp.Simulation(
    cell_size=cell_size,
    sources=[source],
    boundary_layers=pml_layers,
    geometry=geometry,
    resolution=resolution,
    force_complex_fields=True,
    )

def simulate(sim, sim_path, sim_time, n_frames):
    
    # Elimina simulaciones previas (en caso sobreeescritura).
    if os.path.exists(sim_path):
        os.remove(sim_path)

    # Tiempo delta (en fs) entre fotogramas.
    # Se sustrae 1 pues se incluye el estado inicial del primer frame.
    fs_delta = sim_time / (n_frames - 1)
    
    # Exporta los resultados a un archivo H5.
    with h5py.File(sim_path, 'a') as f:
    
        # Guarda los parametros de la simulación como referencia futura.
        f.attrs['sim_time'] = sim_time
        f.attrs['n_frames'] = n_frames
        f.attrs['fs_delta'] = fs_delta
        f.attrs['resolution'] = sim.resolution
        
        # Guarda el estado inicial como el primer frame.
        sim.init_sim()
        ez_data, eps_data = get_data(sim, cell_size)
        f.create_dataset(
            'ez_data',
            shape=(n_frames, *ez_data.shape),
            dtype=ez_data.dtype,
            )
        f.create_dataset(
            'eps_data',
            shape=eps_data.shape,
            dtype=eps_data.dtype,
            )
        f['ez_data'][0]  = ez_data
        f['eps_data'][:] = eps_data
    
        # Simula y captura los fotogramas restantes.
        for i in range(1, n_frames):
    
            # Corre el programa hasta el siguiente frame.
            sim.run(until=SOL * fs_delta)
    
            # Captura la data del campo eléctrico (Ez).   
            ez_data, _ = get_data(sim, cell_size)
            f['ez_data'][i]  = ez_data

# Ejecuta la simulación y configura los parámetros para la visualización.
simulate(sim, sim_path, sim_time, n_frames)

# Definición del tamaño y número de subtramas para los gráficos
fig_rows = 3
fig_cols = 2
n_subplots = fig_rows * fig_cols

# Creación de la figura principal y de los ejes para las subtramas
fig, ax = plt.subplots(figsize=(10, 8))

# Toma el archivo H5 y lo llama devuelta para generar cada frame del video de la simulación.
def animate(frame):
    ax.clear()
    with h5py.File(sim_path, 'r') as f:
        ez_data = f['ez_data'][frame]
        eps_data = f['eps_data'][:]
        elapsed = frame * f.attrs['fs_delta']
        vmax = 0.6  # Fuerza una constante de brillo.
        plot_ez_data(ez_data, domain, ax=ax, vmax=vmax, elapsed=elapsed)
        plot_eps_data(eps_data, domain, ax=ax)
        plot_pml(pml_thickness, domain, ax=ax)
    return ax

# Determina número total de frames.
with h5py.File(sim_path, 'r') as f:
    n_frames = f['ez_data'].shape[0]

# Función para crear la animación a partir de fotogramas.
anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200, blit=False)

def calculate_and_plot_gradient(ez_data, domain):
    # Calcula las derivadas parciales.
    dy, dx = np.gradient(np.real(ez_data))
    
    # Calcula la magniutd de la gradiente.
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    
    plt.figure(figsize=(9, 6))
    plt.imshow(gradient_magnitude.T, extent=domain, origin='lower', cmap='viridis')
    plt.colorbar(label='Gradient Magnitude')
    plt.title('Gradient of Electric Field (Final Snapshot)')
    plt.xlabel('x (μm)')
    plt.ylabel('y (μm)')
    plt.savefig('field_gradient.png')
    plt.close()

# Setea el escritor ffmpeg.
writer = FFMpegWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)

# Exporta la animación en formato mp4.
anim.save('simulation_video.mp4', writer=writer)

# Opcional: Guarda cada fotograma individualmente.
#with h5py.File(sim_path, 'r') as f:
#    for k in range(n_subplots):
#        fig, _ax = plt.subplots(figsize=(8, 6))
#        ez_data = f['ez_data'][k]
#        eps_data = f['eps_data'][:]
#        elapsed = k * f.attrs['fs_delta']
#        vmax = 0.6
#        plot_ez_data(ez_data, domain, ax=_ax, vmax=vmax, elapsed=elapsed)
#        plot_eps_data(eps_data, domain, ax=_ax)
#        plot_pml(pml_thickness, domain, ax=_ax)
#        plt.tight_layout()
#        plt.savefig(f'snapshot_{k:02d}.png', dpi=300, bbox_inches='tight')
#        plt.close()
        
# Toma la simulación final sin el tiempo estimado. 
with h5py.File(sim_path, 'r') as f:
    final_snap = f['ez_data'][-1]
    plt.figure(figsize=(9, 6))
    plt.imshow(final_snap.T, cmap='inferno', extent=domain, origin='lower')
    plt.colorbar(label='Intensity')
    plt.title('Final Simulation Snapshot')
    plt.xlabel('x (μm)')
    plt.ylabel('y (μm)')
    plt.savefig('final_snapshot.png')
    plt.close()

# Computa la intesidad como el cuadrado de la amplitud compleja.
# Probablemente hay otro método mejor que este...
final_snap = np.abs(final_snap)**2

# Toma secciones a distintas distancias de la celosía (producida por la división entre los muros).
slice_dists = [10, 11, 12, 15, 20, 25]
slices = [final_snap[i * resolution] for i in slice_dists]
yarr = np.linspace(domain[2], domain[3], final_snap.shape[1])

# Función para visualizar los resultados.
def plot_intensity(slice, yarr, ax1, ax2, vmax=None, xval=None, xlabel=False, ylabel=False):
    ax1.plot(yarr, slice)
    ax1.tick_params(axis='x', labelbottom=False)
    if ylabel:
        ax1.set_ylabel('$|E|^2$')
    else:
        ax1.tick_params('y', labelleft=False)
    if xval:
        ax1.annotate(
            f'x={xval}',
            xy=(1, 1),
            xytext=(-4, -4),
            xycoords='axes fraction',
            textcoords='offset pixels',
            horizontalalignment='right',
            verticalalignment='top',
            )
    ax2.imshow(
        np.vstack(slice).T,
        cmap='inferno',
        aspect='auto',
        vmax=vmax,
        extent=[yarr[0], yarr[-1], 0, 1],
        )
    ax2.set_xlim([yarr[0], yarr[-1]])
    ax2.tick_params('y', labelleft=False)
    ax2.set_yticks([])
    if xlabel:
        ax2.set_xlabel('y (μm)')
    else:
        ax2.tick_params(axis='x', labelbottom=False)

fig, ax = plt.subplots(
    4, 3,
    figsize=(9, 6),
    gridspec_kw=dict(
        width_ratios=(4, 4, 4),
        height_ratios=(4, 1, 4, 1),
        wspace=0.12,
        hspace=0.1,
        ),
    sharex='col',
    sharey='row',
    )
for k, slice in enumerate(slices):
    i = 2 * int(k / 3)
    j = k % 3
    plot_intensity(
        slice, yarr, ax[i][j], ax[i+1][j],
        vmax=np.max(slices[:3]) if k < 3 else np.max(slices[3:]),
        xval=slice_dists[k],
        xlabel=(i==2),
        ylabel=(j==0))
    plt.tight_layout()
    plt.savefig('intensity_plots.png')
    plt.close()

with h5py.File(sim_path, 'r') as f:
    final_snap = f['ez_data'][-1]
    calculate_and_plot_gradient(final_snap, domain)