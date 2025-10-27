import numpy as np
import os
import sys 

# Pokus o import CuPy pro GPU akceleraci
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy nalezeno, výpočty poběží na GPU.")
except ImportError:
    cp = np # Fallback na NumPy
    GPU_AVAILABLE = False
    print("CuPy nenalezeno, výpočty poběží na CPU.")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import time

# ======================= KONFIGURACE SIMULACE ======================

CONFIGURATION = 'chaos_3body' # 'circle_pair', 'ellipse_pair', 'chaos_3body'
DATA_FILENAME = 'chaos_simulation_data.npz' # Soubor pro uložení dat simulace

# --- Parametry pro 2 Tělesa ---
BODY_PARAMS_2BODY = {
    'circle_pair': {
        'masses': [10.0, 0.1],
        'distance': 8.0,
        'velocity_factor': 1.0
    },
    'ellipse_pair': {
        'masses': [10.0, 0.2],
        'distance': 12.0,
        'velocity_factor': 0.5
    }
}

# --- Parametry pro Demonstraci Chaosu (3 Tělesa) ---
CHAOS_CONFIG = {
    'n_bodies': 3,
    'n_variations': 3, # Počet simulací s mírně odlišnými podmínkami
    'initial_state': { # Počáteční stav těles
    'bodies': [
        [1.0, [2.0, 1.5, 0], [-0.25, 0.3, 0.1]],
        
        [1.0, [-2.0, 1.0, 0], [0.3, -0.2, -0.2]],
        
        [1.0, [0.0, -2.5, 0], [-0.05, -0.1, 0.1]]
    ]
},
    # Perturbace (malé změny pro další variace)
    'perturbation': {
        'body_index': 0,            # Index tělesa, které perturbujeme
        'position_delta': 1e-5,     # Malá změna přidaná k pozici
        'velocity_delta': 1e-5,     # Malá změna přidaná k rychlosti
        'axes': ['x', 'y']          # Osa pro perturbaci pozice (var 1) a rychlosti (var 2)
                                    # Var 0: žádná perturbace
                                    # Var 1: perturbace pozice v ose axes[0]
                                    # Var 2: perturbace rychlosti v ose axes[1]
    },
    'color_schemes': ['#FF4444', '#44FF44', '#6666FF'] # Základní barvy pro variace (Hex kody)
}

# --- Simulační Parametry ---
SIM_PARAMS = {
    'dt': 0.005,                 # Zmenšeno pro stabilitu chaosu
    'total_time': 200,          # Zvětšeno pro zobrazení divergence
    'G': 1.0,                   # Gravitační konstanta
    'softening': 0.01,          # Změkčení (menší pro chaos!)
}

# --- Parametry Vizualizace  ---
VIZ_FPS = 30
VIZ_PLOT_EVERY = 2           
VIZ_TRAIL_LENGTH = 100
VIZ_MARKER_SCALE_FACTOR = 150
VIZ_DEFAULT_BOX_SIZE = 2.0
LINEWIDTH = 1.5

# Globální proměnné pro interaktivní pohled
last_azim = -60
last_elev = 30

# ========================= POMOCNÉ FUNKCE ============================

def create_colormap(n, base_color_hex):
    """Vytvoří barevnou paletu (odstíny) pro n těles z hex barvy."""
    try:
        base_rgb = colors.to_rgb(base_color_hex)
        colors_array = []
        for i in range(n):
            factor = 0.4 + 0.6 * (i / max(1, n - 1)) # od 0.4 do 1.0
            c = tuple(min(1.0, base_rgb[j] * factor + (1 - factor)) for j in range(3))
            colors_array.append(c + (1.0,)) # Přidáme alpha
        return colors_array
    except ValueError:
        print(f"Varování: Neplatná hex barva '{base_color_hex}', použije se výchozí.")
        cmap = plt.cm.viridis
        return [cmap(i / max(1, n - 1)) for i in range(n)]

def get_body_size(masses, scale_factor):
    """Určí velikost těles lineárně podle hmotnosti."""
    min_size, max_size = 10, 300
    scaled_size = masses * scale_factor
    return np.clip(scaled_size, min_size, max_size)

def calculate_energy(positions, velocities, masses, G):
    """Vypočítá kinetickou, potenciální a celkovou energii. Používá softening pro konzistenci."""
    softening = SIM_PARAMS['softening'] # Použijeme softening i pro energii
    ke = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    pe = 0.0; n = len(masses)
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            r_sqr = np.sum(diff**2)
            # Důležité: Použití softeningu pro konzistenci
            r_ij = np.sqrt(r_sqr + softening**2) 
            pe -= G * masses[i] * masses[j] / r_ij
    return ke, pe, ke + pe

def calculate_system_divergence(positions_set):
    """Vypočítá průměrnou euklidovskou vzdálenost mezi variacemi."""
    variation_count = len(positions_set)
    if variation_count <= 1 or positions_set[0] is None: return None

    # Zjistíme nejkratší historii
    n_timesteps = min(p.shape[0] for p in positions_set if p is not None) 
    if n_timesteps == 0: return None

    # Ořízneme na stejnou délku a převedeme na CPU
    pos_set_np = []
    for pos in positions_set:
        if pos is None: return None 
        # Zajišťujeme, že data jsou zkrácena a převedena na NumPy
        pos_np = cp.asnumpy(pos[:n_timesteps]) if GPU_AVAILABLE and isinstance(pos, cp.ndarray) else pos[:n_timesteps]
        pos_set_np.append(pos_np)

    divergence = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        total_dist = 0; pair_count = 0
        for i in range(variation_count):
            for j in range(i + 1, variation_count):
                diff = pos_set_np[i][t] - pos_set_np[j][t]
                dist = np.sqrt(np.sum(diff**2))
                total_dist += dist
                pair_count += 1
        if pair_count > 0:
             divergence[t] = total_dist / pair_count

    return divergence

# ========================= UKLÁDÁNÍ/NAČÍTÁNÍ DAT ============================

def save_simulation_data(all_pos, all_ene, time_points, masses, divergence):
    """Uloží výsledky simulace do komprimovaného NumPy souboru (.npz)."""
    try:
        # Převedeme veškerá data na NumPy pole před uložením
        # Zajišťujeme, že i CuPy pole jsou převedena na NumPy před ukládáním
        pos_np = [cp.asnumpy(p) if GPU_AVAILABLE and isinstance(p, cp.ndarray) else p for p in all_pos]
        
        # Sestavíme data do slovníku pro uložení
        save_dict = {
            f'positions_var_{i}': pos_np[i] for i in range(len(pos_np))
        }
        # Energetické historie jsou již v NumPy z run_simulation
        save_dict.update({
            f'ke_var_{i}': all_ene[i][0] for i in range(len(all_ene))
        })
        save_dict.update({
            f'pe_var_{i}': all_ene[i][1] for i in range(len(all_ene))
        })
        save_dict.update({
            f'total_e_var_{i}': all_ene[i][2] for i in range(len(all_ene))
        })
        
        save_dict['time_points'] = time_points
        save_dict['masses'] = masses
        save_dict['divergence'] = divergence

        np.savez_compressed(DATA_FILENAME, **save_dict)
        print(f"Data simulace úspěšně uložena do: {DATA_FILENAME}")
    except Exception as e:
        print(f"Chyba při ukládání dat: {e}")

def load_simulation_data():
    """Načte výsledky simulace z .npz souboru."""
    if not os.path.exists(DATA_FILENAME):
        return None

    print(f"Načítám data simulace z: {DATA_FILENAME}")
    try:
        data = np.load(DATA_FILENAME, allow_pickle=True)
        
        # Zjištění počtu variací z uložených klíčů
        n_vars = 0
        while f'positions_var_{n_vars}' in data:
            n_vars += 1
            
        if n_vars == 0:
            print("Chyba: Soubor neobsahuje data variací.")
            return None
            
        all_pos = [data[f'positions_var_{i}'] for i in range(n_vars)]
        all_ene = [
            (data[f'ke_var_{i}'], data[f'pe_var_{i}'], data[f'total_e_var_{i}']) 
            for i in range(n_vars)
        ]
        time_points = data['time_points']
        masses = data['masses']
        divergence = data['divergence']
        
        print("Data simulace úspěšně načtena.")
        return all_pos, all_ene, time_points, masses, divergence

    except Exception as e:
        print(f"Chyba při načítání dat: {e}")
        return None

# =================== FYZIKÁLNÍ VÝPOČTY (GPU/CPU) =====================
# Funkce calculate_accelerations a runge_kutta_step (beze změny)
def calculate_accelerations(positions_gpu, masses_gpu, G, softening):
    """Vypočítá zrychlení (optimalizováno pro GPU)."""
    n = positions_gpu.shape[0]
    diff = positions_gpu[:, cp.newaxis, :] - positions_gpu[cp.newaxis, :, :]
    dist_sqr = cp.sum(diff**2, axis=-1) + softening**2
    dist_sqr = cp.maximum(dist_sqr, 1e-12)
    inv_dist_cubed = dist_sqr**(-1.5)
    cp.fill_diagonal(inv_dist_cubed, 0.0)
    term = masses_gpu[cp.newaxis, :, cp.newaxis] * diff * inv_dist_cubed[:, :, cp.newaxis]
    accelerations = -G * cp.sum(term, axis=1)
    if not cp.all(cp.isfinite(accelerations)):
         print(f"Varování: NaN/Inf v calculate_accelerations. Nahrazuji nulami.")
         accelerations = cp.nan_to_num(accelerations)
    return accelerations

def runge_kutta_step(positions_gpu, velocities_gpu, masses_gpu, dt, G, softening):
    """Jeden krok RK4."""
    a1 = calculate_accelerations(positions_gpu, masses_gpu, G, softening)
    k1_v = a1 * dt; k1_r = velocities_gpu * dt
    a2 = calculate_accelerations(positions_gpu + k1_r * 0.5, masses_gpu, G, softening)
    k2_v = a2 * dt; k2_r = (velocities_gpu + k1_v * 0.5) * dt
    a3 = calculate_accelerations(positions_gpu + k2_r * 0.5, masses_gpu, G, softening)
    k3_v = a3 * dt; k3_r = (velocities_gpu + k2_v * 0.5) * dt
    a4 = calculate_accelerations(positions_gpu + k3_r, masses_gpu, G, softening)
    k4_v = a4 * dt; k4_r = (velocities_gpu + k3_v) * dt
    new_velocities = velocities_gpu + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6.0
    new_positions = positions_gpu + (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6.0
    # Kontroly NaN/Inf
    if not cp.all(cp.isfinite(new_positions)):
        print("Varování: NaN/Inf v pozicích (RK4). Nahrazuji nulami.")
        new_positions = cp.nan_to_num(new_positions)
    if not cp.all(cp.isfinite(new_velocities)):
        print("Varování: NaN/Inf v rychlostech (RK4). Nahrazuji nulami.")
        new_velocities = cp.nan_to_num(new_velocities)
    return new_positions, new_velocities


# ====================== INICIALIZACE TĚLES =========================
def initialize_bodies(variation_idx=0):
    """Inicializuje tělesa podle CONFIGURATION."""
    if CONFIGURATION == 'chaos_3body':
        n_bodies = CHAOS_CONFIG['n_bodies']
        base_state = CHAOS_CONFIG['initial_state']['bodies']
        pert_cfg = CHAOS_CONFIG['perturbation']

        if len(base_state) != n_bodies:
            raise ValueError(f"Počet tělěs v CHAOS_CONFIG ({len(base_state)}) nesouhlasí s n_bodies ({n_bodies})")

        masses = np.array([b[0] for b in base_state], dtype=float)
        positions = np.array([b[1] for b in base_state], dtype=float)
        velocities = np.array([b[2] for b in base_state], dtype=float)

        # Aplikace perturbace pro variace > 0
        if variation_idx > 0:
            idx = pert_cfg['body_index']
            if 0 <= idx < n_bodies:
                axis_map = {'x': 0, 'y': 1, 'z': 2}
                # Perturbace pozice (pro variation_idx = 1, 3, 5, ...)
                if variation_idx % 2 != 0:
                     axis_idx = axis_map[pert_cfg['axes'][0]]
                     delta = pert_cfg['position_delta']
                     positions[idx, axis_idx] += delta
                     print(f"Variace {variation_idx}: Perturbuji pozici tělesa {idx} v ose {pert_cfg['axes'][0]} o {delta}")
                # Perturbace rychlosti (pro variation_idx = 2, 4, 6, ...)
                else:
                     axis_idx = axis_map[pert_cfg['axes'][1]]
                     delta = pert_cfg['velocity_delta']
                     velocities[idx, axis_idx] += delta
                     print(f"Variace {variation_idx}: Perturbuji rychlost tělesa {idx} v ose {pert_cfg['axes'][1]} o {delta}")
            else:
                print(f"Varování: Neplatný index tělesa pro perturbaci: {idx}")

    elif CONFIGURATION in ['circle_pair', 'ellipse_pair']:
        n_bodies = 2
        params = BODY_PARAMS_2BODY[CONFIGURATION]
        masses = np.array(params['masses'], dtype=float)
        distance = params['distance']
        velocity_factor = params['velocity_factor']
        positions = np.zeros((n_bodies, 3)); velocities = np.zeros((n_bodies, 3))
        m1, m2 = masses[0], masses[1]; m_total = m1 + m2
        if m_total < 1e-10: m_total=1e-10 # Zabraň dělení nulou
        r1 = distance * m2 / m_total; r2 = distance * m1 / m_total
        positions[0] = [-r1, 0, 0]; positions[1] = [ r2, 0, 0]
        v_ref = np.sqrt(SIM_PARAMS['G'] * m_total / distance)
        v1 = v_ref * (m2 / m_total) * velocity_factor
        v2 = v_ref * (m1 / m_total) * velocity_factor
        velocities[0] = [0, -v1, 0]; velocities[1] = [0,  v2, 0]
    else:
        raise ValueError(f"Neznámá konfigurace: {CONFIGURATION}")

    # Korekce COM a hybnosti pro všechny konfigurace
    total_mass = np.sum(masses)
    if total_mass > 1e-10:
        com_pos = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
        positions -= com_pos
        com_vel = np.sum(velocities * masses[:, np.newaxis], axis=0) / total_mass
        velocities -= com_vel
    else:
        print("Varování: Nulová celková hmotnost.")

    # Závěrečná kontrola
    if not np.all(np.isfinite(positions)): print(f"CHYBA: NaN/Inf v pozicích po inicializaci (variace {variation_idx})")
    if not np.all(np.isfinite(velocities)): print(f"CHYBA: NaN/Inf v rychlostech po inicializaci (variace {variation_idx})")

    return positions, velocities, masses


# ======================= HLAVNÍ SIMULACE =============================
def run_simulation(variation_idx=0):
    """Spustí N-body simulaci a vrátí historii."""
    print(f"--- Spouštím simulaci {'(variace '+str(variation_idx)+')' if CONFIGURATION=='chaos_3body' else ''} ---")
    positions_np, velocities_np, masses_np = initialize_bodies(variation_idx)
    n_bodies = positions_np.shape[0]

    # Použití cp.asarray pro kompatibilitu s CuPy/NumPy
    positions_gpu = cp.asarray(positions_np)
    velocities_gpu = cp.asarray(velocities_np)
    masses_gpu = cp.asarray(masses_np)

    dt = SIM_PARAMS['dt']; total_time = SIM_PARAMS['total_time']
    G = SIM_PARAMS['G']; softening = SIM_PARAMS['softening']
    plot_every = VIZ_PLOT_EVERY; n_steps = int(total_time / dt)
    if n_steps <= 0: print("CHYBA: Neplatný počet kroků."); return None

    n_frames_saved = (n_steps // plot_every) + 1
    positions_history = np.zeros((n_frames_saved, n_bodies, 3))
    ke_history = np.zeros(n_frames_saved); pe_history = np.zeros(n_frames_saved)
    total_energy_history = np.zeros(n_frames_saved); time_points = np.zeros(n_frames_saved)
    history_idx = 0

    print(f"Parametry: N={n_bodies}, dt={dt}, T={total_time}, Kroky={n_steps}")
    start_time = time.time(); last_print_time = start_time

    for step in range(n_steps):
        if step % plot_every == 0:
             current_pos_np = cp.asnumpy(positions_gpu)
             current_vel_np = cp.asnumpy(velocities_gpu)
             if not np.all(np.isfinite(current_pos_np)) or not np.all(np.isfinite(current_vel_np)):
                 print(f"CHYBA: Nestabilita v kroku {step}! Přerušuji."); return None 
             if history_idx < n_frames_saved:
                 positions_history[history_idx] = current_pos_np
                 # KINETICKÁ A POTENCIÁLNÍ ENERGIE MUSÍ BÝT VYPOČÍTÁNA ZDE (na CPU datech)
                 ke, pe, total_e = calculate_energy(current_pos_np, current_vel_np, masses_np, G)
                 ke_history[history_idx] = ke; pe_history[history_idx] = pe; total_energy_history[history_idx] = total_e
                 time_points[history_idx] = step * dt
                 history_idx += 1
             # Zjednodušený výpis pokroku
             current_time = time.time()
             if current_time - last_print_time > 10.0: # Každých 10s
                 print(f"  Pokrok: {((step + 1) / n_steps * 100):.1f}%")
                 last_print_time = current_time

        positions_gpu, velocities_gpu = runge_kutta_step(
            positions_gpu, velocities_gpu, masses_gpu, dt, G, softening
        )

    end_time = time.time(); print(f"--- Simulace dokončena za {end_time - start_time:.2f}s ---")
    # Uložíme poslední krok
    if n_steps % plot_every == 0 and history_idx < n_frames_saved:
         current_pos_np = cp.asnumpy(positions_gpu); current_vel_np = cp.asnumpy(velocities_gpu)
         if np.all(np.isfinite(current_pos_np)) and np.all(np.isfinite(current_vel_np)):
             positions_history[history_idx] = current_pos_np
             ke, pe, total_e = calculate_energy(current_pos_np, current_vel_np, masses_np, G)
             ke_history[history_idx]=ke; pe_history[history_idx]=pe; total_energy_history[history_idx]=total_e
             time_points[history_idx] = n_steps * dt; history_idx += 1

    # Ořízneme pole
    return (positions_history[:history_idx], ke_history[:history_idx], pe_history[:history_idx],
            total_energy_history[:history_idx], time_points[:history_idx], masses_np)


# ========================= VIZUALIZACE ===============================

def _update_artists(trail, marker, positions_history, frame, body_idx, trail_length):
    start_idx = max(0, frame - trail_length + 1)
    current_pos = positions_history[frame, body_idx]
    if not np.all(np.isfinite(current_pos)):
        marker._offsets3d = ([], [], []); trail.set_data([], []); trail.set_3d_properties([])
        return False
    marker._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])
    trail_data = positions_history[start_idx : frame + 1, body_idx, :]
    valid_trail_data = trail_data[np.all(np.isfinite(trail_data), axis=1)]
    if valid_trail_data.shape[0] > 0:
        trail.set_data(valid_trail_data[:, 0], valid_trail_data[:, 1])
        trail.set_3d_properties(valid_trail_data[:, 2])
    else:
        trail.set_data([], []); trail.set_3d_properties([])
    return True

# ... (visualize_2body_simulation beze změny) ...
def visualize_2body_simulation(positions_history, ke, pe, total_e, time_points, masses):
    """Vizualizuje simulaci 2 těles."""
    global last_azim, last_elev
    if positions_history is None or positions_history.shape[0] == 0: print("Prázdná data."); return
    n_frames, n_bodies, _ = positions_history.shape
    colors = ['red', 'blue']; sizes = get_body_size(masses, VIZ_MARKER_SCALE_FACTOR)
    valid_pos = positions_history[np.all(np.isfinite(positions_history), axis=(1,2))]
    max_r = VIZ_DEFAULT_BOX_SIZE
    if valid_pos.shape[0] > 0: max_r = max(np.max(np.abs(valid_pos))*1.1, max_r)

    fig = plt.figure(figsize=(14, 7)); gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[:, 0], projection='3d')
    ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[1, 1])
    ax1.set_xlim(-max_r, max_r); ax1.set_ylim(-max_r, max_r); ax1.set_zlim(-max_r, max_r)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z'); ax1.set_title(f'Simulace: {CONFIGURATION}')
    ax2.set_title('Energie'); ax2.grid(True); ax3.set_title('Zachování E'); ax3.grid(True)
    trails = [ax1.plot([],[],[], '-', color=colors[i], alpha=0.6, linewidth=LINEWIDTH)[0] for i in range(n_bodies)]
    markers = [ax1.scatter([],[],[], s=sizes[i], color=colors[i], alpha=0.9, ec='k', lw=0.5) for i in range(n_bodies)]
    time_txt = ax1.text2D(0.05, 0.95, '', transform=ax1.transAxes)

    if len(time_points) > 0: # Statické grafy
        ax2.plot(time_points, ke, 'r:', label='KE'); ax2.plot(time_points, pe, 'b:', label='PE')
        ax2.plot(time_points, total_e, 'g-', label='Total'); ax2.legend(fontsize='small')
        e0 = total_e[0]; e_change = (total_e - e0) / (abs(e0) + 1e-10)
        ax3.plot(time_points, e_change, 'purple'); ax3.set_ylabel('Rel. změna E')
        max_ch = np.max(np.abs(e_change)); ax3.set_ylim(-max_ch*1.1, max_ch*1.1)

    def update(frame): # Animace
        global last_azim, last_elev
        artists = []
        for i in range(n_bodies):
            _update_artists(trails[i], markers[i], positions_history, frame, i, VIZ_TRAIL_LENGTH)
            artists.extend([trails[i], markers[i]])
        t = time_points[frame]; time_txt.set_text(f'T: {t:.2f}'); artists.append(time_txt)
        # Interaktivní pohled
        azim, elev = ax1.azim, ax1.elev
        if abs(azim - last_azim) > 0.1 or abs(elev - last_elev) > 0.1: last_azim, last_elev = azim, elev
        ax1.view_init(elev=last_elev, azim=last_azim)
        return artists

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000/VIZ_FPS, blit=False, repeat=True)
    plt.tight_layout(pad=1.5); plt.show()


# Vizualizace pro více variací (chaos) - UPRAVENO pro ukládání MP4
def visualize_multiple_variations(all_positions, all_energies, time_points, masses, divergence, save_video=False):
    """
    Vizualizuje 3D trajektorie, logaritmus divergence a zachování energie.
    """
    global last_azim, last_elev
    if not all_positions or any(p is None for p in all_positions): print("Prázdná data."); return

    var_count = len(all_positions)
    n_bodies = masses.shape[0]
    n_frames = min(p.shape[0] for p in all_positions if p is not None)
    if n_frames == 0: print("Žádné snímky k zobrazení."); return

    base_colors_hex = CHAOS_CONFIG['color_schemes']
    sizes = get_body_size(masses, VIZ_MARKER_SCALE_FACTOR)
    tp = time_points[:n_frames] # Společný čas

    # 1. Nastavení Layoutu (3 grafy vedle sebe)
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d') # 3D Animace
    ax2 = fig.add_subplot(gs[0, 1]) # Divergence
    ax3 = fig.add_subplot(gs[0, 2]) # Energie

    # --- 3D Nastavení ---
    max_r = VIZ_DEFAULT_BOX_SIZE
    valid_coords = []
    for pos_hist in all_positions:
        valid_pos = pos_hist[:n_frames][np.all(np.isfinite(pos_hist[:n_frames]), axis=(1,2))]
        if valid_pos.shape[0] > 0: valid_coords.append(np.max(np.abs(valid_pos)))
    if valid_coords: max_r = max(max(valid_coords) * 1.1, max_r)
    ax1.set_xlim(-max_r, max_r); ax1.set_ylim(-max_r, max_r); ax1.set_zlim(-max_r, max_r)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title('Trajektorie (Chaos)')
    
    all_trails = []; all_markers = []
    for v_idx in range(var_count):
        current_color = base_colors_hex[v_idx % len(base_colors_hex)]
        var_trails = [ax1.plot([],[],[], '-', color=current_color, alpha=0.6, linewidth=LINEWIDTH)[0] for b in range(n_bodies)]
        var_markers = [ax1.scatter([],[],[], s=sizes[b], color=current_color, alpha=0.9, ec='k', lw=0.5) for b in range(n_bodies)]
        all_trails.append(var_trails); all_markers.append(var_markers)

    time_txt = ax1.text2D(0.02, 0.98, '', transform=ax1.transAxes, ha='left', va='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Graf Divergence (ax2) ---
    if divergence is not None and len(divergence) > 0:
        # Použijeme log10 pro exponenciální růst
        log_divergence = np.log10(np.maximum(divergence[:n_frames], 1e-15))
        ax2.plot(tp, log_divergence, 'k-')
        ax2.set_title('Exponenciální Divergence Trajektorií')
        ax2.set_xlabel('Čas (T)')
        ax2.set_ylabel('Log10(Prům. Vzdálenost)')
        ax2.grid(True)
        ax2.axhline(y=np.log10(max_r*0.5), color='r', linestyle='--', label='Saturace (velikost systému)')
        ax2.legend()
    else:
        ax2.set_title('Divergence: Chyba dat')
        ax2.grid(True)

    # --- Graf Zachování Energie (ax3) ---
    # Zobrazíme změnu E pro první variaci
    ke0, pe0, total_e_var0 = all_energies[0]
    e0 = total_e_var0[0]
    e_change = (total_e_var0[:n_frames] - e0) / (np.abs(e0) + 1e-10)

    ax3.plot(tp, e_change, color='purple', linewidth=LINEWIDTH)
    ax3.set_title('Relativní Změna Celkové Energie (Var. 0)')
    ax3.set_xlabel('Čas (T)')
    ax3.set_ylabel(r'$\Delta E / |E_0|$')
    ax3.grid(True)
    max_ch = np.max(np.abs(e_change))
    ax3.set_ylim(-max_ch*1.1, max_ch*1.1)

    # Funkce update (jen pro 3D graf)
    def update(frame):
        global last_azim, last_elev
        artists = []
        for v_idx in range(var_count):
            for b_idx in range(n_bodies):
                _update_artists(all_trails[v_idx][b_idx], all_markers[v_idx][b_idx], all_positions[v_idx], frame, b_idx, VIZ_TRAIL_LENGTH)
                artists.extend([all_trails[v_idx][b_idx], all_markers[v_idx][b_idx]])
        t = tp[frame]; time_txt.set_text(f'T: {t:.2f}'); artists.append(time_txt)
        azim, elev = ax1.azim, ax1.elev
        if abs(azim - last_azim) > 0.1 or abs(elev - last_elev) > 0.1: last_azim, last_elev = azim, elev
        ax1.view_init(elev=last_elev, azim=last_azim)
        return artists

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000/VIZ_FPS, blit=False, repeat=True)
    
    if save_video:
        VIDEO_FILENAME = 'chaos_3body_simulation.mp4'
        print(f"Ukládám animaci do {VIDEO_FILENAME}. To může trvat několik minut...")
        try:
            # Uložení videa (vyžaduje FFMPEG)
            ani.save(VIDEO_FILENAME, fps=VIZ_FPS, extra_args=['-vcodec', 'libx264'])
            print("Ukládání videa dokončeno.")
        except Exception as e:
            print("CHYBA PŘI UKLÁDÁNÍ VIDEA: Ujistěte se, že máte nainstalovaný FFMPEG.")
            print(f"Podrobná chyba: {e}")
            
    plt.tight_layout(pad=0.5)
    plt.show()

# ========================= SPUŠTĚNÍ ============================

def run_multiple_variations():
    """Spustí více variací pro chaos demo."""
    n_vars = CHAOS_CONFIG['n_variations']
    print(f"====== SPUŠTĚNÍ {n_vars} VARIACÍ PRO CHAOS DEMO ======")
    all_pos = [None] * n_vars; all_ene = [None] * n_vars
    masses = None; time_points = None; success = True

    for i in range(n_vars):
        result = run_simulation(variation_idx=i)
        if result is None: success = False; break
        pos, ke, pe, tot, tp, m = result
        all_pos[i] = pos; all_ene[i] = (ke, pe, tot) 
        if i == 0: masses, time_points = m, tp

    if not success: print("Některá variace selhala."); return None

    print("Výpočet divergence...")
    divergence = calculate_system_divergence(all_pos) 
    if divergence is None: print("Nepodařilo se spočítat divergenci."); return None

    return all_pos, all_ene, time_points, masses, divergence


def main():
    """Hlavní funkce s logikou pro uložení/načtení."""
    print(f"====== SIMULACE N-TĚLES ({'GPU' if GPU_AVAILABLE else 'CPU'}) ======")
    print(f" Konfigurace: {CONFIGURATION}")

    if CONFIGURATION == 'chaos_3body':
        
        # 1. Zkusíme nejprve načíst existující data
        results = load_simulation_data() 

        if results:
            print("Přecházím rovnou k vizualizaci (načtená data).")
            all_pos, all_ene, time_points, masses, divergence = results
            
            save_video_q = input("Chcete uložit animaci do MP4 souboru a zavřít program? (a/n): ").lower()
            save_video = save_video_q == 'a'
            
            print("Spouštím vizualizaci chaosu...")
            visualize_multiple_variations(all_pos, all_ene, time_points, masses, divergence, save_video=save_video)
            
        else:
            print("Data nenalezena. Spouštím novou simulaci...")
            # 2. Spustit novou simulaci, pokud data neexistují
            results = run_multiple_variations()
            
            if results:
                all_pos, all_ene, time_points, masses, divergence = results
                # 3. Uložíme výsledky nové simulace
                save_simulation_data(all_pos, all_ene, time_points, masses, divergence)
                
                save_video_q = input("Nová data byla uložena. Chcete uložit animaci do MP4 souboru a zavřít program? (a/n): ").lower()
                save_video = save_video_q == 'a'

                print("Spouštím vizualizaci chaosu...")
                visualize_multiple_variations(all_pos, all_ene, time_points, masses, divergence, save_video=save_video)
            else:
                print("Vizualizace chaosu přeskočena kvůli chybě.")
    
    # ... (logika pro 2 tělesa zůstává beze změny) ...
    elif CONFIGURATION in ['circle_pair', 'ellipse_pair']:
        result = run_simulation()
        if result:
            pos, ke, pe, total_e, tp, m = result
            print("Spouštím vizualizaci 2 těles...")
            visualize_2body_simulation(pos, ke, pe, total_e, tp, m)
        else:
            print("Vizualizace 2 těles přeskočena kvůli chybě.")
    else:
        print(f"Neznámá nebo nepodporovaná konfigurace: {CONFIGURATION}")


if __name__ == "__main__":
    main()