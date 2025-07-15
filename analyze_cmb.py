import matplotlib.pyplot as plt
import healpy as hp
import csv
from cmb_anomaly.array_backend import cp, np, array_load, array_save, print_backend_info

print_backend_info()

# Файлы
TEMPERATURE_PATH = 'data/cmb_temperature.npy'
MASK_PATH = 'data/mask_common.npy'
RESULTS_CSV = 'results_anomalies_multi.csv'

# Параметры перебора
RADIUS_DEG_LIST = list(range(1, 16))  # 1°–15°
STEP_DEG = 5  # шаг по небу
TOP_N = 20

# Загрузка данных
print('Loading temperature...')
temperature = array_load(TEMPERATURE_PATH)
print('Loading mask...')
mask = array_load(MASK_PATH)
print('Mask dtype:', mask.dtype, 'shape:', mask.shape)  # DEBUG

# Применение маски
valid_pixels = mask > 0
temperature_masked = np.where(valid_pixels, temperature, hp.UNSEEN)

# Глобальная статистика
mean_global = np.mean(temperature[valid_pixels])
std_global = np.std(temperature[valid_pixels])

# Сетка центров
NSIDE = hp.npix2nside(len(temperature))
npix = len(temperature)
step_pix = int(STEP_DEG / 60 * npix) if STEP_DEG > 0 else 1
centers = np.arange(0, npix, int(npix / (4 * 180 // STEP_DEG)))  # ~равномерно по небу

results = []
print('Scanning sky...')
for radius_deg in RADIUS_DEG_LIST:
    radius_rad = np.deg2rad(radius_deg)
    for pix in centers:
        if not valid_pixels[pix]:
            continue
        region_pix = hp.query_disc(NSIDE, hp.pix2vec(NSIDE, pix), radius_rad, inclusive=True, fact=4)
        region_pix = region_pix[valid_pixels[region_pix]]
        if len(region_pix) < 100:
            continue
        vals = temperature[region_pix]
        mean = np.mean(vals)
        std = np.std(vals)
        S = np.abs(mean - mean_global) / std_global
        theta, phi = hp.pix2ang(NSIDE, pix)
        l = np.rad2deg(phi)
        b = 90 - np.rad2deg(theta)
        results.append({
            'center_pix': pix,
            'radius_deg': radius_deg,
            'S': S,
            'mean': mean,
            'std': std,
            'npix': len(region_pix),
            'l': l,
            'b': b
        })
    print(f'  Done radius {radius_deg}°')

# Сохраняем все результаты в CSV
with open(RESULTS_CSV, 'w', newline='') as csvfile:
    fieldnames = ['center_pix', 'radius_deg', 'S', 'mean', 'std', 'npix', 'l', 'b']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)
print(f'All anomaly region results saved to {RESULTS_CSV}')

# Топ-N аномалий по S
results = sorted(results, key=lambda x: abs(x['S']), reverse=True)
print(f'\nTop {TOP_N} most anomalous regions (multi-radius):')
print(f"{'#':<2} {'pix':<8} {'r':>3} {'S':>8} {'mean':>10} {'std':>10} {'npix':>6} {'l':>7} {'b':>7}")
for i, r in enumerate(results[:TOP_N]):
    print(f"{i+1:<2} {r['center_pix']:<8} {r['radius_deg']:>3} {r['S']:>8.2f} {r['mean']:>10.3e} {r['std']:>10.3e} {r['npix']:>6} {r['l']:7.2f} {r['b']:7.2f}")

# Визуализация топ-N аномалий на карте
anomaly_pix = [r['center_pix'] for r in results[:TOP_N]]
coords = np.array(hp.pix2ang(NSIDE, anomaly_pix))
hp.mollview(temperature_masked, title=f'CMB Temperature with Top-{TOP_N} Multi-Scale Anomalies', unit='K', cmap='coolwarm', min=np.nanpercentile(temperature[valid_pixels], 0.5), max=np.nanpercentile(temperature[valid_pixels], 99.5))
hp.projscatter(coords[1], np.pi/2 - coords[0], lonlat=True, s=80, c='black', marker='o', label='Anomaly')
plt.legend()
plt.savefig('cmb_mollweide_anomalies_multi.png', dpi=200)
plt.show()

# Примеры S(r) для топ-5 аномалий
plt.figure()
for i, r0 in enumerate(results[:5]):
    s_curve = [rr['S'] for rr in results if rr['center_pix'] == r0['center_pix']]
    r_curve = [rr['radius_deg'] for rr in results if rr['center_pix'] == r0['center_pix']]
    plt.plot(r_curve, s_curve, label=f'pix {r0["center_pix"]}, l={r0["l"]:.1f}, b={r0["b"]:.1f}')
plt.xlabel('Radius [deg]')
plt.ylabel('S (anomaly)')
plt.title('S(r) for Top-5 Anomalies')
plt.legend()
plt.savefig('cmb_S_vs_r.png', dpi=200)
plt.show() 