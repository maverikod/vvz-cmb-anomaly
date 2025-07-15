import pandas as pd
import matplotlib.pyplot as plt
from .array_backend import cp, np, array_load, array_save

def run_scale_clustering_analysis(csv_path: str, alpha: float = 0.7, r0: float = None, plot: bool = True, top_n: int = 10) -> None:
    """
    Analyze clustering of anomaly radii by discrete theoretical levels (no mask).
    Args:
        csv_path (str): Path to CSV with anomalies (must have 'radius_deg')
        alpha (float): Logarithmic step (default 0.7)
        r0 (float): Minimal radius (default: min(radius_deg) in file)
        plot (bool): Whether to plot histogram and save
        top_n (int): Number of top levels to print
    """
    print(f'Loading anomalies from {csv_path}...')
    df = pd.read_csv(csv_path)
    if 'radius_deg' not in df.columns:
        raise ValueError('CSV must contain column "radius_deg"')
    radii = df['radius_deg'].values
    if r0 is None:
        r0 = np.min(radii)
    log_r = np.log10(radii)
    i_levels = np.round((np.log10(radii / r0)) / alpha).astype(int)
    r_levels = r0 * 10 ** (alpha * i_levels)
    df['discrete_level'] = i_levels
    df['r_level'] = r_levels
    # Гистограмма по уровням
    unique_levels, counts = np.unique(i_levels, return_counts=True)
    r_centers = r0 * 10 ** (alpha * unique_levels)
    print('\nГистограмма числа аномалий по дискретным уровням r_i:')
    print(f"{'i':>3} {'r_i':>10} {'count':>6}")
    for i, r, c in zip(unique_levels, r_centers, counts):
        print(f"{i:>3} {r:10.3f} {c:6}")
    # Топ-N уровней
    top_idx = np.argsort(counts)[::-1][:top_n]
    print(f"\nТоп-{top_n} уровней с максимальным числом аномалий:")
    for idx in top_idx:
        print(f"i={unique_levels[idx]}, r_i={r_centers[idx]:.3f}, count={counts[idx]}")
    # Визуализация
    if plot:
        plt.figure()
        plt.bar(r_centers, counts, width=0.15*r_centers, align='center', alpha=0.7, edgecolor='k')
        plt.xscale('log')
        plt.xlabel('Discrete radius $r_i$ [deg]')
        plt.ylabel('Number of anomalies')
        plt.title(f'Clustering of anomaly radii (alpha={alpha}, r0={r0:.2f})')
        plt.tight_layout()
        plt.savefig('anomaly_radius_clustering.png', dpi=200) 
    # Краткий вывод
    print('\n--- Вывод ---')
    if np.max(counts) > np.mean(counts) + 2*np.std(counts):
        print('Обнаружены выраженные кластеры аномалий на определённых радиусах (дискретизация масштабов подтверждается).')
    else:
        print('Явной кластеризации по дискретным масштабам не обнаружено.')
    print('См. график anomaly_radius_clustering.png и таблицу выше.') 