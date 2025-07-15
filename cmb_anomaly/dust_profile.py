import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os
from tqdm import tqdm
from .array_backend import cp, np, array_load, array_save

def dust_stats_for_anomaly(center_l: float, center_b: float, radius_deg: float, dust_map: np.ndarray, nside: int, border_width: float = 0.2) -> dict:
    """
    Вычисляет статистики пыли внутри аномалии и в кольце на границе.
    Args:
        center_l, center_b: центр аномалии (градусы)
        radius_deg: радиус аномалии (градусы)
        dust_map: карта пыли (HEALPix)
        nside: параметр HEALPix
        border_width: ширина кольца (в радиусах, например 0.2*R)
    Returns:
        dict со статистиками
    """
    center_vec = hp.ang2vec(np.deg2rad(90-center_b), np.deg2rad(center_l))
    r_rad = np.deg2rad(radius_deg)
    border_in = r_rad
    border_out = r_rad * (1 + border_width)
    # Внутри аномалии
    pix_in = hp.query_disc(nside, center_vec, r_rad, inclusive=True, fact=4)
    vals_in = dust_map[pix_in]
    # В кольце (border)
    pix_out = hp.query_disc(nside, center_vec, border_out, inclusive=True, fact=4)
    pix_border = np.setdiff1d(pix_out, pix_in)
    vals_border = dust_map[pix_border]
    # Статистики
    stats = {
        'mean_in': np.mean(vals_in),
        'std_in': np.std(vals_in),
        'min_in': np.min(vals_in),
        'max_in': np.max(vals_in),
        'p10_in': np.percentile(vals_in, 10),
        'p90_in': np.percentile(vals_in, 90),
        'mean_border': np.mean(vals_border),
        'std_border': np.std(vals_border),
        'contrast1': (np.mean(vals_border) - np.mean(vals_in)) / np.mean(vals_in) if np.mean(vals_in) != 0 else np.nan,
        'contrast2': (np.mean(vals_in) / (np.mean(vals_border) - np.mean(vals_in))) if (np.mean(vals_border) - np.mean(vals_in)) != 0 else np.nan,
        'n_in': len(vals_in),
        'n_border': len(vals_border)
    }
    return stats

def radial_profile(center_l: float, center_b: float, radius_deg: float, dust_map: np.ndarray, nside: int, r_max: float = 2.0, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Строит радиальный профиль пыли от центра аномалии до r_max*R.
    Returns:
        r_bins (deg), mean_dust (per bin)
    """
    center_vec = hp.ang2vec(np.deg2rad(90-center_b), np.deg2rad(center_l))
    r_edges = np.linspace(0, np.deg2rad(radius_deg*r_max), n_bins+1)
    r_bins = 0.5*(r_edges[:-1] + r_edges[1:]) / np.pi * 180
    mean_dust = []
    for r1, r2 in zip(r_edges[:-1], r_edges[1:]):
        pix_ring = hp.query_disc(nside, center_vec, r2, inclusive=True, fact=4)
        if r1 > 0:
            pix_inner = hp.query_disc(nside, center_vec, r1, inclusive=True, fact=4)
            pix_ring = np.setdiff1d(pix_ring, pix_inner)
        vals = dust_map[pix_ring]
        mean_dust.append(np.mean(vals))
    return r_bins, np.array(mean_dust)

def analyze_dust_profile(anomaly_csv: str, dust_npy: str, out_dir: str, zone_col: str = 'zone', border_width: float = 0.2, r_max: float = 2.0, n_bins: int = 20, n_control: int = 1000, seed: int = 42):
    """
    Основная функция анализа морфологии пыли вокруг CMB-анoмалий.
    Сохраняет профили, статистики, гистограммы, summary в out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(anomaly_csv)
    dust_map = array_load(dust_npy)
    nside = hp.get_nside(dust_map)
    rng = np.random.default_rng(seed)
    # По зонам и в среднем
    zones = df[zone_col].unique() if zone_col in df.columns else ['all']
    for zone in zones:
        if zone == 'all':
            dfa = df
        else:
            dfa = df[df[zone_col] == zone]
        stats_list = []
        profiles = []
        for idx, row in tqdm(list(dfa.iterrows()), total=len(dfa), desc=f"Anomalies ({zone})"):
            stats = dust_stats_for_anomaly(row['l'], row['b'], row['radius_deg'], dust_map, nside, border_width)
            stats_list.append(stats)
            r_bins, prof = radial_profile(row['l'], row['b'], row['radius_deg'], dust_map, nside, r_max, n_bins)
            profiles.append(prof)
        # Сводная таблица
        stats_df = pd.DataFrame(stats_list)
        stats_npy = stats_df.to_numpy(dtype=np.float64)
        np.save(os.path.join(out_dir, f'dust_stats_{zone}.npy'), stats_npy)
        stats_df = stats_df.applymap(to_py_scalar)
        stats_df.to_csv(os.path.join(out_dir, f'dust_stats_{zone}.csv'), index=False)
        # Гистограмма контраста
        plt.figure()
        plt.hist(stats_df['contrast1'], bins=30, alpha=0.7, label='Contrast1')
        plt.hist(stats_df['contrast2'], bins=30, alpha=0.7, label='Contrast2')
        plt.xlabel('Contrast')
        plt.ylabel('Count')
        plt.title(f'Dust contrast distribution ({zone})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'dust_contrast_hist_{zone}.png'), dpi=200)
        plt.close()
        # Усреднённый профиль
        if profiles:
            mean_profile = np.nanmean(profiles, axis=0)
            r_bins = np.array(r_bins, dtype=np.float64)
            mean_profile = np.array(mean_profile, dtype=np.float64)
            np.save(os.path.join(out_dir, f'dust_radial_profile_{zone}.npy'), np.stack([r_bins, mean_profile]))
            plt.figure()
            plt.plot(r_bins, mean_profile, label='Mean dust profile')
            plt.xlabel('Radius [deg]')
            plt.ylabel('Mean dust')
            plt.title(f'Radial dust profile ({zone})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'dust_radial_profile_{zone}.png'), dpi=200)
            plt.close()
    # Контрольные точки (по зонам)
    for zone in zones:
        if zone == 'all':
            dfa = df
        else:
            dfa = df[df[zone_col] == zone]
        controls_stats = []
        controls_profiles = []
        for idx, row in tqdm(list(dfa.iterrows()), total=len(dfa), desc=f"Control ({zone})"):
            for i in tqdm(range(n_control // max(1, len(dfa))), leave=False, desc=f"Ctrl {idx+1}/{len(dfa)}", position=1):
                if zone == 'thin_disk':
                    b = rng.uniform(-10, 10)
                elif zone == 'thick_disk':
                    b = rng.uniform(-30, -10) if rng.random() < 0.5 else rng.uniform(10, 30)
                elif zone == 'halo':
                    b = rng.uniform(-90, -30) if rng.random() < 0.5 else rng.uniform(30, 90)
                else:
                    b = rng.uniform(-90, 90)
                l = rng.uniform(0, 360)
                r = row['radius_deg']
                stats = dust_stats_for_anomaly(l, b, r, dust_map, nside, border_width)
                controls_stats.append(stats)
                _, prof = radial_profile(l, b, r, dust_map, nside, r_max, n_bins)
                controls_profiles.append(prof)
        # Сводная таблица
        if controls_stats:
            controls_df = pd.DataFrame(controls_stats)
            controls_df.to_csv(os.path.join(out_dir, f'dust_stats_control_{zone}.csv'), index=False)
            # Гистограмма контраста
            plt.figure()
            plt.hist(controls_df['contrast1'], bins=30, alpha=0.7, label='Contrast1')
            plt.hist(controls_df['contrast2'], bins=30, alpha=0.7, label='Contrast2')
            plt.xlabel('Contrast')
            plt.ylabel('Count')
            plt.title(f'Dust contrast (control, {zone})')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'dust_contrast_hist_control_{zone}.png'), dpi=200)
            plt.close()
            # Усреднённый профиль
            if controls_profiles:
                mean_profile = np.nanmean(controls_profiles, axis=0)
                plt.figure()
                plt.plot(r_bins, mean_profile, label='Mean dust profile (control)')
                plt.xlabel('Radius [deg]')
                plt.ylabel('Mean dust')
                plt.title(f'Radial dust profile (control, {zone})')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'dust_radial_profile_control_{zone}.png'), dpi=200)
                plt.close()
    # Итоговая сводная таблица
    summary = []
    for zone in zones:
        stats_path = os.path.join(out_dir, f'dust_stats_{zone}.csv')
        control_path = os.path.join(out_dir, f'dust_stats_control_{zone}.csv')
        if os.path.exists(stats_path) and os.path.exists(control_path):
            df_stats = pd.read_csv(stats_path)
            df_control = pd.read_csv(control_path)
            summary.append({
                'zone': zone,
                'mean_contrast_anom': df_stats['contrast1'].mean(),
                'mean_contrast_control': df_control['contrast1'].mean(),
                'std_contrast_anom': df_stats['contrast1'].std(),
                'std_contrast_control': df_control['contrast1'].std(),
                'n_anom': len(df_stats),
                'n_control': len(df_control)
            })
    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(out_dir, 'dust_contrast_summary.csv'), index=False) 