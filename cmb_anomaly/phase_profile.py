import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
import os
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm
from .array_backend import cp, np, array_load, array_save

def extract_profile(center_l, center_b, radius_deg, map_arr, nside, r_max=2.0, n_bins=20):
    """
    Extracts radial profile for a given center and radius on a HEALPix map.
    Logs npix, NaN/Inf, out-of-bounds for each bin.
    """
    center_vec = hp.ang2vec(np.deg2rad(90-center_b), np.deg2rad(center_l))
    r_edges = np.linspace(0, np.deg2rad(radius_deg*r_max), n_bins+1)
    r_bins = 0.5*(r_edges[:-1] + r_edges[1:]) / np.pi * 180
    profile = []
    for i, (r1, r2) in enumerate(zip(r_edges[:-1], r_edges[1:])):
        pix_ring = hp.query_disc(nside, center_vec, r2, inclusive=True, fact=4)
        if r1 > 0:
            pix_inner = hp.query_disc(nside, center_vec, r1, inclusive=True, fact=4)
            pix_ring = np.setdiff1d(pix_ring, pix_inner)
        # Filter out-of-bounds
        pix_ring = pix_ring[pix_ring < map_arr.size]
        npix = len(pix_ring)
        if npix == 0:
            logging.warning(f"[extract_profile] Bin {i}: no pixels (out-of-bounds)")
            profile.append(np.nan)
        else:
            vals = map_arr[pix_ring]
            n_nan = np.isnan(vals).sum()
            n_inf = np.isinf(vals).sum()
            if n_nan > 0 or n_inf > 0:
                logging.warning(f"[extract_profile] Bin {i}: {n_nan} NaN, {n_inf} Inf in profile region")
            profile.append(np.nanmean(vals))
    return r_bins, np.array(profile)

def extract_stats(center_l, center_b, radius_deg, map_arr, nside, border_width=0.2, min_npix=10):
    """
    Extracts statistics for anomaly region and border. Logs npix, NaN/Inf, out-of-bounds.
    Filters regions with too few pixels.
    """
    center_vec = hp.ang2vec(np.deg2rad(90-center_b), np.deg2rad(center_l))
    r_rad = np.deg2rad(radius_deg)
    border_out = r_rad * (1 + border_width)
    pix_in = hp.query_disc(nside, center_vec, r_rad, inclusive=True, fact=4)
    pix_out = hp.query_disc(nside, center_vec, border_out, inclusive=True, fact=4)
    pix_border = np.setdiff1d(pix_out, pix_in)
    # Filter out-of-bounds
    pix_in = pix_in[pix_in < map_arr.size]
    pix_border = pix_border[pix_border < map_arr.size]
    n_in = len(pix_in)
    n_border = len(pix_border)
    if n_in < min_npix or n_border < min_npix:
        logging.warning(f"[extract_stats] Too few pixels: n_in={n_in}, n_border={n_border} (min_npix={min_npix})")
        return {k: np.nan for k in ['mean_in','std_in','min_in','max_in','p10_in','p90_in','mean_border','std_border','contrast','n_in','n_border']}
    vals_in = map_arr[pix_in]
    vals_border = map_arr[pix_border]
    n_nan_in = np.isnan(vals_in).sum()
    n_inf_in = np.isinf(vals_in).sum()
    n_nan_border = np.isnan(vals_border).sum()
    n_inf_border = np.isinf(vals_border).sum()
    if n_nan_in > 0 or n_inf_in > 0 or n_nan_border > 0 or n_inf_border > 0:
        logging.warning(f"[extract_stats] NaN/Inf: in: {n_nan_in} NaN, {n_inf_in} Inf; border: {n_nan_border} NaN, {n_inf_border} Inf")
    stats = {
        'mean_in': np.nanmean(vals_in) if n_in else np.nan,
        'std_in': np.nanstd(vals_in) if n_in else np.nan,
        'min_in': np.nanmin(vals_in) if n_in else np.nan,
        'max_in': np.nanmax(vals_in) if n_in else np.nan,
        'p10_in': np.nanpercentile(vals_in, 10) if n_in else np.nan,
        'p90_in': np.nanpercentile(vals_in, 90) if n_in else np.nan,
        'mean_border': np.nanmean(vals_border) if n_border else np.nan,
        'std_border': np.nanstd(vals_border) if n_border else np.nan,
        'contrast': (np.nanmean(vals_border) - np.nanmean(vals_in)) / np.nanmean(vals_in) if n_in and np.nanmean(vals_in) != 0 else np.nan,
        'n_in': n_in,
        'n_border': n_border
    }
    return stats

def analyze_phase_profile(anomaly_csv: str, dust_npy: str, hi_npy: str, out_dir: str, co_npy: str = None, zone_col: str = 'zone', border_width: float = 0.2, r_max: float = 2.0, n_bins: int = 20, n_control: int = 1000, seed: int = 42, min_npix: int = 10):
    """
    Фазовый морфологический анализ CMB-анoмалий: dust, HI, (CO).
    Сохраняет профили, статистики, гистограммы, корреляции, summary в out_dir.
    Все этапы логгируются. Анализ проводится по агрегированным кластерам.
    """
    logging.basicConfig(level=logging.INFO)
    # Проверка наличия файлов
    for path, name in [(anomaly_csv, 'CSV с аномалиями'), (dust_npy, 'dust npy'), (hi_npy, 'HI npy')]:
        if not os.path.exists(path):
            logging.error(f"[phase_profile] Не найден входной файл: {path} ({name})")
            raise FileNotFoundError(f"[phase_profile] Не найден входной файл: {path} ({name})")
        if os.path.getsize(path) == 0:
            logging.error(f"[phase_profile] Входной файл пуст: {path} ({name})")
            raise ValueError(f"[phase_profile] Входной файл пуст: {path} ({name})")
    if co_npy:
        if not os.path.exists(co_npy):
            logging.error(f"[phase_profile] Не найден входной файл: {co_npy} (CO npy)")
            raise FileNotFoundError(f"[phase_profile] Не найден входной файл: {co_npy} (CO npy)")
        if os.path.getsize(co_npy) == 0:
            logging.error(f"[phase_profile] Входной файл пуст: {co_npy} (CO npy)")
            raise ValueError(f"[phase_profile] Входной файл пуст: {co_npy} (CO npy)")
    # Проверка содержимого CSV
    df = pd.read_csv(anomaly_csv)
    if df.empty:
        logging.error(f"[phase_profile] CSV с аномалиями пуст: {anomaly_csv}")
        raise ValueError(f"[phase_profile] CSV с аномалиями пуст: {anomaly_csv}")
    # Проверка npy
    dust_map = array_load(dust_npy)
    hi_map = array_load(hi_npy)
    if dust_map.size == 0:
        logging.error(f"[phase_profile] dust npy пустой: {dust_npy}")
        raise ValueError(f"[phase_profile] dust npy пустой: {dust_npy}")
    if hi_map.size == 0:
        logging.error(f"[phase_profile] HI npy пустой: {hi_npy}")
        raise ValueError(f"[phase_profile] HI npy пустой: {hi_npy}")
    if dust_map.shape != hi_map.shape:
        logging.error(f"[phase_profile] Размеры карт не совпадают: dust {dust_map.shape}, HI {hi_map.shape}")
        raise ValueError(f"[phase_profile] Размеры карт не совпадают: dust {dust_map.shape}, HI {hi_map.shape}")
    nside = hp.npix2nside(len(dust_map))
    co_map = array_load(co_npy) if co_npy else None
    if co_npy and co_map.size == 0:
        logging.error(f"[phase_profile] CO npy пустой: {co_npy}")
        raise ValueError(f"[phase_profile] CO npy пустой: {co_npy}")
    if co_map is not None and co_map.shape != dust_map.shape:
        logging.error(f"[phase_profile] Размеры карт не совпадают: dust {dust_map.shape}, CO {co_map.shape}")
        raise ValueError(f"[phase_profile] Размеры карт не совпадают: dust {dust_map.shape}, CO {co_map.shape}")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    components = [('dust', dust_map), ('hi', hi_map)]
    if co_map is not None:
        components.append(('co', co_map))
    zones = df[zone_col].unique() if zone_col in df.columns else ['all']
    for zone in zones:
        dfa = df if zone == 'all' else df[df[zone_col] == zone]
        if dfa.empty:
            logging.info(f"[phase_profile] В зоне {zone} нет аномалий, пропускаю.")
            continue
        stats_all = {comp: [] for comp, _ in components}
        profiles_all = {comp: [] for comp, _ in components}
        logging.info(f"[phase_profile] Анализ аномалий, зона: {zone}, N={len(dfa)}")
        for _, row in tqdm(dfa.iterrows(), total=len(dfa), desc=f"Аномалии ({zone})"):
            for comp, arr in components:
                stats = extract_stats(row['l'], row['b'], row['radius_deg'], arr, nside, border_width, min_npix)
                stats_all[comp].append(stats)
                _, prof = extract_profile(row['l'], row['b'], row['radius_deg'], arr, nside, r_max, n_bins)
                profiles_all[comp].append(prof)
        # Сохраняем статистики
        for comp in stats_all:
            stats_df = pd.DataFrame(stats_all[comp])
            stats_npy = stats_df.to_numpy(dtype=np.float64)
            np.save(os.path.join(out_dir, f'{comp}_stats_{zone}.npy'), stats_npy)
            stats_df = stats_df.applymap(to_py_scalar)
            stats_df.to_csv(os.path.join(out_dir, f'{comp}_stats_{zone}.csv'), index=False)
        # Гистограммы контраста
        for comp in stats_all:
            plt.figure()
            plt.hist([s['contrast'] for s in stats_all[comp] if not np.isnan(s['contrast'])], bins=30, alpha=0.7)
            plt.xlabel('Contrast')
            plt.ylabel('Count')
            plt.title(f'{comp.upper()} contrast ({zone})')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{comp}_contrast_hist_{zone}.png'), dpi=200)
            plt.close()
        # Усреднённые профили
        for comp in profiles_all:
            if profiles_all[comp]:
                mean_profile = np.nanmean(profiles_all[comp], axis=0)
                r_bins, _ = extract_profile(dfa.iloc[0]['l'], dfa.iloc[0]['b'], dfa.iloc[0]['radius_deg'], components[0][1], nside, r_max, n_bins)
                plt.figure()
                plt.plot(r_bins, mean_profile, label=f'{comp.upper()} mean profile')
                plt.xlabel('Radius [deg]')
                plt.ylabel(f'{comp.upper()} value')
                plt.title(f'Radial {comp.upper()} profile ({zone})')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'{comp}_radial_profile_{zone}.png'), dpi=200)
                plt.close()
        logging.info(f"[phase_profile] Зона {zone}: {len(dfa)} аномалий обработано.")
    # Аналогично — контрольные точки (фон)
    for zone in zones:
        dfa = df if zone == 'all' else df[df[zone_col] == zone]
        if dfa.empty:
            logging.info(f"[phase_profile] В зоне {zone} нет аномалий для контроля, пропускаю.")
            continue
        controls_stats = {comp: [] for comp, _ in components}
        controls_profiles = {comp: [] for comp, _ in components}
        logging.info(f"[phase_profile] Контрольные точки, зона: {zone}, N={len(dfa)}, n_control={n_control}")
        for _, row in tqdm(dfa.iterrows(), total=len(dfa), desc=f"Контроль ({zone})"):
            for _ in range(n_control // max(1, len(dfa))):
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
                for comp, arr in components:
                    stats = extract_stats(l, b, r, arr, nside, border_width, min_npix)
                    controls_stats[comp].append(stats)
                    _, prof = extract_profile(l, b, r, arr, nside, r_max, n_bins)
                    controls_profiles[comp].append(prof)
        for comp in controls_stats:
            stats_df = pd.DataFrame(controls_stats[comp])
            stats_npy = stats_df.to_numpy(dtype=np.float64)
            np.save(os.path.join(out_dir, f'{comp}_stats_control_{zone}.npy'), stats_npy)
            stats_df = stats_df.applymap(to_py_scalar)
            stats_df.to_csv(os.path.join(out_dir, f'{comp}_stats_control_{zone}.csv'), index=False)
            plt.figure()
            plt.hist([s['contrast'] for s in controls_stats[comp] if not np.isnan(s['contrast'])], bins=30, alpha=0.7)
            plt.xlabel('Contrast')
            plt.ylabel('Count')
            plt.title(f'{comp.upper()} contrast (control, {zone})')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{comp}_contrast_hist_control_{zone}.png'), dpi=200)
            plt.close()
        for comp in controls_profiles:
            if controls_profiles[comp]:
                mean_profile = np.nanmean(controls_profiles[comp], axis=0)
                r_bins, _ = extract_profile(dfa.iloc[0]['l'], dfa.iloc[0]['b'], dfa.iloc[0]['radius_deg'], components[0][1], nside, r_max, n_bins)
                r_bins = np.array(r_bins, dtype=np.float64)
                mean_profile = np.array(mean_profile, dtype=np.float64)
                np.save(os.path.join(out_dir, f'{comp}_radial_profile_control_{zone}.npy'), np.stack([r_bins, mean_profile]))
                plt.figure()
                plt.plot(r_bins, mean_profile, label=f'{comp.upper()} mean profile (control)')
                plt.xlabel('Radius [deg]')
                plt.ylabel(f'{comp.upper()} value')
                plt.title(f'Radial {comp.upper()} profile (control, {zone})')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'{comp}_radial_profile_control_{zone}.png'), dpi=200)
                plt.close()
        logging.info(f"[phase_profile] Контрольные точки, зона {zone}: {len(dfa)} аномалий обработано.")
    # Корреляции профилей и summary (по зонам)
    for zone in zones:
        summary = {}
        for comp, _ in components:
            stats_path = os.path.join(out_dir, f'{comp}_stats_{zone}.csv')
            control_path = os.path.join(out_dir, f'{comp}_stats_control_{zone}.csv')
            if os.path.exists(stats_path) and os.path.exists(control_path):
                df_stats = pd.read_csv(stats_path)
                df_control = pd.read_csv(control_path)
                summary[f'{comp}_mean_contrast_anom'] = df_stats['contrast'].mean()
                summary[f'{comp}_mean_contrast_control'] = df_control['contrast'].mean()
                summary[f'{comp}_std_contrast_anom'] = df_stats['contrast'].std()
                summary[f'{comp}_std_contrast_control'] = df_control['contrast'].std()
        pd.DataFrame([summary]).to_csv(os.path.join(out_dir, f'phase_contrast_summary_{zone}.csv'), index=False) 
        logging.info(f"[phase_profile] Summary for zone {zone}: {summary}") 

def to_py_scalar(x):
    if hasattr(x, 'item'):
        return x.item()
    return x 