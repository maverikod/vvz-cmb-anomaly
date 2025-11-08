import pandas as pd
import numpy as np
import logging
from .known_regions import load_known_regions_from_yaml
from .array_backend import cp, np, array_load, array_save

def filter_and_sort_matches(df_sub, l0, b0, r0, radius_tol, top_n, unique_region=False, used_indices=None):
    """
    Фильтрует и сортирует совпадения по угловому расстоянию и относительной разнице радиусов.
    Если unique_region=True, не повторяет совпадения по одному региону.
    used_indices: set индексов уже сопоставленных аномалий.
    """
    dists = angular_distance_deg(l0, b0, df_sub['l'], df_sub['b'])
    df_sub = df_sub.assign(similarity_score=dists)
    df_sub = df_sub.sort_values('similarity_score')
    if unique_region and used_indices is not None:
        df_sub = df_sub[~df_sub.index.isin(used_indices)]
    df_sub = df_sub.head(top_n)
    return df_sub

def find_similar_regions(anomaly_csv, known_yaml, output_csv=None, radius_tol=0.2, type_key='type', top_n=3, unique_region=False, summary_report=True):
    """
    Для каждого известного региона из yaml ищет top-N наиболее похожих регионов среди найденных аномалий (по координатам и радиусу).
    Похожесть определяется по расстоянию на сфере и относительной разнице радиусов.
    Если указан type, ищет только среди совпадающих типов (если есть).
    unique_region: если True, не повторяет совпадения по одному и тому же региону аномалий.
    Сохраняет результат в output_csv (если задан).
    Возвращает DataFrame с результатами.
    """
    logging.basicConfig(level=logging.INFO)
    known_regions = load_known_regions_from_yaml(known_yaml)
    df = pd.read_csv(anomaly_csv)
    results = []
    used_indices = set()
    for reg in known_regions:
        l0, b0, r0 = reg['l'], reg['b'], reg['radius_deg']
        reg_type = reg.get(type_key, None)
        # Фильтрация по типу (если есть)
        df_sub = df.copy()
        if reg_type is not None and type_key in df_sub.columns:
            df_sub = df_sub[df_sub[type_key] == reg_type]
        # Фильтрация по радиусу (относительное отклонение)
        df_sub = df_sub[np.abs(df_sub['radius_deg'] - r0) / r0 <= radius_tol]
        if df_sub.empty:
            continue
        df_sub = filter_and_sort_matches(df_sub, l0, b0, r0, radius_tol, top_n, unique_region, used_indices)
        for _, row in df_sub.iterrows():
            res = dict(reg)
            for col in df.columns:
                res[f'found_{col}'] = row[col]
            res['distance_deg'] = row['similarity_score']
            results.append(res)
            if unique_region:
                used_indices.add(row.name)
    out_df = pd.DataFrame(results)
    if output_csv:
        out_df.to_csv(output_csv, index=False)
    if summary_report:
        log_match_summary(out_df, stage="region_match")
    return out_df

def angular_distance_deg(l1, b1, l2, b2):
    """
    Вычисляет угловое расстояние (deg) между (l1, b1) и массивами (l2, b2) на сфере.
    """
    l1 = np.deg2rad(l1)
    b1 = np.deg2rad(b1)
    l2 = np.deg2rad(np.array(l2))
    b2 = np.deg2rad(np.array(b2))
    # Формула сферического косинуса
    cos_d = np.sin(b1)*np.sin(b2) + np.cos(b1)*np.cos(b2)*np.cos(l1 - l2)
    cos_d = np.clip(cos_d, -1, 1)
    return np.rad2deg(np.arccos(cos_d)) 

def compare_anomaly_catalogs(csv1, csv2, output_csv=None, max_dist_deg=2.0, radius_tol=0.2, top_n=1, unique_region=False, summary_report=True):
    """
    Сравнивает два каталога аномалий (например, CMB и dust) и ищет совпадающие регионы.
    Совпадение: угловое расстояние < max_dist_deg и относительная разница радиусов < radius_tol.
    Для каждого региона из csv1 ищет top-N ближайших из csv2.
    unique_region: если True, не повторяет совпадения по одному региону.
    Сохраняет результат в output_csv (если задан).
    Возвращает DataFrame с совпадениями.
    """
    logging.basicConfig(level=logging.INFO)
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    results = []
    used_indices = set()
    for _, row1 in df1.iterrows():
        l1, b1, r1 = row1['l'], row1['b'], row1['radius_deg']
        # Фильтрация по радиусу
        df2_sub = df2[np.abs(df2['radius_deg'] - r1) / r1 <= radius_tol]
        if df2_sub.empty:
            continue
        dists = angular_distance_deg(l1, b1, df2_sub['l'], df2_sub['b'])
        df2_sub = df2_sub.assign(distance_deg=dists)
        df2_sub = df2_sub[df2_sub['distance_deg'] <= max_dist_deg]
        if unique_region:
            df2_sub = df2_sub[~df2_sub.index.isin(used_indices)]
        if df2_sub.empty:
            continue
        df2_sub = df2_sub.sort_values('distance_deg').head(top_n)
        for _, row2 in df2_sub.iterrows():
            res = {f'cmb_{k}': v for k, v in row1.items()}
            for k, v in row2.items():
                res[f'dust_{k}'] = v
            res['distance_deg'] = row2['distance_deg']
            results.append(res)
            if unique_region:
                used_indices.add(row2.name)
    out_df = pd.DataFrame(results)
    if output_csv:
        out_df.to_csv(output_csv, index=False)
    if summary_report:
        log_match_summary(out_df, stage="catalog_match")
    return out_df 

def log_match_summary(df, stage=""):
    """
    Логгирует summary-отчёт по совпадениям: число совпадений, min/mean/max по расстояниям, радиусам, типам.
    """
    if df.empty:
        logging.info(f"[{stage}] No matches found.")
        return
    if 'distance_deg' in df.columns:
        d = df['distance_deg']
        logging.info(f"[{stage}] distance_deg: min={d.min():.3f}, mean={d.mean():.3f}, max={d.max():.3f}")
    if 'found_radius_deg' in df.columns:
        r = df['found_radius_deg']
        logging.info(f"[{stage}] found_radius_deg: min={r.min():.3f}, mean={r.mean():.3f}, max={r.max():.3f}")
    if 'radius_deg' in df.columns:
        r = df['radius_deg']
        logging.info(f"[{stage}] radius_deg: min={r.min():.3f}, mean={r.mean():.3f}, max={r.max():.3f}")
    logging.info(f"[{stage}] total matches: {len(df)}") 