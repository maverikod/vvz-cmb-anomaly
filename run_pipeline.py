import os
import shutil
from cmb_anomaly.convert import run_convert_command
from cmb_anomaly.multiscale import run_multiscale_anomaly_search
from cmb_anomaly.scale_clustering import run_scale_clustering_analysis
from cmb_anomaly.galactic_correlation import run_galactic_correlation_analysis
from cmb_anomaly.dust_correlation import run_dust_correlation_analysis
from cmb_anomaly.region_match import find_similar_regions, compare_anomaly_catalogs
from cmb_anomaly.dust_profile import analyze_dust_profile
from cmb_anomaly.phase_profile import analyze_phase_profile
from cmb_anomaly.cluster_postprocess import filter_unique_clusters
import argparse
import numpy as np
import healpy as hp
import sys
import traceback
import pandas as pd
import numpy as np
import astropy.io.fits as fits
import json
from cmb_anomaly.utils import get_centers
from cmb_anomaly.array_backend import print_backend_info

# Пути к исходным данным
CMB_FITS = 'data/COM_CMB_IQU-smica_2048_R3.00_full.fits'
MASK_FITS = 'data/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits'
DUST_FITS = 'data/COM_CompMap_dust-commrul_2048_R1.00.fits'
HI_FITS = 'data/NHI_HPX.fits.gz'
KNOWN_YAML = 'data/cmb_anomalies.yaml'  # если есть

# Каталоги результатов
DIR_CONVERT = 'results/01_convert'
DIR_MULTISCALE = 'results/02_multiscale'
DIR_CLUSTER = 'results/03_clustering'
DIR_GALCORR = 'results/04_galactic_corr'
DIR_DUSTCORR = 'results/05_dust_corr'
DIR_REGION = 'results/06_region_match'
DIR_COMPARE = 'results/07_catalog_compare'
DIR_DUSTPROFILE = 'results/08_dust_profile'
DIR_PHASEPROFILE = 'results/09_phase_profile'

CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)
for d in [DIR_CONVERT, DIR_MULTISCALE, DIR_CLUSTER, DIR_GALCORR, DIR_DUSTCORR, DIR_REGION, DIR_COMPARE, DIR_DUSTPROFILE, DIR_PHASEPROFILE]:
    os.makedirs(d, exist_ok=True)

def clear_cache():
    """Удаляет все файлы из каталога cache/ (кроме .gitkeep и подкаталогов)."""
    removed = 0
    for fname in os.listdir(CACHE_DIR):
        fpath = os.path.join(CACHE_DIR, fname)
        if os.path.isfile(fpath) and not fname.startswith('.gitkeep'):
            try:
                os.remove(fpath)
                print(f"[CACHE] Удалён: {fpath}")
                removed += 1
            except Exception as e:
                print(f"[WARNING] Не удалось удалить {fpath}: {e}")
    print(f"[CACHE] Очистка завершена. Удалено файлов: {removed}")

def clear_results():
    """Удаляет все файлы данных из results/*, кроме документации (README.md, *.md, *.txt)."""
    import glob
    removed = 0
    for root, dirs, files in os.walk('results'):
        for fname in files:
            if fname.lower().endswith(('.md', '.txt')) or fname.lower() == 'readme':
                continue
            fpath = os.path.join(root, fname)
            try:
                os.remove(fpath)
                print(f"[RESULTS] Удалён: {fpath}")
                removed += 1
            except Exception as e:
                print(f"[WARNING] Не удалось удалить {fpath}: {e}")
    print(f"[RESULTS] Очистка завершена. Удалено файлов: {removed}")

def clear_all():
    """Полная очистка: удаляет все файлы данных из cache/ и results/*, кроме документации."""
    clear_cache()
    clear_results()

def get_nside_from_npy(path):
    arr = np.load(path, mmap_mode='r')
    npix = arr.shape[0]
    nside = hp.npix2nside(npix)
    return nside, npix

def get_nside_from_fits(path, column=None):
    import astropy.io.fits as fits
    with fits.open(path) as hdul:
        if column is not None:
            arr = hdul[1].data[column]
        else:
            arr = hdul[1].data[0] if hasattr(hdul[1].data, '__getitem__') else hdul[1].data
        npix = arr.shape[0]
        nside = hp.npix2nside(npix)
    return nside, npix

def ensure_npy(path_fits, path_npy, column=None):
    if not os.path.exists(path_npy):
        if column:
            run_convert_command(path_fits, path_npy)
        else:
            run_convert_command(path_fits, path_npy)
    return path_npy

def downgrade_to_nside(path_in, nside_target, path_out, is_mask=False):
    arr = np.load(path_in)
    arr_out = hp.ud_grade(arr.astype(float), nside_out=nside_target)
    if is_mask:
        arr_out = arr_out > 0.5
    np.save(path_out, arr_out)
    return path_out

def is_file_valid(path, filetype=None):
    """
    Проверяет существование и читаемость файла.
    filetype: 'npy', 'csv', 'fits' или None (определяется по расширению)
    Возвращает True, если файл валиден, иначе False.
    """
    if not os.path.exists(path):
        return False
    try:
        if filetype is None:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.npy':
                filetype = 'npy'
            elif ext == '.csv':
                filetype = 'csv'
            elif ext in ['.fits', '.fit', '.fz', '.gz']:
                filetype = 'fits'
        if filetype == 'npy':
            arr = np.load(path, mmap_mode='r')
            _ = arr.shape
        elif filetype == 'csv':
            df = pd.read_csv(path, nrows=5)
        elif filetype == 'fits':
            with fits.open(path) as hdul:
                _ = hdul[0].data
        else:
            with open(path, 'rb') as f:
                f.read(10)
        return True
    except Exception:
        return False

def fail(msg, step=None, file=None, exc=None):
    print("\n[ОШИБКА]" + (f" [{step}]" if step else "") + (f" Файл: {file}" if file else ""))
    print(msg)
    if exc:
        print("Причина:", str(exc))
    sys.exit(1)

def safe_remove(path):
    """Удаляет файл, если он существует."""
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"[INFO] Удалён старый файл: {path}")
    except Exception as e:
        print(f"[WARNING] Не удалось удалить файл {path}: {e}")

def load_config(config_path=None):
    config_files = [config_path, 'config.json', 'config.example.json'] if config_path else ['config.json', 'config.example.json']
    for fname in config_files:
        if fname and os.path.exists(fname):
            with open(fname, 'r') as f:
                return json.load(f)
    return {}

# Описание зависимостей между шагами
STEP_DEPENDENCIES = {
    'convert': [],
    'multiscale': ['convert'],
    'clustering': ['multiscale'],
    'galactic_corr': ['clustering'],
    'dust_corr': ['galactic_corr'],
    'region_match': ['galactic_corr'],
    'catalog_compare': ['galactic_corr', 'dust_corr'],
    'dust_profile': ['galactic_corr'],
    'phase_profile': ['galactic_corr', 'dust_corr'],
}

STEP_OUTPUTS = {
    'convert': [
        os.path.join(CACHE_DIR, 'cmb_temperature.npy'),
        os.path.join(CACHE_DIR, 'mask_common.npy'),
        os.path.join(CACHE_DIR, 'dust_map.npy'),
        os.path.join(CACHE_DIR, 'hi_map.npy'),
    ],
    'multiscale': [os.path.join(DIR_MULTISCALE, 'results_anomalies_multi.csv')],
    'clustering': [os.path.join(DIR_CLUSTER, 'unique_clusters.csv')],
    'galactic_corr': [os.path.join(DIR_GALCORR, 'galactic_corr_anomalies_1deg.csv')],
    'dust_corr': [os.path.join(DIR_DUSTCORR, 'dust_corr_anomalies_halo_1deg.csv')],
    'region_match': [os.path.join(DIR_REGION, 'region_matches.csv')],
    'catalog_compare': [os.path.join(DIR_COMPARE, 'matched_cmb_dust_anomalies.csv')],
    'dust_profile': [os.path.join(DIR_DUSTPROFILE, 'dust_contrast_summary.csv')],
    'phase_profile': [os.path.join(DIR_PHASEPROFILE, 'phase_contrast_summary_halo.csv')],
}

STEP_FUNCS = {}

# Индикатор расчёта/конвертации/использования кэша

def print_step_indicator(step, action):
    print(f"[STEP] {step}: {action}")

# Рекурсивная проверка и запуск зависимостей

def ensure_step_outputs(step, call_stack=None, use_mask=False, s_threshold=5.0, step_deg=5, top_n=20, radii_deg=None, centers=None):
    if call_stack is None:
        call_stack = []
    if step in call_stack:
        fail(f"Обнаружено зацикливание шагов: {' -> '.join(call_stack + [step])}", step=step)
    call_stack.append(step)
    # Проверяем зависимости
    for dep in STEP_DEPENDENCIES.get(step, []):
        ensure_step_outputs(dep, call_stack, use_mask=use_mask, s_threshold=s_threshold, step_deg=step_deg, top_n=top_n, radii_deg=radii_deg, centers=centers)
    # Проверяем выходные файлы шага
    for out in STEP_OUTPUTS.get(step, []):
        if not is_file_valid(out):
            print_step_indicator(step, f"выходной файл {out} отсутствует или повреждён — выполняю шаг")
            run_step(step, use_mask=use_mask, s_threshold=s_threshold, step_deg=step_deg, top_n=top_n, radii_deg=radii_deg, centers=centers)
            break
        else:
            print_step_indicator(step, f"использую кэш {out}")
    call_stack.pop()

# Обёртка для запуска шага

def run_step(step, use_mask=False, s_threshold=5.0, step_deg=5, top_n=20, radii_deg=None, centers=None):
    if step == 'convert':
        # Конвертация FITS→NPY
        for path, name, src in [
            (os.path.join(CACHE_DIR, 'cmb_temperature.npy'), 'cmb_temperature.npy', CMB_FITS),
            (os.path.join(CACHE_DIR, 'mask_common.npy'), 'mask_common.npy', MASK_FITS),
            (os.path.join(CACHE_DIR, 'dust_map.npy'), 'dust_map.npy', DUST_FITS),
            (os.path.join(CACHE_DIR, 'hi_map.npy'), 'hi_map.npy', HI_FITS)
        ]:
            if not is_file_valid(path, 'npy'):
                safe_remove(path)
                print_step_indicator(step, f"конвертация {src} → {name}")
                try:
                    ensure_npy(src, path)
                except Exception as e:
                    fail(f"Не удалось сконвертировать {src} в {name}", step=step, file=src, exc=e)
            else:
                print_step_indicator(step, f"использую кэш {name}")
    elif step == 'multiscale':
        temperature_npy = os.path.join(CACHE_DIR, 'cmb_temperature.npy')
        mask_npy = os.path.join(CACHE_DIR, 'mask_common.npy')
        multiscale_csv = os.path.join(DIR_MULTISCALE, 'results_anomalies_multi.csv')
        if not is_file_valid(multiscale_csv, 'csv'):
            safe_remove(multiscale_csv)
            print_step_indicator(step, "расчёт поиска аномалий (multiscale)")
            try:
                import healpy as hp
                arr = np.load(temperature_npy, mmap_mode='r')
                nside = hp.npix2nside(arr.shape[0])
                if centers is None:
                    centers = get_centers(nside, step_deg)
                if use_mask:
                    print('[INFO] Маска используется для multiscale')
                    run_multiscale_anomaly_search(temperature_npy, mask_npy, multiscale_csv, step_deg=step_deg, top_n=top_n, s_threshold=s_threshold, radii_deg=radii_deg, centers=centers)
                else:
                    print('[INFO] Маска НЕ используется для multiscale')
                    run_multiscale_anomaly_search(temperature_npy, None, multiscale_csv, step_deg=step_deg, top_n=top_n, s_threshold=s_threshold, radii_deg=radii_deg, centers=centers)
            except Exception as e:
                fail("Не удалось выполнить поиск аномалий (multiscale)", step=step, file=multiscale_csv, exc=e)
        else:
            print_step_indicator(step, "использую кэш results_anomalies_multi.csv")
    elif step == 'clustering':
        multiscale_csv = os.path.join(DIR_MULTISCALE, 'results_anomalies_multi.csv')
        unique_csv = os.path.join(DIR_CLUSTER, 'unique_clusters.csv')
        if not is_file_valid(unique_csv, 'csv'):
            safe_remove(unique_csv)
            print_step_indicator(step, "фильтрация уникальных кластеров")
            try:
                filter_unique_clusters(multiscale_csv, unique_csv)
            except Exception as e:
                fail("Не удалось выполнить фильтрацию уникальных кластеров", step=step, file=unique_csv, exc=e)
        else:
            print_step_indicator(step, "использую кэш unique_clusters.csv")
    elif step == 'galactic_corr':
        multiscale_csv = os.path.join(DIR_MULTISCALE, 'results_anomalies_multi.csv')
        gal_prefix = os.path.join(DIR_GALCORR, 'galactic_corr')
        if not os.path.exists(f'{gal_prefix}_anomalies_1deg.csv'):
            print_step_indicator(step, "расчёт галактической корреляции")
            try:
                run_galactic_correlation_analysis(multiscale_csv, radii=[1,5,25], out_prefix=gal_prefix, top_n=20)
            except Exception as e:
                fail("Не удалось выполнить галактическую корреляцию", step=step, file=f'{gal_prefix}_anomalies_1deg.csv', exc=e)
        else:
            print_step_indicator(step, "использую кэш galactic_corr_anomalies_1deg.csv")
    elif step == 'dust_corr':
        gal_prefix = os.path.join(DIR_GALCORR, 'galactic_corr')
        dust_prefix = os.path.join(DIR_DUSTCORR, 'dust_corr')
        if not os.path.exists(f'{dust_prefix}_anomalies_halo_1deg.csv'):
            print_step_indicator(step, "расчёт корреляции с картой пыли")
            try:
                import healpy as hp
                arr = np.load(os.path.join(CACHE_DIR, 'dust_map.npy'), mmap_mode='r')
                nside = hp.npix2nside(arr.shape[0])
                if centers is None:
                    centers = get_centers(nside, step_deg)
                run_multiscale_anomaly_search_dust(os.path.join(CACHE_DIR, 'dust_map.npy'), None, f'{dust_prefix}_anomalies_halo_1deg.csv', step_deg=step_deg, top_n=top_n, radii_deg=radii_deg, centers=centers)
            except Exception as e:
                fail("Не удалось выполнить корреляцию с картой пыли", step=step, file=f'{dust_prefix}_anomalies_halo_1deg.csv', exc=e)
        else:
            print_step_indicator(step, "использую кэш dust_corr_anomalies_halo_1deg.csv")
    elif step == 'region_match':
        gal_prefix = os.path.join(DIR_GALCORR, 'galactic_corr')
        if os.path.exists(KNOWN_YAML):
            region_csv = os.path.join(DIR_REGION, 'region_matches.csv')
            if not os.path.exists(region_csv):
                print_step_indicator(step, "сопоставление с известными аномалиями")
                try:
                    find_similar_regions(f'{gal_prefix}_anomalies_1deg.csv', KNOWN_YAML, output_csv=region_csv, radius_tol=0.2, top_n=3)
                except Exception as e:
                    fail("Не удалось выполнить сопоставление с известными аномалиями", step=step, file=region_csv, exc=e)
        else:
            print_step_indicator(step, "использую кэш region_matches.csv")
    elif step == 'catalog_compare':
        cmb_csv = f'{gal_prefix}_anomalies_1deg.csv'
        dust_csv = f'{dust_prefix}_anomalies_halo_1deg.csv'
        compare_csv = os.path.join(DIR_COMPARE, 'matched_cmb_dust_anomalies.csv')
        if os.path.exists(cmb_csv) and os.path.exists(dust_csv):
            if not os.path.exists(compare_csv):
                print_step_indicator(step, "сравнение каталогов CMB/dust")
                try:
                    compare_anomaly_catalogs(cmb_csv, dust_csv, output_csv=compare_csv, max_dist_deg=2.0, radius_tol=0.2, top_n=1)
                except Exception as e:
                    fail("Не удалось выполнить сравнение каталогов", step=step, file=compare_csv, exc=e)
        else:
            print_step_indicator(step, "использую кэш matched_cmb_dust_anomalies.csv")
    elif step == 'dust_profile':
        anomaly_csv = f'{gal_prefix}_anomalies_1deg.csv'
        if os.path.exists(anomaly_csv) and os.path.exists(os.path.join(CACHE_DIR, 'dust_map.npy')):
            print_step_indicator(step, "морфология и профили пыли вокруг CMB-аномалий")
            try:
                analyze_dust_profile(anomaly_csv, os.path.join(CACHE_DIR, 'dust_map.npy'), DIR_DUSTPROFILE)
            except Exception as e:
                fail("Не удалось выполнить анализ пыли", step=step, file=DIR_DUSTPROFILE, exc=e)
        else:
            print_step_indicator(step, "использую кэш dust_contrast_summary.csv")
    elif step == 'phase_profile':
        anomaly_csv = f'{gal_prefix}_anomalies_1deg.csv'
        if os.path.exists(anomaly_csv) and os.path.exists(os.path.join(CACHE_DIR, 'dust_map.npy')) and os.path.exists(os.path.join(CACHE_DIR, 'hi_map.npy')):
            print_step_indicator(step, "фазовый морфологический анализ (dust, HI, CO)")
            try:
                analyze_phase_profile(anomaly_csv, os.path.join(CACHE_DIR, 'dust_map.npy'), os.path.join(CACHE_DIR, 'hi_map.npy'), DIR_PHASEPROFILE)
            except Exception as e:
                fail("Не удалось выполнить фазовый анализ", step=step, file=DIR_PHASEPROFILE, exc=e)
        else:
            print_step_indicator(step, "использую кэш phase_contrast_summary_halo.csv")


def main():
    print_backend_info()
    parser = argparse.ArgumentParser(
        description="CMB Anomaly Full Pipeline. Всё проходит автоматически по данным из data/.\n\n"
                    "Для ускорения на GPU (CUDA) установите пакет cupy-cuda12x.\n"
                    "Если CuPy не установлен или не видит GPU, будет использоваться CPU (NumPy).\n"
                    "Подробнее — см. раздел 'Ускорение на GPU (CUDA)' в README.md."
    )
    parser.add_argument('--config', type=str, default=None, help='Путь к конфигу (JSON)')
    parser.add_argument('--s-threshold', type=float, default=None, help='Порог S для аномалий (default: из конфига или 5.0)')
    parser.add_argument('--step-deg', type=int, default=None, help='Шаг сетки по небу (default: из конфига или 5)')
    parser.add_argument('--top-n', type=int, default=None, help='Сколько аномалий выводить (default: из конфига или 20)')
    parser.add_argument('--radii-deg', type=str, default=None, help='Список радиусов через запятую (default: из конфига или 1-15)')
    parser.add_argument('--only', type=str, default=None, choices=[
        'convert', 'multiscale', 'clustering', 'galactic_corr', 'dust_corr', 'region_match', 'catalog_compare', 'dust_profile', 'phase_profile'
    ], help='Выполнить только указанный шаг (и его зависимости, если нужны).')
    parser.add_argument('--clear-cache', action='store_true', help='Очистить все промежуточные файлы из cache/')
    parser.add_argument('--clear-results', action='store_true', help='Удалить все файлы данных из results/* (кроме документации)')
    parser.add_argument('--clear-all', action='store_true', help='Полная очистка: удалить все данные из cache/ и results/* (кроме документации)')
    parser.add_argument('--use-mask', action='store_true', help='Использовать маску при поиске аномалий (по умолчанию выключено)')
    parser.add_argument('--run', action='store_true', help='Запустить полный пайплайн (самый низкий приоритет)')
    args = parser.parse_args()

    # Если только --help, выводим подробную справку и завершаем работу
    if len(sys.argv) == 2 and (sys.argv[1] == '--help' or sys.argv[1] == '-h'):
        parser.print_help()
        return

    # Если нет ни одного ключа кроме help/run — выводим краткую справку
    if not (args.run or args.only or args.clear_cache or args.clear_results or args.clear_all or args.use_mask):
        print("""
CMB Anomaly Full Pipeline

Назначение:
  Автоматизация полного анализа аномалий на карте CMB (Planck SMICA).
  Для запуска используйте ключ --run или --help для справки.
  Подробнее: python3 run_pipeline.py --help
""")
        return

    config = load_config(args.config)
    def get_param(name, default=None, cast=None):
        val = getattr(args, name, None)
        if val is not None:
            if cast and val is not None:
                return cast(val)
            return val
        if name in config:
            return config[name]
        return default
    s_threshold = get_param('s_threshold', 5.0, float)
    step_deg = get_param('step_deg', 5, int)
    top_n = get_param('top_n', 20, int)
    if args.radii_deg:
        radii_deg = [int(x) for x in args.radii_deg.split(',')]
    elif 'radii_deg' in config:
        radii_deg = config['radii_deg']
    else:
        radii_deg = list(range(1, 16))

    # clear, only, use-mask — приоритет выше run
    if args.clear_all:
        clear_all()
        print('Полная очистка завершена. Завершение работы.')
        return
    if args.clear_results:
        clear_results()
        print('Очистка результатов завершена. Завершение работы.')
        return
    if args.clear_cache:
        clear_cache()
        print('Кэш очищен. Завершение работы.')
        return
    if args.only:
        ensure_step_outputs(args.only, use_mask=args.use_mask, s_threshold=s_threshold, step_deg=step_deg, top_n=top_n, radii_deg=radii_deg, centers=None)
        print(f'Шаг {args.only} и все зависимости успешно выполнены.')
        return
    # --run запускает полный пайплайн
    if args.run:
        for step in ['convert', 'multiscale', 'clustering', 'galactic_corr', 'dust_corr', 'region_match', 'catalog_compare', 'dust_profile', 'phase_profile']:
            ensure_step_outputs(step, use_mask=args.use_mask, s_threshold=s_threshold, step_deg=step_deg, top_n=top_n, radii_deg=radii_deg, centers=None)
        print('\nПайплайн завершён. Все результаты разложены по каталогам results/.')
        return

if __name__ == "__main__":
    main() 