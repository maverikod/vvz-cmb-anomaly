import yaml

def load_known_regions_from_yaml(yaml_path):
    """
    Load known anomaly regions from a YAML file.
    Each region should have: l (deg), b (deg), radius_deg, type (str, optional)
    Returns: list of dicts with keys: l, b, radius_deg, type
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    # Ожидается список регионов или dict с ключом 'regions'
    if isinstance(data, dict) and 'regions' in data:
        regions = data['regions']
    elif isinstance(data, list):
        regions = data
    else:
        raise ValueError('YAML must contain a list of regions or a dict with key "regions"')
    # Проверка структуры
    for reg in regions:
        if not all(k in reg for k in ('l', 'b', 'radius_deg')):
            raise ValueError('Each region must have l, b, radius_deg')
    return regions 