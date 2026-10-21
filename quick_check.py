from src.api.category_map import get_category_map

categories = get_category_map(region_code="IN")
for cid, name in categories.items():
    print(cid, "-", name)