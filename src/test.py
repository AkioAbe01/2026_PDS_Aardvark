import time
import pandas as pd
from loader import load_image, load_mask, build_image_path, build_mask_path
from feature_C import get_color_feature


def estimate_feature_C_time(metadata_path="data/new_metadata.csv"):
    """
    Прогоняет feature_C на одной картинке, замеряет время
    и оценивает, сколько займёт обработка всех валидных картинок.
    """
    metadata = pd.read_csv(metadata_path)
    valid = metadata[metadata["Valid_mask"] == True]

    total = len(valid)
    if total == 0:
        print("Нет валидных картинок.")
        return

    row = valid.iloc[0]
    image_path = build_image_path(row)
    mask_path = build_mask_path(row)

    image = load_image(image_path)
    mask = load_mask(mask_path)

    start = time.perf_counter()
    features = get_color_feature(image, mask)
    elapsed = time.perf_counter() - start

    estimated_total = elapsed * total

    print(f"Картинка:            {row['img_id']}")
    print(f"Время на 1 картинку: {elapsed:.3f} сек")
    print(f"Всего картинок:      {total}")
    print(f"Оценка общего:       {estimated_total:.1f} сек "
          f"(~{estimated_total/60:.1f} мин)")

    return {
        "img_id": row["img_id"],
        "seconds_per_image": elapsed,
        "total_images": total,
        "estimated_total_seconds": estimated_total,
        "features": features,
    }


if __name__ == "__main__":
    estimate_feature_C_time()
