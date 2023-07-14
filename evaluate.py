import click
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from scipy.ndimage import convolve


def convolve_cw(x, k, mode="nearest"):
    return np.stack([convolve(x[:, :, i], k[:, :, i], mode=mode, cval=0) for i in range(x.shape[-1])], axis=-1)


def compute_evaluation_loss(a, b, W):
    blured_a = convolve_cw(a, W)
    blured_b = convolve_cw(b, W)
    diff = np.mean(np.sqrt(np.sum((blured_a - blured_b) ** 2, axis=-1)))
    return diff * 100 / np.sqrt(b.shape[-1])


def load_image(image_path):
    with Image.open(image_path) as image:
        np_image = np.array(image, dtype=np.float32)
    return np_image / 255


def initialize_W(ks, c, factor):
    half_distance = ks // 2
    sigma_squared = -half_distance / np.log(factor)
    W = np.zeros((ks, ks, c), dtype=np.float32)
    for ci in range(c):
        for i in range(ks):
            for j in range(ks):
                dist = np.sqrt((i - ks // 2) ** 2 + (j - ks // 2) ** 2)
                gaussian = np.exp(-dist / sigma_squared)
                W[i, j, ci] = gaussian
    W /= W.sum(axis=(0, 1))[np.newaxis, np.newaxis]
    return W


@click.command()
@click.option("--reference_dir", "-r")
@click.option("--input_dir", "-i")
@click.option("--kernel_size", "-k", type=int, default=3)
def main(reference_dir, input_dir, kernel_size):
    reference_path = Path(reference_dir)
    pngs = list(reference_path.rglob("*.png"))
    jpgs = list(reference_path.rglob("*.jpg"))
    tiffs = list(reference_path.rglob("*.tiff"))
    blur_kernel = initialize_W(kernel_size, 4, 0.125)

    input_path = Path(input_dir)

    ddict = defaultdict(list)
    for path in pngs + jpgs + tiffs:
        name = ".".join(path.name.split(".")[:-1])
        image = load_image(path)
        ddict[name].append((path, image))

    columns = ["Image"]
    for name in ddict.keys():
        for path in input_path.rglob(f"{name}.png"):
            columns.append(path.parts[-2])
            image = load_image(path)
            ddict[name].append((path, image))

    df_dicts = []
    for name, images in ddict.items():
        p0, im0 = images[0]
        bk = blur_kernel[:, :, :im0.shape[-1]]
        row_dict = {"Image": name}
        for i, (p, im) in enumerate(images[1:]):
            try:
                loss = compute_evaluation_loss(im0, im, bk)
                row_dict[p.parts[-2]] = loss
            except:
                print(f"Invalid shapes {name} {p.as_posix()}")
        df_dicts.append(row_dict)
    df = pd.DataFrame(df_dicts)
    df.to_csv("evaluation.csv", sep="\t", index=False)


if __name__ == '__main__':
    main()
