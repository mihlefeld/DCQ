import os
from pathlib import Path
import click


@click.command()
@click.option("--input_dir", "-i", type=click.Path())
@click.option("--output_dir", "-o", type=click.Path())
@click.option("--kernel_size", "-k", type=int, default=3, help="Specify the kernel size used by the DCQ approach.")
@click.option("--palette_number", "-p", type=int, default=8,
              help="Specify the maximum number of colors used for quantization")
@click.option("--dcq_path", "-d", type=click.Path(), default="DCQ", help="Specify the path to the DCQ executable.")
@click.option("--mode", "-m", type=click.Choice(["DCQ", "FS", "FS-ICM", "FS-ICM-DCQ", "FS-DCQ"]),
              help="Specify which dithering algorithm to use.")
@click.option("--alpha_mode", "-a", type=click.Choice(["divide", "quantize", "none"]), default="none",
              help="Specify how the program should handle the alpha channel.")
@click.option("--cluster_mode", "-c", default="k_means", type=click.Choice(["k_means", "median_cuts"]),
              help="Specify the clustering algorithm to initialize the palette for Floyd Steinberg Dithering")
def main(input_dir, output_dir, kernel_size, palette_number,
         dcq_path, mode, alpha_mode, cluster_mode):
    path = Path(input_dir)
    out = Path(output_dir)
    dcq = Path(dcq_path)
    pngs = list(path.rglob("*.png"))
    jpgs = list(path.rglob("*.jpg"))
    globs = pngs + jpgs
    print(globs)
    for p in globs:
        name = ".".join(p.name.split(".")[:-1])
        name += ".png"
        args = f'"{p.as_posix()}" "{out.as_posix()}" {kernel_size} {palette_number} {mode} {alpha_mode} {cluster_mode}'
        cmd = f'"{dcq.absolute().as_posix()}" {args}'
        print(f"running {cmd}")
        os.system(cmd)


if __name__ == '__main__':
    main()
