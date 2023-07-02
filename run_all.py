import os
from pathlib import Path
import click


@click.command()
@click.option("--input_dir", "-i")
@click.option("--output_dir", "-o")
@click.option("--kernel_size", "-k", type=int, default=3)
@click.option("--palette_number", "-p", type=int, default=8)
@click.option("--dcq_path", "-d")
def main(input_dir, output_dir, kernel_size, palette_number,
         dcq_path):
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
        args = f'"{p.as_posix()}" "{out.as_posix()}" {kernel_size} {palette_number}'
        cmd = f'"{dcq.as_posix()}" {args}'
        print(f"running {cmd}")
        os.system(cmd)


if __name__ == '__main__':
    main()
