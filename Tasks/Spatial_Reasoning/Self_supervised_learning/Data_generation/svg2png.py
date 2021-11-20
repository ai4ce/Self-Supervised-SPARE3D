import argparse
import os
import glob
import sys
import cairosvg
from PIL import Image
from cairosvg.surface import PNGSurface

from multiprocessing import pool, cpu_count
from functools import partial
from tqdm import tqdm

# import pdb;pdb.set_trace()

def svg2png_transparent_background(svg_name, args):
    try:
        png_name = svg_name.replace(".svg", '.png')
        with open(svg_name, 'rb') as svg_file:
            PNGSurface.convert(
                bytestring=svg_file.read(),
                write_to=open(png_name, 'wb'),
                output_width=int(args.width),
                output_height=int(args.height),
                )
        os.remove(svg_name)
    except:
        pass

def transparent_background2white(png_name):
    try:
        im = Image.open(png_name)
        fill_color = (255,255,255)  # new background color
        im = im.convert("RGBA")   # it had mode P after DL it from OP
        if im.mode in ('RGBA', 'LA'):
            background = Image.new(im.mode[:-1], im.size, fill_color)
            background.paste(im, im.split()[-1]) # omit transparency
            im = background
        im.convert("RGB").save(png_name)
    except:
        pass

def main(args):
    all_svg = glob.glob(args.file+ "/**/*.svg")
    f1 = partial(svg2png_transparent_background, args=args)
    p = pool.Pool(processes=args.n_cores)
    with p:
        r = list(tqdm(p.imap(f1, all_svg), total=len(all_svg)))
    p.close()
    p.join()

    all_png = glob.glob(args.file+ "/**/*.png")

    p = pool.Pool(processes=args.n_cores)
    with p:
        r = list(tqdm(p.imap(transparent_background2white, all_png), total=len(all_png)))
    p.close()
    p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('-f','--file',type=str,help='file or folder with svg files.')
    parser.add_argument('-W','--width',type=int,default=200,help='svg width.')
    parser.add_argument('-H','--height',type=int,default=200,help='svg height.')
    parser.add_argument('-n','--n_cores',type=int,default=cpu_count(),help='number of processors.')
    args = parser.parse_args(sys.argv[1:])
    main(args)

