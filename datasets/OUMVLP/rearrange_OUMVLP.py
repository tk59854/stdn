import argparse
from concurrent.futures import process
import os
from pathlib import Path
from typing import Union
import multiprocessing as mp
from functools import partial
import logging

from tqdm import tqdm


def mkdir(path):
    if not os.path.exists(path=path):
        os.makedirs(path)


def sanitize(name: str) -> Union[str, str]:
    return name.split('_')[1].split('-')


def rearrange(input_path: Path, output_path: Path) -> None:

    mkdir(output_path)

    total = [list(input_path.iterdir())[-1]]
    print('total: ', total)
    for folder in total:
        print(f'Rearranging {folder}')
        view, seq = sanitize(folder.name)
        progress = tqdm(total=len(os.listdir(folder)))
        for sid in folder.iterdir():
            print(sid.name)
        #     src = os.path.join(input_path, f'Silhouette_{view}-{seq}', sid.name)
        #     dst = os.path.join(output_path, sid.name, seq, view)
        #     print(src)
        #     print(dst)
        #     mkdir(dst)
        #     for subfile in os.listdir(src):
        #         if subfile not in os.listdir(dst) and subfile.endswith('.png'):
        #             os.symlink(os.path.join(src, subfile),
        #                        os.path.join(dst, subfile))
        #     progress.update(1)


def rearrange_(_id, root_path, output_path):
    logging.info('rearranging {} ... '.format(_id))
    view, seq = sanitize(root_path.name)
    src = os.path.join(root_path, _id)
    dst = os.path.join(output_path, _id, seq, view)
    
    mkdir(dst)
    for subfile in os.listdir(src):
        if subfile not in os.listdir(dst) and subfile.endswith('.png'):
            os.symlink(os.path.join(src, subfile),
                       os.path.join(dst, subfile))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OUMVLP rearrange tool')
    parser.add_argument('-i', '--input_path', required=True, type=str,
                        help='Root path of raw dataset.')
    parser.add_argument('-o', '--output_path', default='OUMVLP_rearranged', type=str,
                        help='Root path for output.')

    args = parser.parse_args()
    
    workers = 8
    
    logging.basicConfig(level=logging.INFO, filename='rearrange.log', filemode='w', format='[%(asctime)s - %(levelname)s]: %(message)s')

    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()
    # rearrange(input_path, output_path)
    
    total = [list(input_path.iterdir())[-1]]
    
    for folder in total:
        logging.info(f'Start rearrange {folder}')
        all_id = os.listdir(folder)
        progress = tqdm(total=len(all_id))
        with mp.Pool(workers) as pool:
            for _ in pool.map(partial(rearrange_, root_path=folder, output_path=output_path), all_id):
                progress.update(1)
            logging.info('Done!')
                
