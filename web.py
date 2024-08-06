import argparse
import os

import rembg
import torch
from PIL import Image
from tqdm import tqdm

from sf3d.system import SF3D
from sf3d.utils import remove_background, resize_foreground
from bottle import route, run, template, request, static_file, url, get, post, response, error, abort, redirect, os
import datetime
import uuid

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
base_shape_dir = f"./output/web"

@get('/')
def upload():
    return '''
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="submit" value="Upload"></br>
            <input type="file" name="upload"></br>
        </form>
    '''

@route('/assets/<filepath:path>', name='assets')
def server_static(filepath):
    return static_file(filepath, root=base_shape_dir)

@route('/upload', method='POST')
def do_upload():
    upload = request.files.get('image', '')
    if not upload.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return 'File extension not allowed!'

    now = datetime.datetime.now(JST)
    d = '{:%Y%m%d%H%M%S}'.format(now)
    short_id = str(uuid.uuid4())[:8]
    dir_name = f"{d}_{short_id}"
    shape_dir = os.path.join(base_shape_dir, dir_name)
    os.makedirs(shape_dir, exist_ok=True)
    print(shape_dir)

    filename = upload.filename.lower()
    root, ext = os.path.splitext(filename)
    save_path = os.path.join(shape_dir, filename)
    upload.save(save_path, overwrite=True)

    # initialize the Segment Anything model
    input_raw = Image.open(save_path)

    image = [input_raw]
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            mesh, glob_dict = model.run_image(
                image,
                bake_resolution=args.texture_resolution,
                remesh=args.remesh_option,
            )
    print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")

    if len(image) == 1:
        out_mesh_path = os.path.join(shape_dir, "mesh.glb")
        mesh.export(out_mesh_path, include_normals=True)

    # utilize cost volume-based 3D reconstruction to generate textured 3D mesh
    body = {"status": 0, "data": f"http://20.168.237.190:8000/assets/{dir_name}/mesh.glb"}
    return body

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image", type=str, nargs="+", help="Path to input image(s) or folder."
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="Device to use. If no CUDA-compatible device is found, the baking will fail. Default: 'cuda:0'",
    )
    parser.add_argument(
        "--pretrained-model",
        default="stabilityai/stable-fast-3d",
        type=str,
        help="Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/stable-fast-3d'",
    )
    parser.add_argument(
        "--foreground-ratio",
        default=0.85,
        type=float,
        help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
    )
    parser.add_argument(
        "--output-dir",
        default="output/",
        type=str,
        help="Output directory to save the results. Default: 'output/'",
    )
    parser.add_argument(
        "--texture-resolution",
        default=1024,
        type=int,
        help="Texture atlas resolution. Default: 1024",
    )
    parser.add_argument(
        "--remesh_option",
        choices=["none", "triangle", "quad"],
        default="none",
        help="Remeshing option",
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size for inference"
    )
    args = parser.parse_args()

    # Ensure args.device contains cuda
    if "cuda" not in args.device:
        raise ValueError(
            "CUDA device is required for baking and hence running the method."
        )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"

    model = SF3D.from_pretrained(
        args.pretrained_model,
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.to(device)
    model.eval()
