import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
from typing import *
import json
import warnings

import click


@click.command(help='Measure distance between two points in an image')
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True), required=True, help='Input image path. "jpg" and "png" are supported.')
@click.option('--point1', '-p1', 'point1', type=str, required=True, help='First point coordinates in format "x1,y1"')
@click.option('--point2', '-p2', 'point2', type=str, required=True, help='Second point coordinates in format "x2,y2"')
@click.option('--output', '-o', 'output_path', default=None, type=click.Path(), help='Output folder path. If not provided, results will only be printed.')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default=None, help='Pretrained model name or path. If not provided, the corresponding default model will be chosen.')
@click.option('--version', 'model_version', type=click.Choice(['v1', 'v2']), default='v2', help='Model version. Defaults to "v2"')
@click.option('--device', 'device_name', type=str, default='cuda', help='Device name (e.g. "cuda", "cuda:0", "cpu"). Defaults to "cuda"')
@click.option('--fp16', 'use_fp16', is_flag=True, help='Use fp16 precision for faster inference.')
@click.option('--raw', 'raw_output', is_flag=True, help='Output only the numerical distance value without any formatting.')
def main(
    input_path: str,
    point1: str,
    point2: str,
    output_path: str,
    pretrained_model_name_or_path: str,
    model_version: str,
    device_name: str,
    use_fp16: bool,
    raw_output: bool,
):  
    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm

    from moge.model import import_model_class_by_version
    import utils3d

    # Parse coordinates
    try:
        x1, y1 = map(int, point1.split(','))
        x2, y2 = map(int, point2.split(','))
    except ValueError:
        raise click.BadParameter('Invalid coordinate format. Use "x,y" format for points.')

    device = torch.device(device_name)

    # Load image
    if not Path(input_path).exists():
        raise FileNotFoundError(f'File {input_path} does not exist.')
    image = cv2.cvtColor(cv2.imread(str(input_path)), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # Validate coordinates
    if x1 < 0 or x1 >= width or y1 < 0 or y1 >= height:
        raise click.BadParameter(f'First point ({x1}, {y1}) is out of image bounds. Image size: {width}x{height}')
    if x2 < 0 or x2 >= width or y2 < 0 or y2 >= height:
        raise click.BadParameter(f'Second point ({x2}, {y2}) is out of image bounds. Image size: {width}x{height}')

    # Load model
    if pretrained_model_name_or_path is None:
        DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION = {
            "v1": "Ruicheng/moge-vitl",
            "v2": "Ruicheng/moge-2-vitl-normal",
        }
        pretrained_model_name_or_path = DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION[model_version]
    model = import_model_class_by_version(model_version).from_pretrained(pretrained_model_name_or_path).to(device).eval()
    if use_fp16:
        model.half()

    # Convert image to tensor
    image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

    # Inference with Ultra resolution level (30)
    output = model.infer(image_tensor, resolution_level=30, use_fp16=use_fp16)
    points = output['points'].cpu().numpy()
    depth = output['depth'].cpu().numpy()
    mask = output['mask'].cpu().numpy()

    # Check if points are valid (within mask)
    if not mask[y1, x1]:
        raise click.BadParameter(f'First point ({x1}, {y1}) is in invalid region (outside mask).')
    if not mask[y2, x2]:
        raise click.BadParameter(f'Second point ({x2}, {y2}) is in invalid region (outside mask).')

    # Calculate distance
    point1_3d = points[y1, x1]
    point2_3d = points[y2, x2]
    distance = np.linalg.norm(point1_3d - point2_3d)

    # Get depths
    depth1 = depth[y1, x1]
    depth2 = depth[y2, x2]

    # Print results
    if raw_output:
        print(f"{distance:.2f}")
    else:
        print(f"Measurement Results:")
        print(f"Image: {input_path}")
        print(f"Point 1: ({x1}, {y1}) - Depth: {depth1:.2f}m")
        print(f"Point 2: ({x2}, {y2}) - Depth: {depth2:.2f}m")
        print(f"Distance between points: {distance:.2f}m")

    # Save results if output path is provided
    if output_path:
        save_path = Path(output_path, Path(input_path).stem)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Save results to JSON
        results = {
            'image_path': input_path,
            'point1': {'x': x1, 'y': y1, 'depth': float(depth1), '3d': point1_3d.tolist()},
            'point2': {'x': x2, 'y': y2, 'depth': float(depth2), '3d': point2_3d.tolist()},
            'distance': float(distance)
        }
        with open(save_path / 'measurement.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save visualization
        vis_image = image.copy()
        cv2.circle(vis_image, (x1, y1), radius=5, color=(255, 0, 0), thickness=2)
        cv2.circle(vis_image, (x2, y2), radius=5, color=(255, 0, 0), thickness=2)
        cv2.line(vis_image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        cv2.putText(vis_image, f"Distance: {distance:.2f}m", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite(str(save_path / 'measurement_vis.jpg'), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # Only print save path if not in raw output mode
        if not raw_output:
            print(f"Results saved to: {save_path}")


if __name__ == '__main__':
    main()
