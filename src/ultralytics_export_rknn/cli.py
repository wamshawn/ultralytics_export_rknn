import sys
from typing import List, Any
import click

from .version import get_version
from .onnx import OnnxExport
from .rknn import RKNNExport


@click.group()
@click.pass_context
def cli(
    ctx,
):
    return


@cli.command(help="version")
def version():
    print(get_version())
    return


@cli.command(help="info")
def info():

    # 系统信息
    import platform

    system = platform.system()
    release = platform.release()
    # Ultralytics 版本
    import ultralytics

    ultralytics_version = ultralytics.__version__

    # Python 版本
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    # PyTorch 版本
    import torch

    torch_version = torch.__version__

    cuda = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda else None

    if cuda:
        device_count = torch.cuda.device_count()
        gpu_details = []
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory
            gpu_details.append(f"CUDA:{i} ({name}, {memory // (1024**2)}MiB)")
        gpu_str = " ".join(gpu_details)
    else:
        gpu_str = "CPU"

    # ONNX 版本
    import onnx

    onnx_version = onnx.__version__
    import onnxruntime

    onnxruntime_version = onnxruntime.__version__
    import onnxslim  # type: ignore

    onnxslim_version = onnxslim.__version__

    info_lines = [
        f"Version: {get_version()}",
        f"System: {system}-{release}",
        f"Ultralytics: {ultralytics_version}",
        f"Python: {python_version}",
        f"Torch: {torch_version}",
        f"Onnx: {onnx_version}",
        f"OnnxRuntime: {onnxruntime_version}",
        f"OnnxSlim: {onnxslim_version}",
    ]
    if gpu_str == "CPU":
        info_lines.append("Device: CPU")
    else:
        info_lines.append(f"Device: CUDA {cuda_version}")

    info = "\n".join(info_lines)

    print(info)
    return


def parse_int(s, base=10):
    """安全地将字符串转换为整数"""
    try:
        return int(s, base)
    except ValueError as e:
        return None


def parse_device_options(devices: List[str]) -> List[int] | str | None:
    result: Any = None
    if len(devices) > 0:
        if devices[0] == "cpu":
            result = "cpu"
        elif devices[0] == "gpu":
            result = [0]
        elif devices[0] == "cuda":
            result = "cuda"
        elif devices[0] == "mps":
            result = "mps"
        else:
            result: List[int] = []
            for device in devices:
                gpu = parse_int(device)
                if gpu is not None:
                    result.append(gpu)

            if len(result) == 0:
                result = None
            elif len(result) == 1:
                result = result[0]

    return result


@cli.command()
@click.pass_context
@click.option("--dst", required=True, type=click.Path(exists=True, dir_okay=True))
@click.option("--src", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--name", required=True, type=str)
@click.option("--runs_dir", type=click.Path(dir_okay=True, file_okay=False))
@click.option(
    "--datasets_dir", type=click.Path(exists=False, dir_okay=True, file_okay=False)
)
@click.option(
    "--weights_dir", type=click.Path(exists=False, dir_okay=True, file_okay=False)
)
@click.option("--detect", "task", flag_value="detect", default=True)
@click.option("--segment", "task", flag_value="segment")
@click.option("--classify", "task", flag_value="classify")
@click.option("--pose", "task", flag_value="pose")
@click.option("--obb", "task", flag_value="obb")
@click.option("--imgsz", type=int, default=640)
@click.option("--half", is_flag=True, default=False)
@click.option("--dynamic", is_flag=True, default=False)
@click.option("--simplify/--no-simplify", is_flag=True, default=True)
@click.option("--opset", type=int, default=None)
@click.option("--nms", is_flag=True, default=False)
@click.option("--batch", type=int, default=1)
@click.option("--device", type=str, default=None, multiple=True)
@click.option("--platform", required=True, default=None)
@click.option("--quantized_dtype", default="w8a8")
@click.option("--quantized_algorithm", default="normal")
@click.option("--quantized_method", default="channel")
@click.option("--quant_img_rgb2bgr", is_flag=True, default=False)
@click.option(
    "--mean_values",
    multiple=True,
    type=click.FloatRange(0, 255),
)
@click.option(
    "--std_values",
    multiple=True,
    type=click.FloatRange(0, 255),
)
@click.option("--do_quantization", is_flag=True, default=False)
@click.option("--quantized_dataset", type=click.Path(exists=True, dir_okay=False))
def export(
    ctx,
    dst: str,
    src: str,
    name: str,
    runs_dir: str,
    datasets_dir: str,
    weights_dir: str,
    task: str,
    imgsz: int,
    half: bool,
    dynamic: bool,
    simplify: bool,
    opset: int | None,
    nms: bool,
    batch: int,
    device: str | None,
    platform: str | None,
    quantized_dtype: str,
    quantized_algorithm: str,
    quantized_method: str,
    quant_img_rgb2bgr: bool,
    mean_values: List[float] | None,
    std_values: List[float] | None,
    do_quantization: bool,
    quantized_dataset: str,
):
    onnx_export: OnnxExport | None = None
    rknn_export: RKNNExport | None = None
    try:

        onnx_export = OnnxExport(
            runs_dir=runs_dir,
            weights_dir=weights_dir,
            datasets_dir=datasets_dir,
            model=src,
            task=task,
        )

        onnx_out = onnx_export.handle(
            dst_dir=dst,
            filename=name,
            imgsz=imgsz,
            half=half,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            nms=nms,
            batch=batch,
            device=parse_device_options(device),
        )

        rknn_export = RKNNExport(
            target_platform=platform,  # 根据实际芯片选择
            quantized_dtype=quantized_dtype,  # 指定int8量化
            quantized_algorithm=quantized_algorithm,  # 量化算法
            quantized_method=quantized_method,  # 量化方法
            quant_img_RGB2BGR=quant_img_rgb2bgr,  # opencv 就不需要再反转通道
            mean_values=mean_values,  # yolo固定 0, 0, 0
            std_values=std_values,  # yolo固定 255, 255, 255
            verbose=False,
        )

        rknn_export.handle(
            dst=dst,
            src=onnx_out,
            filename=name,
            do_quantization=do_quantization,
            quantized_dataset=quantized_dataset,
        )

    except Exception as e:
        sys.stderr.write(f"{e} \n")
        sys.stderr.flush()
    finally:
        if rknn_export is not None:
            rknn_export.release()

    return


def main():
    cli()
