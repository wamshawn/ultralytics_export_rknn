from rknn.api import RKNN
from typing import List
from pathlib import Path

class RKNNExport:

    model: RKNN = None

    def __init__(
        self,
        target_platform: str | None,
        quantized_dtype: str,
        quantized_algorithm: str,
        quantized_method: str,
        quant_img_RGB2BGR: bool,
        mean_values: List[float] | None,
        std_values: List[float] | None,
        optimization_level: int = 3,
        verbose: bool = False,
    ):
        try:
            self.model = RKNN(verbose=verbose)
            means: List[List[float]] | None = None
            if mean_values is not None and len(mean_values) == 3:
                means = []
                means0 = []
                for v in mean_values:
                    means0.append(v)
                means.append(means0)
            stds: List[List[float]] | None = None
            if std_values is not None and len(std_values) == 3:
                stds = []
                stds0 = []
                for v in std_values:
                    stds0.append(v)
                stds.append(stds0)

            self.model.config(
                mean_values=means,  # yolo固定 0, 0, 0
                std_values=stds,  # yolo固定 255, 255, 255
                quant_img_RGB2BGR=quant_img_RGB2BGR,  # opencv 就不需要再反转通道
                target_platform=target_platform,  # 根据实际芯片选择
                quantized_dtype=quantized_dtype,  # 指定int8量化
                quantized_algorithm=quantized_algorithm,  # 量化算法
                quantized_method=quantized_method,  # 量化方法
                optimization_level=optimization_level,
            )
        except Exception as e:
            if self.model is not None:
                self.release()
            raise e
        return

    def handle(
        self,
        dst: str,
        src: str,
        filename: str,
        do_quantization: bool,
        quantized_dataset: str,
        rknn_batch_size: int | None,
    ) -> str:
        dst_out: str | None
        try:
            ret = self.model.load_onnx(model=src)
            if ret != 0:
                raise Exception("RKNN load model failed!")
            print("RKNN load model succeed.")

            dataset: str | None = None
            if do_quantization:
                if quantized_dataset is None or quantized_dataset == "":
                    raise Exception("quantized_dataset missing!")

                do_quantization = True
                dataset = quantized_dataset

            ret = self.model.build(do_quantization=do_quantization, dataset=dataset, rknn_batch_size=rknn_batch_size)  # 使用 验证 集
            if ret != 0:
                raise Exception("RKNN build model failed!")

            print("RKNN build succeed.")

            dst = Path(dst).resolve()
            dst_out = f"{dst}/{filename}.rknn"
            ret = self.model.export_rknn(dst_out)
            if ret != 0:
                raise Exception("RKNN export model failed!")

            print(f"RKNN export succeed, save as {dst_out}")
        except Exception as e:
            raise e
        return dst_out

    def release(self):
        if self.model is not None:
            self.model.release()