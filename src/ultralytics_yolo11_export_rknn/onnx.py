from pathlib import Path
from typing import List
from ultralytics import YOLO, settings
import time


class OnnxExport:

    model: YOLO = None

    def __init__(
        self, runs_dir: str, weights_dir: str, datasets_dir: str, model: str, task: str
    ):
        if model == "":
            raise Exception("model", "model missing")
        if task == "":
            raise Exception("task", "task missing")

        updated = {}
        if datasets_dir is not None and datasets_dir != "":
            updated["datasets_dir"] = f"{Path(datasets_dir).resolve()}"

        if runs_dir is not None and runs_dir != "":
            updated["runs_dir"] = f"{Path(runs_dir).resolve()}"

        if weights_dir is not None and weights_dir != "":
            updated["weights_dir"] = f"{Path(weights_dir).resolve()}"

        if len(updated) > 0:
            updated["sync"] = False
            settings.update(updated)

        model = Path(model).resolve()

        try:
            self.model = YOLO(model, task=task)
        except Exception as e:
            raise e
        return

    def handle(
        self,
        dst_dir: str,  # dir
        filename: str,
        imgsz: int = 640,
        half: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        opset: int | None = None,
        nms: bool = False,
        batch: int = 1,
        device: str | List[int] | None = None,
        load_external_data: bool = False,
    ) -> str:
        dst_path: str | None
        try:
            out = self.model.export(
                format="onnx_rknn",
                imgsz=imgsz,
                half=half,
                dynamic=dynamic,
                simplify=simplify,
                opset=opset,
                nms=nms,
                batch=batch,
                device=device,
                load_external_data=load_external_data,
            )
            time.sleep(0.2)
            out_path = Path(out)
            ext: str = out_path.suffix
            dst_dir_path = Path(dst_dir).absolute()
            dst_path = out_path.rename(dst_dir_path / f"{filename+ext}")

            dst_path = f"{dst_path}"

            print("")
            print(f"ONNX export success, save as '{dst_path}'")
        except Exception as e:
            raise e

        return dst_path
