# ultralytics export rknn
ultralytics export rknn.

## Compile

### Add Wheels
Create `wheels`.
```shell
mkdir wheels
```

Add [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/packages/x86_64/).
```shell
# download rknn_toolkit2-2.3.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl into ./wheels
uv add ./wheels/rknn_toolkit2-2.3.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

UV sync
```shell
uv sync
```

Show info
```shell
uv run ultralytics-export-rknn-cli info
```

### Build

```shell
rm -fr ./dist/*

uv build
```

### Install

```shell

pipx install --force ./dist/ultralytics_export_rknn-0.1.0-py3-none-any.whl

rm -fr ./dist/*
```

### Usage
Export from `pt`
```shell
ultralytics-export-rknn-cli export --runs_dir ./runs --datasets_dir ./datasets --weights_dir ./weights \
    --dst ./dist --src ./assets/yolov8n.pt --name yolov8n \ 
    --detect --imgsz 640 --simplify --opset 19 --device gpu \ 
    --platform rk3588 --quantized_dtype w8a8 --quant_img_rgb2bgr --do_quantization --quantized_dataset ./assets/coco8.txt
```

Export from `onnx`
```shell
ultralytics-export-rknn-cli export \
    --dst ./dist --src ./assets/yolov8n.onnx --name yolov8n \ 
    --platform rk3588 --quantized_dtype w8a8 --quant_img_rgb2bgr --do_quantization --quantized_dataset ./assets/coco8.txt
```