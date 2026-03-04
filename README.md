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
```

Add [ultralytics revision](https://github.com/wamshawn/ultralytics_revision/releases/tag/rknn-v8-v0.0.2)
```shell
# download https://github.com/wamshawn/ultralytics_revision/releases/download/rknn-v8-v0.0.2/ultralytics-8.2.82+rknn-py3-none-any.whl into  ./wheels
```

UV sync
```shell
uv sync
```

Show info
```shell
uv run ultralytics-yolov8-export-rknn-cli info
```

### Build

```shell
rm -fr ./dist/*

uv build
```

### Install

```shell

pipx install --pip-args "--find-links ./wheels" --force ./dist/ultralytics_yolov8_export_rknn-0.1.0-py3-none-any.whl

rm -fr ./dist/*
```

### Usage
```shell
ultralytics-yolov8-export-rknn-cli export --runs_dir ./runs --datasets_dir ./datasets --weights_dir ./weights \
    --dst ./dist --src ./assets/yolov8n.pt --name yolov8n \ 
    --detect --imgsz 640 --simplify --opset 19 --device gpu \ 
    --platform rk3588 --quantized_dtype w8a8 --quant_img_rgb2bgr --do_quantization --quantized_dataset ./assets/coco8.txt
```