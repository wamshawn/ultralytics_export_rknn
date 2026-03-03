#! /bin/sh

uv run ultralytics-yolox-export-rknn-cli export --runs_dir ./runs --datasets_dir ./datasets --weights_dir ./weights \
   --dst ./dist --src ./assets/yolov8n.pt --name foo \
   --detect --imgsz 640 --simplify --opset 19 --device gpu \
   --platform rk3588 --quantized_dtype w8a8 --do_quantization --quant_img_rgb2bgr --quantized_dataset ./assets/coco8.txt


# uv run ultralytics-yolox-export-rknn-cli export \
#     --dst ./dist --src ./assets/yolov8n.onnx --name bar \
#     --platform rk3588 --quantized_dtype w8a8 --do_quantization --quant_img_rgb2bgr --quantized_dataset ./assets/coco8.txt