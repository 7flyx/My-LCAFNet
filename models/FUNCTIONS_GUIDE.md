# models Function Guide

This document explains what each major function/class does in the `models/` directory.

## models/yolo.py

### Class `Detect`
- `__init__(nc, anchors, ch)`: build prediction conv heads for each detection scale.
- `forward(x)`: convert per-scale feature maps into bbox/objectness/class outputs; decode during inference.
- `_make_grid(nx, ny)`: create spatial coordinate grid for bbox center decoding.

### Class `Model`
- `__init__(cfg, ch, nc, anchors)`: read YAML config, construct graph via `parse_model`, initialize strides/anchors/weights.
- `forward(x, augment, profile)`: entry point for normal or augmented inference.
- `forward_once(x, profile)`: execute one forward pass through all layers with dependency routing.
- `_initialize_biases(cf)`: initialize Detect-layer biases for stable early training.
- `_print_biases()`: print bias statistics for debugging.
- `fuse()`: fuse Conv+BN for faster deployment inference.
- `nms(mode)`: append/remove explicit NMS module.
- `autoshape()`: wrap model with robust preprocessing/postprocessing.
- `info(verbose, img_size)`: print model summary.

### Function `parse_model(d, ch)`
- Parse YAML dict into executable `nn.Sequential` graph.
- Resolve layer arguments, repeats, channels, save-list, and parameter counts.

## models/yolo_test.py (dual-stream)

### Class `Detect`
- Same core role as `yolo.py`, with extra `logits_` output for class-logit analysis/visualization.

### Class `Model`
- `__init__`: build dual-stream model graph from YAML and initialize detect strides/anchors.
- `forward(x, x2, augment, profile)`: dual-input model entry.
- `forward_once(x, x2, profile)`: dual-stream graph execution (`x` visible, `x2` infrared).
- Other utility methods (`_initialize_biases`, `_print_biases`, `fuse`, `nms`, `autoshape`, `info`) are the same idea as in `yolo.py`.

### Function `parse_model(d, ch)`
- Parse multispectral YAML graph; supports custom fusion/attention modules like `DMAF`, `NiNfusion`, `HAFFormer`, `Add2`.

## models/common.py

### Helper functions
- `autopad(k, p)`: infer SAME-style padding from kernel size.
- `DWConv(c1, c2, k, s, act)`: depthwise-style grouped convolution wrapper.
- `to_3d(x)`: reshape BCHW tensor to B(HW)C.
- `to_4d(x, h, w)`: reshape B(HW)C back to BCHW.

### Basic conv/transform blocks
- `Conv`: Conv+BN+activation standard block.
- `Conv_withoutBN`: Conv+activation block.
- `TransformerLayer`: MHA + feed-forward transformer sublayer.
- `TransformerBlock`: ViT-like spatial tokenization + stacked transformer layers.

### CNN backbones/neck blocks
- `VGGblock`: configurable VGG conv stack + maxpool.
- `ResNetblock`: bottleneck residual block.
- `ResNetlayer`: one stage of ResNet or first-stage stem.
- `Bottleneck`: YOLO bottleneck block.
- `BottleneckCSP`: CSP bottleneck block.
- `C3`: CSP block with 3 conv branches.
- `C3TR`: C3 variant with transformer core.
- `SPP`: spatial pyramid pooling block.
- `SPPF`: faster SPP approximation.
- `Focus`: fold spatial information into channels.
- `Contract`: reduce spatial size, increase channels.
- `Expand`: inverse of Contract.
- `Concat`: concatenate tensor list along configured dimension.

### Fusion blocks
- `Add`: weighted sum of two tensors.
- `Add2`: add current tensor with one selected branch from another tensor list.
- `NiNfusion`: concat + 1x1 projection fusion.
- `DMAF`: dual-modal adaptive fusion via cross-modal difference weighting.

### Inference wrappers/results
- `NMS`: module wrapper around non-max suppression.
- `autoShape`: robust input wrapper supporting file/PIL/numpy/torch inputs.
- `Detections`: structured inference result container.
  - `display(...)`: unified rendering/show/save/crop path.
  - `print()`: print summary and timing.
  - `show()`: display images.
  - `save(...)`: save rendered detections.
  - `crop(...)`: save object crops.
  - `render()`: write rendered images back to arrays.
  - `pandas()`: convert outputs to DataFrames.
  - `tolist()`: split batch result into per-image objects.
  - `__len__()`: number of images in result.

### Classification/pooling/attention
- `Classify`: classification head for feature maps.
- `AdaptivePool2d`: adaptive avg/max pool with no-op fallback.
- `SE_Block`: squeeze-and-excitation channel attention.
- `Channel_Attention`: CBAM channel branch.
- `Spatial_Attention`: CBAM spatial branch.
- `CBAM`: channel + optional spatial attention composition.

### ACFormer-style normalization/attention/fusion
- `BiasFree_LayerNorm`: LayerNorm without bias.
- `WithBias_LayerNorm`: LayerNorm with bias.
- `LayerNorm`: 2D feature-map wrapper for above LayerNorms.
- `Mlp`: feed-forward MLP in transformer blocks.
- `SelfAttention`: conv-projected multi-head self-attention.
- `CrossAttention_S`: one-direction cross-attention between two modalities.
- `CrossAttention_M`: bidirectional cross-attention for RGB/IR.
- `HAFFormer`: hybrid attention fusion block for multimodal features.

## models/experimental.py
- `CrossConv`: cross-kernel conv block.
- `Sum`: weighted/unweighted tensor summation.
- `GhostConv`: GhostNet-style cheap feature generation.
- `GhostBottleneck`: bottleneck built from ghost conv + depthwise path.
- `MixConv2d`: mixed depthwise kernels in one block.
- `Ensemble`: aggregate predictions from multiple models.
- `attempt_load(weights, map_location)`: load one/many checkpoints and return model/ensemble.

## models/gradcam.py
- `find_yolo_layer(model, layer_name)`: resolve a nested target layer by hierarchical name.
- `YOLOV5GradCAM.__init__(...)`: register hooks on chosen layer for gradients/activations.
- `YOLOV5GradCAM.forward(...)`: compute Grad-CAM saliency maps for detections.
- `YOLOV5GradCAM.__call__(...)`: callable shortcut to `forward`.

## models/export.py
- Script-style module for exporting `.pt` weights to:
  - TorchScript
  - ONNX (optional simplify)
  - CoreML
- Main flow: parse args -> load model -> dry run -> export per backend -> validate/save.

## models/__init__.py
- Currently empty placeholder module file.
