# Datasets

Dataset files are intentionally not stored in this repository. Keep datasets on
the training server disk, shared storage, or a separate dataset volume, and
symlink them into the paths below.

## Expected Layout

```text
coding/datasets/
  BlendedMVS_lowres/
    BlendedMVS/
      <scene>/
        blended_images/
        cams/
        rendered_depth_maps/
  dtu64/
    Cameras/
    scan*/
      image/
```

## BlendedMVS Low-Res

Used for training and full evaluation. The loader expects:

- RGB images under `blended_images/*.jpg`
- camera files under `cams/*_cam.txt`
- depth maps under `rendered_depth_maps/*.pfm`

The setup script can download the low-resolution BlendedMVS release:

```bash
cd coding
bash setup.sh
```

If the dataset already exists on the server, skip the download and symlink it:

```bash
cd coding
mkdir -p datasets/BlendedMVS_lowres
ln -s /path/to/BlendedMVS datasets/BlendedMVS_lowres/BlendedMVS
SKIP_DATA=1 bash setup.sh
```

Official release page used by the setup script:

```text
https://github.com/YoYo000/BlendedMVS/releases/tag/v1.0.0
```

## DTU64

Used as the second benchmark in the current scripts. The local DTU64 config is
pose-only, so depth and Chamfer metrics are not expected there.

Expected layout:

```text
coding/datasets/dtu64/
  Cameras/
    00000000_cam.txt
    ...
  scan1/
    image/
      000000.png
      ...
```

Recommended practice is to keep the preprocessed DTU64 copy in lab storage and
symlink it:

```bash
cd coding
ln -s /path/to/dtu64 datasets/dtu64
```

DTU dataset page:

```text
https://roboimagedata.compute.dtu.dk/?page_id=36
```
