
### Datasets

Assuming `data_root_path` is `~/datasets`, the datasets should be prepared as follows:

**ImageNet-1k** (ID)
```
~/datasets/imagenet1k
└── imagenet
    ├── train
    │   ├── n01440764
    │   ├── n01443537
    .   .
    .   .
    │   └── n15075141
    └── val
        ├── n01440764
        ├── n01443537
        .
        .
        └── n15075141
```
where each folder in `train` and `val` corresponds to a class:
```
n01440764
├── n01440764_10026.JPEG
├── n01440764_10027.JPEG
.
.
└── n01440764_9981.JPEG
```

**iNaturalist** (OOD)
```
~/datasets/inaturalist
└── iNaturalist
    └── images
        ├── 000309dd0c724a5104df8e716b9008a0.jpg
        ├── 0008e0e9c0ec7ec8f1fb69998ae29887.jpg
        .
        .
        └── fffa5fc70b22d6b4aa7311df49dd08ab.jpg
```

**SUN** (OOD)
```
~/datasets/sun
└── SUN
    └── images
        ├── sun_aaaevyiuguntlerb.jpg
        ├── sun_aaaxbrksvmdyrhpe.jpg
        .
        .
        └── sun_dyzllobcgjpcvijp.jpg

```

**Places** (OOD)
```
~/datasets/places
└── Places
    └── images
        ├── b_badlands_00000038.jpg
        ├── b_badlands_00000043.jpg
        .
        .
        └── w_wheat_field_00004990.jpg
```

**Textures** (OOD)
```
~/datasets/textures
└── dtd
    ├── images
    │   ├── banded
    │   ├── blotchy
    .   .
    .   .
    │   └── zigzagged
    .
    .
```

**OpenImage-O** (OOD)
```
~/datasets/openimage-o
└── images
    ├── 000d5efd9500e718.jpg
    ├── 0010a3d096cd57b2.jpg
    .
    .
    └── fffd0258c243bbea.jpg
```

