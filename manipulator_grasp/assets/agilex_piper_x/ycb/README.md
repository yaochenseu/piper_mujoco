# PiperX YCB objects

The five tabletop objects are derived from the YCB models in
`pybullet-object-models` (source commit 57cc400). Visual meshes keep
their original OBJ UVs; textures were downscaled to at most 1024 px.
Collision meshes were split from `collision_vhacd.obj` into one OBJ
per convex hull because MuJoCo only reads the first OBJ object group.

YCB data is published under CC BY 4.0; see the YCB Object and Model Set
for attribution requirements.

Regenerate with:

```bash
python scripts/prepare_ycb_objects.py
```
