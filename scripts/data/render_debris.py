"""Blender script: render one 3D debris model as a transparent RGBA PNG, lit
and shadowed, viewed from a randomized near-nadir drone-like angle.

Run through Blender itself, not plain python -- bpy only exists inside
Blender's embedded interpreter:

    blender --background --python scripts/data/render_debris.py -- \\
        --model datasets/debris-3d-models/tire/<uid>/scene.gltf \\
        --category tire --out results/synth/tmp/render.png --seed 0

Real-world size, not the model's arbitrary export scale, is what makes the
composite look plausible next to real cars and lane markings, so every
category is normalized to an approximate real dimension (its longest bounding-
box axis) after import -- the model's own scale is never trusted.

Shadow comes from Blender's shadow-catcher material on a ground plane, with
film_transparent on: the plane itself never appears in the output, only the
soft shadow it receives, composited automatically into the alpha channel
alongside the object. This is the standard "product shot on transparent
background" technique, not a hand-rolled approximation.
"""

import argparse
import math
import random
import sys
from pathlib import Path

import bpy

# Longest bounding-box dimension, metres. Rough real-world sizes, not
# precision figures -- v1 synthesis only needs "plausible", see project docs.
# truck-tire/refrigerator added after a user QA pass found the calibrated
# (real-metre) sizes made small categories hard to see in wide/high-altitude
# frames -- these two are genuinely large real highway spillage rather than
# an inflated version of an existing small category.
REAL_SIZE = {
    "tire": 0.65,
    "cardboard-box": 0.55,
    "suitcase": 0.70,
    "traffic-cone": 0.70,
    "barrel": 0.90,
    "wooden-pallet": 1.20,
    "mattress": 1.90,
    "trash-can": 0.80,
    "wooden-plank": 2.00,
    "ladder": 2.00,
    "truck-tire": 1.10,
    "refrigerator": 1.75,
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--category", required=True, choices=list(REAL_SIZE))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=768, help="output pixels, square")
    return ap.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def import_and_normalize(path: str, target_size: float):
    """Import, then bake every transform straight into the mesh vertices
    (rather than the object's location/rotation/scale) so there is never a
    question of what an object's pivot point is -- an off-centre pivot from
    the original file is exactly what first sent a rotated object flying out
    of frame entirely. After this, `obj`'s own transform is the identity and
    its mesh data IS world space, so later per-render randomisation (yaw,
    ground placement) rotates around the object's true centre for free."""
    from mathutils import Matrix, Vector

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    imported = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in imported if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh objects imported from {path}")

    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.context.view_layer.update()

    # Bake the import transform (location/rotation/scale from the glTF file)
    # into the vertices, then reset the object to identity.
    obj.data.transform(obj.matrix_world.copy())
    obj.matrix_world = Matrix.Identity(4)

    def bbox():
        xs = [v.co.x for v in obj.data.vertices]
        ys = [v.co.y for v in obj.data.vertices]
        zs = [v.co.z for v in obj.data.vertices]
        return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))

    # Real highway debris lies flat, whatever pose the model was exported in
    # (product shots -- ladders, suitcases -- are often modelled standing
    # up). If the tallest axis is Z, the object is "standing"; a near-nadir
    # camera would see almost nothing of it, so lay it on its side first.
    (x0, x1), (y0, y1), (z0, z1) = bbox()
    dims0 = (x1 - x0, y1 - y0, z1 - z0)
    if dims0[2] == max(dims0):
        obj.data.transform(Matrix.Rotation(math.pi / 2, 4, "X"))

    # Uniform scale to the target real size, then drop it so its lowest
    # point sits on z=0 and its centre sits over the world origin.
    (x0, x1), (y0, y1), (z0, z1) = bbox()
    longest = max(x1 - x0, y1 - y0, z1 - z0) or 1.0
    scale = target_size / longest
    obj.data.transform(Matrix.Scale(scale, 4))

    (x0, x1), (y0, y1), (z0, z1) = bbox()
    obj.data.transform(Matrix.Translation((-(x0 + x1) / 2, -(y0 + y1) / 2, -z0)))
    bpy.context.view_layer.update()
    return obj


def build_scene(target_size: float):
    bpy.ops.mesh.primitive_plane_add(size=target_size * 6)
    ground = bpy.context.active_object
    ground.is_shadow_catcher = True

    sun_angle = random.uniform(20, 60)  # degrees above horizon
    sun_azimuth = random.uniform(0, 360)
    bpy.ops.object.light_add(type="SUN")
    sun = bpy.context.active_object
    sun.data.energy = random.uniform(2.5, 5.5)
    sun.data.angle = math.radians(2.0)  # soft-ish shadow, not razor sharp
    sun.rotation_euler = (math.radians(90 - sun_angle), 0, math.radians(sun_azimuth))

    bpy.ops.object.light_add(type="SUN")  # weak fill so the shadow side isn't pure black
    fill = bpy.context.active_object
    fill.data.energy = 0.8
    fill.rotation_euler = (math.radians(90 - sun_angle), 0, math.radians(sun_azimuth + 180))
    return ground


def build_camera(target_size: float, res: int) -> float:
    """Returns the render's pixels per metre on the ground at the object, so
    the compositor can scale it exactly (object and shadow together) instead
    of guessing from the alpha extent, which the soft shadow inflates."""
    tilt = random.uniform(0, 25)       # 0 = straight nadir, matches most drone footage
    yaw = random.uniform(0, 360)
    dist = target_size * random.uniform(3.5, 5.0)
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam_data.lens = 35
    x = dist * math.sin(math.radians(tilt)) * math.cos(math.radians(yaw))
    y = dist * math.sin(math.radians(tilt)) * math.sin(math.radians(yaw))
    z = dist * math.cos(math.radians(tilt))
    cam.location = (x, y, z)
    direction = -cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    fov = 2 * math.atan(cam_data.sensor_width / (2 * cam_data.lens))
    return res / (2 * dist * math.tan(fov / 2))


def main():
    args = parse_args()
    random.seed(args.seed)
    clear_scene()
    target_size = REAL_SIZE[args.category]
    obj = import_and_normalize(args.model, target_size)
    obj.rotation_euler = (0, 0, random.uniform(0, 2 * math.pi))  # random yaw on the ground

    ground = build_scene(target_size)
    px_per_metre = build_camera(target_size, args.res)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.render.resolution_x = args.res
    scene.render.resolution_y = args.res
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = args.out

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)

    # Second pass without the shadow catcher: its alpha is the object alone,
    # which is what the label (mask and box) must cover. The first pass's
    # alpha also holds the shadow, and a dark shadow is as opaque as the
    # object, so no alpha threshold separates them.
    ground.hide_render = True
    scene.cycles.samples = 16
    scene.render.filepath = str(Path(args.out).with_name(Path(args.out).stem + "_obj.png"))
    bpy.ops.render.render(write_still=True)
    Path(args.out).with_suffix(".json").write_text(
        '{"px_per_metre": %.6f}' % px_per_metre, encoding="utf-8")


if __name__ == "__main__":
    main()
