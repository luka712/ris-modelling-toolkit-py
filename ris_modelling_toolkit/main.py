# -----------------------
# Example usage
# -----------------------
import numpy as np
import trimesh
import logging

from trimesh.visual import TextureVisuals
from trimesh.visual.material import Material, SimpleMaterial

from ris_modelling_toolkit.src.compute.model import split_geometry_for_tiled_uvs
from ris_modelling_toolkit.src.data.geometry import trimesh_mesh_to_geometry

trimesh.util.attach_to_log(level=logging.DEBUG)
mesh = trimesh.load("content/test.obj")
geometry = trimesh_mesh_to_geometry(mesh)
geometry = split_geometry_for_tiled_uvs(geometry)

faces = []

for i in range(geometry.get_triangle_count()):
    faces.append([i * 3, i * 3 + 1, i * 3 + 2])

print(geometry.vertices)
print(geometry.uvs)

new_mesh = trimesh.Trimesh(vertices=geometry.get_vertices(), faces=faces, process=False)
new_mesh.visual = TextureVisuals(material=SimpleMaterial(image=mesh.visual.material.image))
new_mesh.visual.uv = geometry.get_uv()
new_mesh.export("content/test_split.obj")

import math
import sys

def area(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])

def compute_bary(target, uv1, uv2, uv3):
    a0 = area(uv1, uv2, uv3)
    if abs(a0) < 1e-9:
        return None
    a1 = area(target, uv2, uv3)
    a2 = area(uv1, target, uv3)
    a3 = area(uv1, uv2, target)
    b1 = a1 / a0
    b2 = a2 / a0
    b3 = a3 / a0
    if abs(b1 + b2 + b3 - 1) > 1e-6 or min(b1, b2, b3) < -1e-6:
        return None
    return (b1, b2, b3)

def clip_against_halfplane(input_poly, inside, intersect):
    if not input_poly:
        return []
    output = []
    prev = input_poly[-1]
    prev_inside = inside(prev)
    for curr in input_poly:
        curr_inside = inside(curr)
        if curr_inside:
            if not prev_inside:
                inter = intersect(prev, curr)
                if inter:
                    output.append(inter)
            output.append(curr)
        elif prev_inside:
            inter = intersect(prev, curr)
            if inter:
                output.append(inter)
        prev = curr
        prev_inside = curr_inside
    return output

def clean_poly(poly):
    if len(poly) < 3:
        return []
    cleaned = []
    prev_p = None
    for p in poly:
        if prev_p is None or abs(p[0] - prev_p[0]) > 1e-6 or abs(p[1] - prev_p[1]) > 1e-6:
            cleaned.append(p)
            prev_p = p
    if len(cleaned) > 2 and abs(cleaned[0][0] - cleaned[-1][0]) < 1e-6 and abs(cleaned[0][1] - cleaned[-1][1]) < 1e-6:
        cleaned.pop()
    if len(cleaned) < 3:
        return []
    return cleaned

# For the given OBJ, paste here or read from file
# To read from file: obj_content = open('input.obj', 'r').read()
obj_content = """# Simple test mesh with UVs outside [0,1]
o TestMesh
mtllib test.mtl
usemtl material

# Vertices
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0

# UV coordinates (some outside 0-1)
vt 0.0 0.0
vt 2.0 0.0
vt 0.0 1.0

# Face (using vertex/uv indices)
f 1/1 2/2 3/3"""

lines = obj_content.splitlines()
v_list = []
vt_list = []
faces = []
obj_name = 'TestMesh'
mtl = 'test.mtl'
material = 'material'
for line in lines:
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    if line.startswith('v '):
        parts = line[2:].split()
        v_list.append((float(parts[0]), float(parts[1]), float(parts[2])))
    elif line.startswith('vt '):
        parts = line[3:].split()
        vt_list.append((float(parts[0]), float(parts[1])))
    elif line.startswith('o '):
        obj_name = line[2:]
    elif line.startswith('mtllib '):
        mtl = line[7:]
    elif line.startswith('usemtl '):
        material = line[7:]
    elif line.startswith('f '):
        fs = line[2:].split()
        vidxs = []
        vtidxs = []
        for fpart in fs:
            vs, ts = fpart.split('/')
            vidxs.append(int(vs))
            vtidxs.append(int(ts))
        faces.append((vidxs, vtidxs))

new_v = []
new_vt = []
new_faces = []

for face in faces:
    vidx, vtidx = face
    v1 = v_list[vidx[0] - 1]
    v2 = v_list[vidx[1] - 1]
    v3 = v_list[vidx[2] - 1]
    uv1 = vt_list[vtidx[0] - 1]
    uv2 = vt_list[vtidx[1] - 1]
    uv3 = vt_list[vtidx[2] - 1]
    us = [uv1[0], uv2[0], uv3[0]]
    vs_ = [uv1[1], uv2[1], uv3[1]]
    min_u = min(us)
    max_u = max(us)
    min_v = min(vs_)
    max_v = max(vs_)
    iu_start = math.floor(min_u)
    iu_end = math.floor(max_u)
    jv_start = math.floor(min_v)
    jv_end = math.floor(max_v)
    poly_base = [uv1, uv2, uv3]
    for iu in range(iu_start, iu_end + 1):
        for jv in range(jv_start, jv_end + 1):
            left = iu
            right = iu + 1.0
            bottom = jv
            top = jv + 1.0
            poly = poly_base[:]
            # Clip u >= left
            def inside_left(p): return p[0] >= left - 1e-9
            def inter_left(s, e):
                du = e[0] - s[0]
                if abs(du) < 1e-9:
                    return None
                t = (left - s[0]) / du
                if 0 - 1e-9 <= t <= 1 + 1e-9:
                    return (left, s[1] + t * (e[1] - s[1]))
                return None
            poly = clip_against_halfplane(poly, inside_left, inter_left)
            # Clip u <= right
            def inside_right(p): return p[0] <= right + 1e-9
            def inter_right(s, e):
                du = e[0] - s[0]
                if abs(du) < 1e-9:
                    return None
                t = (right - s[0]) / du
                if 0 - 1e-9 <= t <= 1 + 1e-9:
                    return (right, s[1] + t * (e[1] - s[1]))
                return None
            poly = clip_against_halfplane(poly, inside_right, inter_right)
            # Clip v >= bottom
            def inside_bottom(p): return p[1] >= bottom - 1e-9
            def inter_bottom(s, e):
                dv = e[1] - s[1]
                if abs(dv) < 1e-9:
                    return None
                t = (bottom - s[1]) / dv
                if 0 - 1e-9 <= t <= 1 + 1e-9:
                    return (s[0] + t * (e[0] - s[0]), bottom)
                return None
            poly = clip_against_halfplane(poly, inside_bottom, inter_bottom)
            # Clip v <= top
            def inside_top(p): return p[1] <= top + 1e-9
            def inter_top(s, e):
                dv = e[1] - s[1]
                if abs(dv) < 1e-9:
                    return None
                t = (top - s[1]) / dv
                if 0 - 1e-9 <= t <= 1 + 1e-9:
                    return (s[0] + t * (e[0] - s[0]), top)
                return None
            poly = clip_against_halfplane(poly, inside_top, inter_top)
            poly = clean_poly(poly)
            if len(poly) < 3:
                continue
            poly_verts_idx = []
            skip = False
            for p in poly:
                b = compute_bary(p, uv1, uv2, uv3)
                if b is None:
                    skip = True
                    break
                b1, b2, b3 = b
                px = b1 * v1[0] + b2 * v2[0] + b3 * v3[0]
                py = b1 * v1[1] + b2 * v2[1] + b3 * v3[1]
                pz = b1 * v1[2] + b2 * v2[2] + b3 * v3[2]
                new_v.append((px, py, pz))
                nu = p[0] - iu
                nv = p[1] - jv
                if nu < 0: nu = 0.0
                if nu > 1.0: nu = 1.0
                if nv < 0: nv = 0.0
                if nv > 1.0: nv = 1.0
                new_vt.append((nu, nv))
                poly_verts_idx.append(len(new_v))
            if skip:
                for _ in range(len(poly_verts_idx)):
                    new_v.pop()
                    new_vt.pop()
                continue
            n = len(poly_verts_idx)
            for k in range(2, n):
                idx1 = poly_verts_idx[0]
                idx2 = poly_verts_idx[k - 1]
                idx3 = poly_verts_idx[k]
                f_str = f"f {idx1}/{idx1} {idx2}/{idx2} {idx3}/{idx3}"
                new_faces.append(f_str)

# Output the new OBJ
print("# Wrapped UV Mesh")
print(f"o {obj_name}_wrapped")
print(f"mtllib {mtl}")
print(f"usemtl {material}")
for vv in new_v:
    print(f"v {vv[0]:.6f} {vv[1]:.6f} {vv[2]:.6f}")
for tv in new_vt:
    print(f"vt {tv[0]:.6f} {tv[1]:.6f}")
for ff in new_faces:
    print(ff)




