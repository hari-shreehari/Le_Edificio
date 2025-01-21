import argparse
import json
import math
import os
import shutil
import sys

import bpy
import cv2
import keras_ocr
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
from skimage.morphology import skeletonize
from wand.color import Color
from wand.image import Image as wImage
from ultralytics import YOLO
from stacker import stack_floors

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def convert_to_png(input_path, output_dir):
    output_path = os.path.join(output_dir, "converted_blueprint.png")
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".pdf":
        images = convert_from_path(input_path)
        images[0].save(output_path, "PNG")
    elif ext == ".svg":
        with wImage(filename=input_path, background=Color("white")) as img:
            img.background_color = Color("white")
            img.alpha_channel = 'remove'
            img.format = 'png'
            img.save(filename=output_path)
    elif ext in [".dwg", ".dxf"]: 
        os.system(f"dwg2png \"{input_path}\" \"{output_path}\"")
        if not os.path.exists(output_path):
            raise ValueError("Failed to convert DWG/DXF file. Ensure 'dwg2png' is installed and in PATH.")
    elif ext in [".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".png"]:
        with Image.open(input_path) as img:
            img = img.convert("RGBA")  
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            composite = Image.alpha_composite(background, img)
            composite = composite.convert("RGB") 
            composite.save(output_path, "PNG")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Please provide a valid blueprint file.")
    
    with Image.open(output_path) as img:
        width, height = img.size
        if width > height:
            new_width = 640
            new_height = int(height * (640 / width))
        else:
            new_height = 640
            new_width = int(width * (640 / height))
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_img.save(output_path, "PNG")
        
    return output_path

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

def inpaint_text(img_path, pipeline):
    img = keras_ocr.tools.read(img_path)
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))

        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,
        thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    return(img)

def extract_black_and_non_black_lines(image_path, output_dir, contrast_factor=2.0, black_threshold=50, dilation_size=5):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, black_lines = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)
    
    kernel_thick = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))
    thickened_black_lines = cv2.dilate(black_lines, kernel_thick, iterations=1)
    
    black_lines_img = Image.fromarray(thickened_black_lines)
    black_lines_array = np.array(black_lines_img.convert('RGBA'))
    black_lines_array[(black_lines_array[:, :, 0] == 0) & 
                      (black_lines_array[:, :, 1] == 0) & 
                      (black_lines_array[:, :, 2] == 0)] = [255, 255, 255, 0]
    
    transparent_img = Image.fromarray(black_lines_array, 'RGBA')
    transparent_overlay = np.array(transparent_img)
    
    alpha_channel = transparent_overlay[:, :, 3]
    foreground = cv2.cvtColor(transparent_overlay[:, :, :3], cv2.COLOR_BGR2GRAY)
    mask = alpha_channel > 0

    result = gray.copy()
    result[mask] = foreground[mask]
    
    non_black_lines = result
    non_black_lines_pil = Image.fromarray(non_black_lines)
    enhancer = ImageEnhance.Contrast(non_black_lines_pil)
    enhanced_image = enhancer.enhance(contrast_factor)
    
    black_lines_path = os.path.join(output_dir, "black_lines.png")
    non_black_lines_path = os.path.join(output_dir, "non_black_lines.png")
    cv2.imwrite(black_lines_path, black_lines)
    cv2.imwrite(non_black_lines_path, np.array(enhanced_image))

    return cv2.bitwise_not(black_lines), np.array(enhanced_image)


def process_image_for_dark_lines(image_path, scale_factor, output_dir):
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    black_lines, non_black_lines = extract_black_and_non_black_lines(os.path.join(output_dir, "flipped.png"), output_dir)
    flipped=cv2.imread(os.path.join(output_dir, "flipped.png"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY))
    _, thin_lines = cv2.threshold(enhanced, 120, 255, cv2.THRESH_BINARY_INV)
    _, thick_lines = cv2.threshold(enhanced, 60, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thin_lines_closed = cv2.morphologyEx(thin_lines, cv2.MORPH_CLOSE, kernel, iterations=2)
    thin_lines_dilated = cv2.dilate(thin_lines_closed, kernel, iterations=1)
    thin_lines_skel = skeletonize(thin_lines_dilated // 255)
    thin_lines_skel = (thin_lines_skel * 255).astype(np.uint8)

    output_combined = np.zeros_like(original)
    output_combined[thick_lines == 255] = [0, 0, 255]
    output_combined[thin_lines_skel == 255] = [255, 0, 0]

    contours, _ = cv2.findContours(thick_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_data = []
    for i, contour in enumerate(contours):
        scaled_coordinates = [(point[0][0] * scale_factor, point[0][1] * scale_factor) for point in contour]
        contour_info = {
            "id": i,
            "coordinates": scaled_coordinates,
            "area": cv2.contourArea(contour) * (scale_factor ** 2)
        }
        contour_data.append(contour_info)

    json_path = os.path.join(output_dir, "contours.json")
    with open(json_path, "w") as json_file:
        json.dump(contour_data, json_file, indent=4)

    contours_image_path = os.path.join(output_dir, "detected_lines.png")
    cv2.imwrite(contours_image_path, output_combined)

    return json_path

def process_image(image_path, scale_factor, output_dir):
    original = cv2.imread(image_path)
    flipped=cv2.imread(os.path.join(output_dir, "flipped.png"))
    gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    adaptive_thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    edges = cv2.Canny(adaptive_thresh, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 50
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    approx_contours = [cv2.approxPolyDP(c, 0.005 * cv2.arcLength(c, True), True) for c in filtered_contours]

    contour_data = []
    for i, contour in enumerate(approx_contours):
        scaled_coordinates = [(point[0][0] * scale_factor, point[0][1] * scale_factor) for point in contour]
        contour_info = {
            "id": i,
            "coordinates": scaled_coordinates,
            "area": cv2.contourArea(contour) * (scale_factor ** 2)
        }
        contour_data.append(contour_info)

    json_path = os.path.join(output_dir, "contours.json")
    with open(json_path, "w") as json_file:
        json.dump(contour_data, json_file, indent=4)

    output = np.zeros_like(original)
    cv2.drawContours(output, approx_contours, -1, (0, 255, 0), 2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return json_path

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def calculate_centroid(contours):
    all_points = [point for contour in contours for point in contour["coordinates"]]
    x_coords = [x for x, y in all_points]
    y_coords = [y for x, y in all_points]
    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)
    return centroid_x, centroid_y

def shift_coordinates(contours, centroid):
    shifted_contours = []
    for contour in contours:
        shifted_coords = [(x - centroid[0], y - centroid[1]) for x, y in contour["coordinates"]]
        shifted_contours.append({"id": contour["id"], "coordinates": shifted_coords})
    return shifted_contours

def create_walls(contour_coords, height, material, name="Wall"):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    base_vertices = [(x, y, 0) for x, y in contour_coords]
    top_vertices = [(x, y, height) for x, y in contour_coords]
    
    vertices = base_vertices + top_vertices
    faces = []
    
    num_points = len(base_vertices)
    for i in range(num_points):
        next_i = (i + 1) % num_points
        faces.append((i, next_i, num_points + next_i, num_points + i))
    
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj.data.materials.append(material)

    return obj

def create_floor(contour_coords, material, name="Floor"):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    vertices = [(x, y, 0) for x, y in contour_coords]
    edges = []
    faces = [tuple(range(len(vertices)))]

    mesh.from_pydata(vertices, edges, faces)
    mesh.update()
    obj.data.materials.append(material)

    return obj

def export_as_glb(output_path):
    glb_output_path = os.path.splitext(output_path)[0] + ".glb"
    bpy.ops.export_scene.gltf(filepath=glb_output_path, export_format='GLB')
    return glb_output_path

def export_as_fbx(output_path):
    fbx_output_path = os.path.splitext(output_path)[0] + ".fbx"
    bpy.ops.export_scene.fbx(filepath=fbx_output_path, use_selection=False)
    return fbx_output_path

def create_material_with_texture(name, texture_image_path):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    texture_node = nodes.new("ShaderNodeTexImage")
    texture_node.image = bpy.data.images.load(texture_image_path)
    bsdf_node = nodes.get("Principled BSDF")
    if bsdf_node:
        links.new(texture_node.outputs["Color"], bsdf_node.inputs["Base Color"])
    return material

def create_walls_with_texture(contour_coords, height, material, thickness, name="Wall"):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    base_vertices = [(x, y, 0) for x, y in contour_coords]
    top_vertices = [(x, y, height) for x, y in contour_coords]

    vertices = base_vertices + top_vertices
    faces = []

    num_points = len(base_vertices)
    for i in range(num_points):
        next_i = (i + 1) % num_points
        faces.append((i, next_i, num_points + next_i, num_points + i))

    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj.data.materials.append(material)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")
    
    
    solidify_mod = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
    solidify_mod.thickness = thickness
    solidify_mod.offset = 0 

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=solidify_mod.name)

    return obj

def create_floor_with_texture(contour_coords, material, name="Floor"):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    vertices = [(x, y, 0) for x, y in contour_coords]
    faces = [tuple(range(len(vertices)))]

    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    obj.data.materials.append(material)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.uv.smart_project()  
    bpy.ops.object.mode_set(mode="OBJECT")

    return obj

def calculate_object_area(obj):
    if obj.type == 'MESH':
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='OBJECT')
        obj.data.update(calc_edges=True, calc_edges_loose=True)
        return sum(polygon.area for polygon in obj.data.polygons)
    return 0

def remove_smaller_floors():
    floor_objects = [obj for obj in bpy.data.objects if obj.name.startswith("Floor") and obj.type == 'MESH']
    
    if not floor_objects:
        print("No floor objects found.")
        return

    largest_floor = max(floor_objects, key=calculate_object_area)
    largest_area = calculate_object_area(largest_floor)
    print(f"Largest floor: {largest_floor.name} with area {largest_area:.2f}")

    for obj in floor_objects:
        if obj != largest_floor:
            print(f"Removing {obj.name}")
            bpy.data.objects.remove(obj)

def object_detection(image_path, model_path, output_dir):
    model = YOLO(model_path)
    results = model(image_path)
    results[0].save(os.path.join(output_dir, "detection_results.jpg"))
    class_mapping = {0: "column", 1: "door", 2:"double-door", 3:"sliding-door", 4:"stairs", 5:"window"}

    detection_data = []

    if not results or not results[0].boxes:
        print("No objects detected.")
    else:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])

            if class_id in class_mapping.keys():
                label = class_mapping[class_id]

                detection_data.append({
                    "class": label,
                    "bounding_box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "confidence": round(confidence.item(), 2)
                })

    output_json_path = os.path.join(output_dir, "detections.json")
    with open(output_json_path, 'w') as json_file:
        json.dump(detection_data, json_file, indent=4)

    return output_json_path

def create_door(location, height=5, width=2, thickness=0.35, rotation=0, texture_path="./assets/furniture/wood.jpg"):
    location = (location[0], location[1], height / 2)
    bpy.ops.mesh.primitive_cube_add(location=location, scale=(width, thickness, height/2))
    door = bpy.context.object
    door.name = "Door"
    door.rotation_euler[2] = rotation

    door_material = bpy.data.materials.new(name="Door_Material")
    door_material.use_nodes = True
    nodes = door_material.node_tree.nodes
    links = door_material.node_tree.links

    for node in nodes:
        nodes.remove(node)

    texture_node = nodes.new(type="ShaderNodeTexImage")
    texture_node.location = (-300, 0)
    if os.path.exists(texture_path):
        texture_node.image = bpy.data.images.load(texture_path)
    else:
        print(f"Texture file not found at {texture_path}")

    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled_node.location = (0, 0)

    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (300, 0)

    links.new(texture_node.outputs["Color"], principled_node.inputs["Base Color"])
    links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])

    door.data.materials.append(door_material)

    door.visible_shadow = False
    door.visible_diffuse = False
    door.visible_glossy = False

    handle_location = (location[0], location[1]+thickness/2, location[2])
    handle_twist=1.5708
    if rotation==1.5708:
        handle_twist=0

    bpy.ops.mesh.primitive_cylinder_add(location=handle_location, radius=0.2, depth=thickness*4, rotation=(0, 1.5708, handle_twist))
    handle = bpy.context.object
    handle.name = "Door_Handle"

    handle_material = bpy.data.materials.new(name="Handle_Material")
    handle_material.use_nodes = True
    handle_nodes = handle_material.node_tree.nodes
    handle_links = handle_material.node_tree.links

    for node in handle_nodes:
        handle_nodes.remove(node)

    handle_principled_node = handle_nodes.new(type="ShaderNodeBsdfPrincipled")
    handle_principled_node.location = (0, 0)
    handle_principled_node.inputs["Base Color"].default_value = (0.1, 0.1, 0.1, 1)
    handle_principled_node.inputs["Metallic"].default_value = 0.8
    handle_principled_node.inputs["Roughness"].default_value = 0.3

    handle_output_node = handle_nodes.new(type="ShaderNodeOutputMaterial")
    handle_output_node.location = (200, 0)

    handle_links.new(handle_principled_node.outputs["BSDF"], handle_output_node.inputs["Surface"])

    handle.data.materials.append(handle_material)

    handle.visible_shadow = False
    handle.visible_diffuse = False
    handle.visible_glossy = False

    return door, handle

def create_double_door(location, height=5, width=4, thickness=0.35, rotation=0, texture_path="./assets/furniture/wood.jpg"):
    location = (location[0], location[1], height / 2)
    bpy.ops.mesh.primitive_cube_add(location=location, scale=(width, thickness, height/2))
    door = bpy.context.object
    door.name = "Door"
    door.rotation_euler[2] = rotation

    door_material = bpy.data.materials.new(name="Door_Material")
    door_material.use_nodes = True
    nodes = door_material.node_tree.nodes
    links = door_material.node_tree.links

    for node in nodes:
        nodes.remove(node)

    texture_node = nodes.new(type="ShaderNodeTexImage")
    texture_node.location = (-300, 0)
    if os.path.exists(texture_path):
        texture_node.image = bpy.data.images.load(texture_path)
    else:
        print(f"Texture file not found at {texture_path}")

    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled_node.location = (0, 0)

    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (300, 0)

    links.new(texture_node.outputs["Color"], principled_node.inputs["Base Color"])
    links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])

    door.data.materials.append(door_material)

    door.visible_shadow = False
    door.visible_diffuse = False
    door.visible_glossy = False
    
    handle_twist=1.5708
    if rotation==1.5708:
        handle_twist=0

    offset=width/4
    if rotation==0:
        handle_location = (location[0]-offset, location[1]+thickness/2, location[2])
    else:
        handle_location = (location[0], location[1]-offset, location[2])

    bpy.ops.mesh.primitive_cylinder_add(location=handle_location, radius=0.2, depth=thickness*4, rotation=(0, 1.5708, handle_twist))
    handle = bpy.context.object
    handle.name = "Left_Door_Handle"

    handle_material = bpy.data.materials.new(name="Handle_Material")
    handle_material.use_nodes = True
    handle_nodes = handle_material.node_tree.nodes
    handle_links = handle_material.node_tree.links

    for node in handle_nodes:
        handle_nodes.remove(node)

    handle_principled_node = handle_nodes.new(type="ShaderNodeBsdfPrincipled")
    handle_principled_node.location = (0, 0)
    handle_principled_node.inputs["Base Color"].default_value = (0.1, 0.1, 0.1, 1)
    handle_principled_node.inputs["Metallic"].default_value = 0.8
    handle_principled_node.inputs["Roughness"].default_value = 0.3

    handle_output_node = handle_nodes.new(type="ShaderNodeOutputMaterial")
    handle_output_node.location = (200, 0)

    handle_links.new(handle_principled_node.outputs["BSDF"], handle_output_node.inputs["Surface"])

    handle.data.materials.append(handle_material)

    handle.visible_shadow = False
    handle.visible_diffuse = False
    handle.visible_glossy = False

    if rotation==0:
        handle_location = (location[0]+offset, location[1]+thickness/2, location[2])
    else:
        handle_location = (location[0], location[1]+offset, location[2])

    bpy.ops.mesh.primitive_cylinder_add(location=handle_location, radius=0.2, depth=thickness*4, rotation=(0, 1.5708, handle_twist))
    handle = bpy.context.object
    handle.name = "Right_Door_Handle"

    handle_material = bpy.data.materials.new(name="Handle_Material")
    handle_material.use_nodes = True
    handle_nodes = handle_material.node_tree.nodes
    handle_links = handle_material.node_tree.links

    for node in handle_nodes:
        handle_nodes.remove(node)

    handle_principled_node = handle_nodes.new(type="ShaderNodeBsdfPrincipled")
    handle_principled_node.location = (0, 0)
    handle_principled_node.inputs["Base Color"].default_value = (0.1, 0.1, 0.1, 1)
    handle_principled_node.inputs["Metallic"].default_value = 0.8
    handle_principled_node.inputs["Roughness"].default_value = 0.3

    handle_output_node = handle_nodes.new(type="ShaderNodeOutputMaterial")
    handle_output_node.location = (200, 0)

    handle_links.new(handle_principled_node.outputs["BSDF"], handle_output_node.inputs["Surface"])

    handle.data.materials.append(handle_material)

    handle.visible_shadow = False
    handle.visible_diffuse = False
    handle.visible_glossy = False

    return door, handle

def create_stairs(location, height=10, radius=3, step_height=0.2, step_width=1, step_thickness=0.1, texture_path="./assets/furniture/wood.jpg"):
    height=height+height/4
    location=(location[0], location[1], height/2)
    num_steps = math.ceil(height / (step_height*2))
    central_pole_height = height + step_thickness
    
    bpy.ops.mesh.primitive_cylinder_add(
        location=location,
        radius=radius,
        depth=central_pole_height
    )
    central_pole = bpy.context.object
    central_pole.name = "Central_Pole"

    pole_material = bpy.data.materials.new(name="Pole_Material")
    pole_material.use_nodes = True
    nodes = pole_material.node_tree.nodes
    links = pole_material.node_tree.links

    for node in nodes:
        nodes.remove(node)

    pole_texture_node = nodes.new(type="ShaderNodeTexImage")
    pole_texture_node.location = (-300, 0)
    if os.path.exists(texture_path):
        pole_texture_node.image = bpy.data.images.load(texture_path)
    else:
        print(f"Texture file not found at {texture_path}")

    pole_principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    pole_principled_node.location = (0, 0)

    pole_output_node = nodes.new(type="ShaderNodeOutputMaterial")
    pole_output_node.location = (300, 0)

    links.new(pole_texture_node.outputs["Color"], pole_principled_node.inputs["Base Color"])
    links.new(pole_principled_node.outputs["BSDF"], pole_output_node.inputs["Surface"])

    central_pole.data.materials.append(pole_material)

    for i in range(num_steps):
        angle = i*(270/num_steps)
        rad_angle = math.radians(angle)
        
        x = location[0] + (radius + step_width) *  math.cos(rad_angle)
        y = location[1] + (radius + step_width) *  math.sin(rad_angle)
        z = i * 2 * step_height

        bpy.ops.mesh.primitive_cube_add(
            location=(x, y, z),
            scale=(step_width, step_thickness, step_height / 2)
        )
        step = bpy.context.object
        step.name = f"Step_{i+1}"

        step.rotation_euler[2] = rad_angle

        step_material = bpy.data.materials.new(name=f"Step_Material_{i+1}")
        step_material.use_nodes = True
        step_nodes = step_material.node_tree.nodes
        step_links = step_material.node_tree.links

        for node in step_nodes:
            step_nodes.remove(node)

        step_texture_node = step_nodes.new(type="ShaderNodeTexImage")
        step_texture_node.location = (-300, 0)
        if os.path.exists(texture_path):
            step_texture_node.image = bpy.data.images.load(texture_path)
        else:
            print(f"Texture file not found at {texture_path}")

        step_principled_node = step_nodes.new(type="ShaderNodeBsdfPrincipled")
        step_principled_node.location = (0, 0)

        step_output_node = step_nodes.new(type="ShaderNodeOutputMaterial")
        step_output_node.location = (300, 0)

        step_links.new(step_texture_node.outputs["Color"], step_principled_node.inputs["Base Color"])
        step_links.new(step_principled_node.outputs["BSDF"], step_output_node.inputs["Surface"])

        step.data.materials.append(step_material)

    return central_pole, [f"Step_{i+1}" for i in range(num_steps)]


def create_window(location, material, height=4, width=2, thickness=0.2, rotation=0):
    location = (location[0], location[1], height)

    top_block_location = (location[0], location[1], height-height/8)
    bpy.ops.mesh.primitive_cube_add(location=top_block_location, scale=(width, thickness, height/8))
    top_block = bpy.context.object
    top_block.name = "Window_Top_Block"
    top_block.data.materials.append(material)
    
    bottom_block_location = (location[0], location[1], height/8)
    bpy.ops.mesh.primitive_cube_add(location=bottom_block_location, scale=(width, thickness, height/8))
    bottom_block = bpy.context.object
    bottom_block.name = "Window_Bottom_Block"
    bottom_block.data.materials.append(material)
    
    middle_pane_location = (location[0], location[1], height/2)
    bpy.ops.mesh.primitive_cube_add(location=middle_pane_location, scale=(width, thickness, height/4))
    middle_pane = bpy.context.object
    middle_pane.name = "Window_Pane"
    
    glass_material = bpy.data.materials.new(name="Glass_Material")
    glass_material.use_nodes = True
    nodes = glass_material.node_tree.nodes
    links = glass_material.node_tree.links

    for node in nodes:
        nodes.remove(node)

    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled_node.location = (0, 0)
    principled_node.inputs["Base Color"].default_value = (1, 1, 1, 1)
    principled_node.inputs["Roughness"].default_value = 0.1
    principled_node.inputs["Alpha"].default_value = 0.3

    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    output_node.location = (200, 0)

    links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])
    glass_material.blend_method = 'BLEND'

    middle_pane.data.materials.append(glass_material)

    for obj in [top_block, bottom_block, middle_pane]:
        obj.rotation_euler[2] = rotation
    
    return top_block, bottom_block, middle_pane



def main():
    try:
        parser = argparse.ArgumentParser(description="Process blueprint and generate 3D model.")

        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--convert', action='store_true', help="Convert blueprint to 3D model.")
        group.add_argument('--stack', action='store_true', help="Stack multiple models.")

        parser.add_argument('image', type=str, nargs="?", help="Path to blueprint image.")
        parser.add_argument('--scale_factor', type=float, default=0.1, help="Scale of building (default: 0.1).")
        parser.add_argument('--wall_height', type=float, default=10, help="Height of the wall (default: 10).")
        parser.add_argument('--wall_thickness', type=float, default=0.5, help="Thickness of the wall (default: 0.5).")
        parser.add_argument('--wall_texture', type=str, help="wall texture.")
        parser.add_argument('--floor_texture', type=str, help="floor texture.")
        parser.add_argument('--output_directory', type=str, required=True, help="Path to write outputs.")
        args = parser.parse_args()
        
        output_dir = args.output_directory.strip().strip('"')  
        
        if args.convert:

            os.makedirs(output_dir, exist_ok=True)
            print(f"Using output directory: {output_dir}")

            image_path = args.image.strip().strip('"')
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"The file '{image_path}' does not exist.")

            converted_path = convert_to_png(image_path, output_dir)

            flipped_image = cv2.flip(cv2.imread(converted_path), 2)
            flip_image_path = os.path.join(output_dir, "original_flipped_image.jpg")
            cv2.imwrite(flip_image_path, flipped_image)
            
            original = cv2.imread(converted_path)
            textless=inpaint_text(original, pipeline = keras_ocr.pipeline.Pipeline())

            flipped = cv2.flip(textless, 2)
            cv2.imwrite(os.path.join(output_dir, "flipped.png"), flipped)

            floor_texture_path = resource_path("./assets/floor/" + args.floor_texture.strip().strip("'"))

            if not os.path.exists(floor_texture_path):
                raise FileNotFoundError(f"The floor texture file '{floor_texture_path}' does not exist.")
            
            new_floor_texture_path = shutil.copy(floor_texture_path, output_dir)

            floor_texture_path=os.path.abspath(new_floor_texture_path)

            floor_material = create_material_with_texture("FloorTexture", floor_texture_path)

            wall_texture_path = resource_path("./assets/wall/" + args.wall_texture.strip().strip("'"))
            
            if not os.path.exists(wall_texture_path):
                raise FileNotFoundError(f"The wall texture file '{wall_texture_path}' does not exist.")

            new_wall_texture_path = shutil.copy(wall_texture_path, output_dir)
            
            wall_texture_path=os.path.abspath(new_wall_texture_path)
            
            wall_material = create_material_with_texture("WallTexture", wall_texture_path)

            thick=args.wall_thickness

            # Filled Blender file

            # clear_scene()
            #
            # json_path = process_image(converted_path, args.scale_factor, output_dir)
            #
            # with open(json_path, "r") as json_file:
            #     data = json.load(json_file)
            #
            # contours = data
            # centroid = calculate_centroid(contours)
            # shifted_contours = shift_coordinates(contours, centroid)
            #
            # for contour in shifted_contours:
            #     wall = create_walls_with_texture(contour["coordinates"], args.wall_height, wall_material, thick)
            #     floor = create_floor_with_texture(contour["coordinates"], floor_material)
            #
            # remove_smaller_floors()
            #
            # bpy.ops.wm.save_as_mainfile(filepath=(os.path.join(output_dir, "filled.blend")))
            # glb_path = export_as_glb(os.path.join(output_dir, "filled.glb"))

            # Unfilled Blender file

            clear_scene()
            
            json_path = process_image(converted_path, args.scale_factor, output_dir)

            with open(json_path, "r") as json_file:
                data = json.load(json_file)

            contours = data
            centroid = calculate_centroid(contours)
            shifted_contours = shift_coordinates(contours, centroid)

            for contour in shifted_contours:
                floor = create_floor_with_texture(contour["coordinates"], floor_material)

            remove_smaller_floors()

            json_path = process_image_for_dark_lines(converted_path, args.scale_factor, output_dir)

            with open(json_path, "r") as json_file:
                data = json.load(json_file)

            contours = data
            shifted_contours = shift_coordinates(contours, centroid)

            for contour in shifted_contours:
                wall = create_walls_with_texture(contour["coordinates"], args.wall_height, wall_material, thick)

            # bpy.ops.wm.save_as_mainfile(filepath=(os.path.join(output_dir, "unfilled.blend")))
            # glb_path = export_as_glb(os.path.join(output_dir, "unfilled.glb"))

            # Furnishing
            object_detection(flip_image_path, model_path=resource_path("./assets/model/model.pt"), output_dir=output_dir)

            detections_path = os.path.join(output_dir, "detections.json")
            with open(detections_path, "r") as f:
                detections = json.load(f)

            for detection in detections:
                bbox = detection["bounding_box"]
                class_name = detection["class"]

                center_x = (bbox["x1"] + bbox["x2"]) / 2
                center_y = (bbox["y1"] + bbox["y2"]) / 2

                scaled_x = center_x * args.scale_factor
                scaled_y = center_y * args.scale_factor

                shifted_x = scaled_x - centroid[0]
                shifted_y = scaled_y - centroid[1]

                dim1 = bbox["x2"] - bbox["x1"]
                dim2 = bbox["y2"] - bbox["y1"]

                rotation = 0

                if dim1 < dim2:
                    rotation = 1.5708

                if class_name == "door":
                    create_door(location=(shifted_x, shifted_y), height=args.wall_height, thickness=args.wall_thickness/2, width=(min(dim1, dim2)/2)*args.scale_factor, rotation=0 if rotation else rotation, texture_path= resource_path("./assets/furniture/wood.jpg"))    
                elif class_name == "window":
                    create_window(location=(shifted_x, shifted_y), height=args.wall_height, thickness=args.wall_thickness/2, material=wall_material,  width=((max(dim1,dim2) + 20)/2)*args.scale_factor, rotation=rotation)
                elif class_name == "double-door" or class_name == "sliding-door":
                    create_double_door(location=(shifted_x, shifted_y), height=args.wall_height, thickness=args.wall_thickness/2,  width=(max(dim1, dim2)/2)*args.scale_factor, rotation=rotation, texture_path= resource_path("./assets/furniture/wood.jpg") )
                elif class_name == "stairs":
                    create_stairs(location=(shifted_x, shifted_y), height=args.wall_height, radius=1, step_height=0.4, step_width=((min(dim1, dim2)/2)*args.scale_factor)/3, step_thickness=0.8, texture_path= resource_path("./assets/furniture/wood.jpg"))

            bpy.ops.wm.save_as_mainfile(filepath=(os.path.join(output_dir, "furnished.blend")))
            glb_path = export_as_glb(os.path.join(output_dir, "furnished.glb"))
       
        elif args.stack:
            clear_scene()
            stack_floors(output_dir)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
