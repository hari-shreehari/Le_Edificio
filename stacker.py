import bpy
import os
import numpy as np

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    print("Scene cleared.")

def import_glb(filepath):
    bpy.ops.import_scene.gltf(filepath=filepath)

def get_tallest_wall_height(objects):
    tallest_wall_height = 0.0
    for obj in objects:
        if obj.type == 'MESH' and "wall" in obj.name.lower():
            obj.update_from_editmode()
            world_matrix = np.array([[obj.matrix_world[i][j] for j in range(4)] for i in range(4)])
            corners = np.array([[x, y, z, 1.0] for x, y, z in obj.bound_box])
            transformed_corners = np.dot(corners, world_matrix.T)
            height = np.max(transformed_corners[:, 2])
            tallest_wall_height = max(tallest_wall_height, height)
    return tallest_wall_height if tallest_wall_height > 0 else None

def duplicate_floor(objects):
    floor_objects = [obj for obj in objects if "floor" in obj.name.lower()]
    duplicated_floors = []

    for floor in floor_objects:
        floor_copy = floor.copy()
        bpy.context.collection.objects.link(floor_copy)
        duplicated_floors.append(floor_copy)

    return duplicated_floors

def stack_floors(directory):
    glb_files = {
        int((f.split('.')[0])[5:]): f for f in os.listdir(directory)
        if f.lower().endswith('.glb') and f[:5].lower() == "floor"
    }

    current_z_offset = 0.0
    all_imported_objects = []

    for floor_number in sorted(glb_files.keys()):
        glb_file = glb_files[floor_number]
        filepath = os.path.join(directory, glb_file)

        import_glb(filepath)

        imported_objects = [obj for obj in bpy.context.selected_objects if obj.type in {'MESH', 'EMPTY', 'LIGHT'}]

        wall_height = get_tallest_wall_height(imported_objects)
        floor_height = wall_height if wall_height is not None else 0.0

        roof_objects = duplicate_floor(imported_objects)

        for obj in imported_objects:
            obj.location.z += current_z_offset

        for roof in roof_objects:
            roof.location.z = current_z_offset + floor_height

        current_z_offset += floor_height

        all_imported_objects.extend(imported_objects)
        all_imported_objects.extend(roof_objects)

        bpy.ops.object.select_all(action='DESELECT')

    export_filepath = os.path.join(directory, "final.glb")
    export_to_glb(all_imported_objects, export_filepath)
    print(f"Exported stacked floors with duplicated roofs to {export_filepath}")

def export_to_glb(objects, filepath):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=filepath, export_format='GLB', export_apply=True)
    bpy.ops.object.select_all(action='DESELECT')

