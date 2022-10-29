
# Example:
# blender --background --python mytest.py -- --views 10 /path/to/my.obj
#

import argparse, sys, os, math, re, random

try:
    import bpy, mathutils
except ModuleNotFoundError:
    print("Blender libraries not found. Invoke via blender 2.90 via:")
    print("\t blender --background --python gen_msn.py -- [args]")
    exit(1)
import numpy as np
from glob import glob

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=30,
                    help='number of views to be rendered')
parser.add_argument('--base_scene_blendfile', default='data/base_scene_bg.blend',
    help="Base blender file on which all scenes are based; includes " +
         "ground plane, lights, and camera.")
parser.add_argument('--output_folder', type=str, default='.',
                    help='The path the output will be dumped to.')
parser.add_argument('--load_metadata', type=str,
                    help='Metadata to render.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=0.025,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument('--engine', type=str, default='CYCLES',
                    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')

parser.add_argument('--width', default=320, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=240, type=int,
    help="The height (in pixels) for the rendered images")

parser.add_argument('--render_tile_size', default=2048, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--use_gpu', action='store_true',
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--start_idx', default=0, type=int,
    help="Start generating at this index.")
parser.add_argument('--end_idx', default=100000, type=int,
    help="Stop generating at this index.")
parser.add_argument('--resume', action='store_true',
    help="Resume from backup metadata file.")
parser.add_argument('--onlymasks', action='store_true',
    help="Make ONLY the mask pass.")

parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")

parser.add_argument('--obj_scale', type=float, default=2.9,
                    help='Scale of the objects.')
parser.add_argument('--mode', type=str, default='train',
                    help='Mode for selecting the set of objects (train/val/test)')
argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


def remove_fancyness(objects):
    bpy.data.worlds['World'].cycles.sample_as_light = False
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.cycles.transparent_min_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 1

    for obj in objects:
        obj.data.materials.clear()
        #for material_slot in obj.material_slots:
            #material = material_slot.material
            #material.blend_method = 'OPAQUE'
        #obj.cycles_visibility.glossy = fancy


def setup_renderer(fancy=False):
    # Set up rendering
    context = bpy.context
    scene = bpy.context.scene
    render = bpy.context.scene.render
    render.engine = args.engine
    render.image_settings.color_mode = 'RGBA' # ('RGB', 'RGBA', ...)
    render.image_settings.color_depth = args.color_depth # ('8', '16')
    render.image_settings.file_format = args.format # ('PNG', 'OPEN_EXR', 'JPEG, ...)
    render.resolution_x = args.width
    render.resolution_y = args.height
    render.resolution_percentage = 100
    render.film_transparent = True
    if args.use_gpu:
        print('Using CUDA')
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # or "OPENCL"
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            print(list(d.items()))
            #d["use"] = 1 # Using all devices, include GPU and CPU
    render.tile_x = args.render_tile_size
    render.tile_y = args.render_tile_size

    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces

    scene.use_nodes = True

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    render_layers = nodes.new('CompositorNodeRLayers')

    # Create depth output nodes
    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.base_path = ''
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = args.format
    depth_file_output.format.color_depth = '16'
    if args.format == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
        depth_file_output.format.color_mode = "BW"

        # Remap as other types can not represent the full range of depth.
        affine_node = nodes.new(type="CompositorNodeMapValue")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        affine_node.size = [args.depth_scale]

        links.new(render_layers.outputs['Depth'], affine_node.inputs[0])
        links.new(affine_node.outputs[0], depth_file_output.inputs[0])

    # Create normal output nodes
    scale_node = nodes.new(type="CompositorNodeMixRGB")
    scale_node.blend_type = 'MULTIPLY'
    # scale_node.use_alpha = True
    scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs['Normal'], scale_node.inputs[1])

    bias_node = nodes.new(type="CompositorNodeMixRGB")
    bias_node.blend_type = 'ADD'
    # bias_node.use_alpha = True
    bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_node.outputs[0], bias_node.inputs[1])

    normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    normal_file_output.base_path = ''
    normal_file_output.file_slots[0].use_node_format = True
    normal_file_output.format.file_format = args.format
    links.new(bias_node.outputs[0], normal_file_output.inputs[0])

    # Create albedo output nodes
    """
    alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
    links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
    links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])

    albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = 'Albedo Output'
    albedo_file_output.base_path = ''
    albedo_file_output.file_slots[0].use_node_format = True
    albedo_file_output.format.file_format = args.format
    albedo_file_output.format.color_mode = 'RGBA'
    albedo_file_output.format.color_depth = args.color_depth
    links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])
    """

    # Create id map output nodes
    id_file_output = nodes.new(type="CompositorNodeOutputFile")
    id_file_output.label = 'ID Output'
    id_file_output.base_path = ''
    id_file_output.file_slots[0].use_node_format = True
    id_file_output.format.file_format = args.format
    id_file_output.format.color_depth = '8'

    if args.format == 'OPEN_EXR':
        links.new(render_layers.outputs['IndexOB'], id_file_output.inputs[0])
    else:
        id_file_output.format.color_mode = 'BW'

        divide_node = nodes.new(type='CompositorNodeMath')
        divide_node.operation = 'DIVIDE'
        divide_node.use_clamp = False
        divide_node.inputs[1].default_value = 2**(int(args.color_depth))

        links.new(render_layers.outputs['IndexOB'], divide_node.inputs[0])
        links.new(divide_node.outputs[0], id_file_output.inputs[0])

    outputs = {'depth': depth_file_output,
               'normal': normal_file_output,
               #'albedo': albedo_file_output,
               'id_file': id_file_output}
    return outputs


def get_category_dir(obj_type):
    obj_type_name = ['chairs', 'tables', 'cabinets'][obj_type]
    category_dir = os.path.join('data', 'shapes', obj_type_name)
    return category_dir, obj_type_name


shape_dict = {}
def sample_shape(obj_type, mode='train'):
    category_dir, obj_type_name = get_category_dir(obj_type)
    if not obj_type_name in shape_dict:
        with open(os.path.join('data', 'shapes', f'{obj_type_name}_{mode}.lst'), 'r') as shape_list:
            shapes = shape_list.readlines()
            shapes = [s.rstrip() for s in shapes if len(s) > 3]
            shape_dict[obj_type_name] = shapes
    else:
        shapes = shape_dict[obj_type_name]
    shape_name = random.choice(shapes)
    model_path = os.path.join('models', 'model_normalized.obj')
    shape_file = os.path.join(category_dir, shape_name, model_path)
    if not os.path.exists(shape_file):
        print('Warning: shape {shape_file} not found.')
        return sample_shape(obj_type, mode)
    return shape_file, shape_name


def get_bbox_corners(obj):
    bbox_vertices = [mathutils.Vector(v) for v in obj.bound_box]
    print(bbox_vertices)
    mat = obj.matrix_world
    world_bbox_vertices = [mat @ v for v in bbox_vertices]
    print(world_bbox_vertices)

    return world_bbox_vertices


def delete_object(obj):
    """ Delete a specified blender object """
    for o in bpy.data.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.ops.object.delete()


def generate_scene(metadata, idx, max_obj=4, num_objects=None):
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    if args.camera_jitter > 0:
        camera_jitter = np.random.uniform(-args.camera_jitter,
                                          args.camera_jitter, size=3)
        camera_pos = np.array(bpy.data.objects['Camera'].location) + camera_jitter
        metadata['camera_pos'][idx, 0] = camera_pos

        for i in range(3):
            bpy.data.objects['Camera'].location[i] = camera_pos[i]

    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        metadata['light_jitter'][idx, 0] = np.random.uniform(-args.key_light_jitter,
                                                             args.key_light_jitter, size=3)
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += metadata['light_jitter'][idx, 0, i]
    if args.back_light_jitter > 0:
        metadata['light_jitter'][idx, 1] = np.random.uniform(-args.back_light_jitter,
                                                             args.back_light_jitter, size=3)
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += metadata['light_jitter'][idx, 1, i]
    if args.fill_light_jitter > 0:
        metadata['light_jitter'][idx, 2] = np.random.uniform(-args.fill_light_jitter,
                                                             args.fill_light_jitter, size=3)
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += metadata['light_jitter'][idx, 2, i]


    if num_objects is None:
        num_objects = random.randrange(2, max_obj+1)
    objects_placed = 0
    num_tries = 0
    objects = []
    positions = []
    sizes = []

    while objects_placed < num_objects:
        num_tries += 1
        if num_tries > 20:  # Start over when we're stuck.
            return generate_scene(metadata, idx, max_obj, num_objects=num_objects)

        # Try to place object at random position and angle.
        loc_x = random.uniform(-2.9, 2.9)
        loc_y = random.uniform(-2.9, 2.9)
        theta = random.uniform(0, 2 * math.pi)
        obj_type = random.randrange(3)
        shape_file, shape_name = sample_shape(obj_type)
        obj = import_object(shape_file, (loc_x, loc_y), theta=theta, idx=objects_placed+1)
        dimensions = obj.dimensions
        size = math.sqrt((dimensions[0]/2)**2 + (dimensions[1]/2)**2)

        success = True
        # Check if the new object is too close to previously placed ones.
        for i in range(objects_placed):
            dx = loc_x - positions[i][0]
            dy = loc_y - positions[i][1]
            d = math.sqrt(dx**2 + dy**2)
            if d < (size + sizes[i]) * 1.1:
                success = False
                break
        if not success:
            print('DELETE OBJECT', dx, dy, size, sizes[i])
            delete_object(obj)
        else:
            objects_placed += 1
            objects.append(obj)
            positions.append((loc_x, loc_y))
            sizes.append(size)

            metadata['rotation'][idx, objects_placed] = theta
            metadata['shape'][idx, objects_placed] = obj_type + 1
            metadata['3d_coords'][idx, objects_placed] = np.array(obj.location)
            print('SHAPENAME', shape_name)
            metadata['shape_file'][idx, objects_placed] = shape_name
            print(metadata['shape_file'][idx, objects_placed])

    return objects


def load_scene(metadata, idx):
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    for i in range(3):
        bpy.data.objects['Camera'].location[i] = metadata['camera_pos'][idx, 0, i]

    for i in range(3):
        bpy.data.objects['Lamp_Key'].location[i] += metadata['light_jitter'][idx, 0, i]
    for i in range(3):
        bpy.data.objects['Lamp_Back'].location[i] += metadata['light_jitter'][idx, 1, i]
    for i in range(3):
        bpy.data.objects['Lamp_Fill'].location[i] += metadata['light_jitter'][idx, 2, i]

    max_obj = metadata['shape'].shape[1]
    objects = []

    for i in range(max_obj):
        shape = metadata['shape'][idx, i]
        if shape == 0:
            continue

        loc_x = metadata['3d_coords'][idx, i, 0]
        loc_y = metadata['3d_coords'][idx, i, 1]
        theta = metadata['rotation'][idx, i]
        shape_name = metadata['shape_file'][idx, i].decode('utf-8')
        print(shape_name)

        category_dir, _ = get_category_dir(shape - 1)
        shape_file = os.path.join(category_dir, shape_name, 'models', 'model_normalized.obj')
        
        obj = import_object(shape_file, (loc_x, loc_y), theta=theta, idx=i)
        objects.append(obj)
    return objects


def import_object(path, loc, theta=0, idx=1):
    # Import textured mesh
    bpy.ops.object.select_all(action='DESELECT')

    bpy.ops.import_scene.obj(filepath=path)

    context = bpy.context
    obj = bpy.context.selected_objects[0]
    context.view_layer.objects.active = obj

    # Possibly disable specular shading
    for slot in obj.material_slots:
        node = slot.material.node_tree.nodes['Principled BSDF']
        node.inputs['Specular'].default_value = 0.05

    if args.remove_doubles:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
    if args.edge_split:
        bpy.ops.object.modifier_add(type='EDGE_SPLIT')
        context.object.modifiers["EdgeSplit"].split_angle = 1.32645
        bpy.ops.object.modifier_apply(modifier="EdgeSplit")

    # Set objekt IDs
    obj.pass_index = idx

    x, y = loc
    obj.rotation_euler[2] = theta
    bpy.ops.transform.resize(value=(args.obj_scale, args.obj_scale, args.obj_scale))
    dimensions = np.array(obj.dimensions)
    corners = get_bbox_corners(obj)
    bottom = np.array(corners)[:, 2].min()
    z = -bottom
    obj.location = mathutils.Vector((x, y, z))

    return obj


def rotate_around_z_axis(points, theta):
    # Rotate point around the z axis
    results = np.zeros_like(points)
    results[..., 2] = points[..., 2]
    results[..., 0] = np.cos(theta) * points[..., 0] - np.sin(theta) * points[..., 1]
    results[..., 1] = np.sin(theta) * points[..., 0] + np.cos(theta) * points[..., 1]

    return results

def create_dataset(start_idx=0, end_idx=100000, max_obj=4, num_views=3, load_metadata=None):
    num_scenes = 100000
    outpath = os.path.join(os.path.abspath(args.output_folder))
    backup_file = os.path.join(outpath, f'metadata_{start_idx}-{end_idx}-backup.npz')
    loop_start_idx = start_idx
    if args.resume:
        metadata = np.load(backup_file)
        metadata = {k: v for (k, v) in metadata.items()}  # Load all arrays
        nonzero_entries = metadata['shape'].sum(1) > 0
        nonzero_idxs = np.arange(num_scenes)[nonzero_entries]
        max_nonzero_idx = nonzero_idxs.max()
        print(f'Loaded metadata file, highest entry is {max_nonzero_idx}. Resuming from {max_nonzero_idx+1}.')
        loop_start_idx = max_nonzero_idx + 1

    elif load_metadata is None:
        metadata = dict()
        metadata['camera_pos'] = np.zeros((num_scenes, num_views, 3), dtype=np.float32)
        metadata['light_jitter'] = np.zeros((num_scenes, 3, 3), dtype=np.float32)
        metadata['rotation'] = np.zeros((num_scenes, max_obj+1), dtype=np.float32)
        metadata['shape'] = np.zeros((num_scenes, max_obj+1), dtype=np.uint8)
        metadata['3d_coords'] = np.zeros((num_scenes, max_obj+1, 3), dtype=np.float32)
        metadata['shape_file'] = np.empty((num_scenes, max_obj+1), dtype='|S128')
    else:
        metadata = np.load(load_metadata)


    for i in range(loop_start_idx, end_idx):
        if load_metadata is None:
            objects = generate_scene(metadata, i, max_obj)
        else:
            objects = load_scene(metadata, i)

        base_camera_pos = metadata['camera_pos'][i, 0]
        outputs = setup_renderer()
        for j in range(num_views):
            if j > 0:
                theta = 2 * math.pi * (float(j) / num_views)
                if load_metadata:
                    camera_pos = metadata['camera_pos'][i, j]
                else:
                    camera_pos = rotate_around_z_axis(base_camera_pos, theta)
                    metadata['camera_pos'][i, j] = camera_pos
                for k in range(3):
                    bpy.data.objects['Camera'].location[k] = camera_pos[k]

            if not args.onlymasks:
                bpy.context.scene.render.filepath = os.path.join(outpath, 'images', f'img_{i}_{j}.png')
                bpy.ops.render.render(write_still=True)

        # render depths with reflections disabled
        if not args.onlymasks:
            bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
            bpy.context.scene.view_layers["RenderLayer"].use_pass_diffuse_color = True
        bpy.context.scene.view_layers["RenderLayer"].use_pass_object_index = True
        remove_fancyness(objects)
        for j in range(num_views):
            for k in range(3):
                bpy.data.objects['Camera'].location[k] = metadata['camera_pos'][i, j, k]
            if not args.onlymasks:
                depth_path = os.path.join(outpath, 'depths', f'depths_{i}_{j}.png')
                normal_path = os.path.join(outpath, 'normals', f'normals_{i}_{j}.png')
                outputs['depth'].file_slots[0].path = depth_path
                outputs['normal'].file_slots[0].path = normal_path
            label_path = os.path.join(outpath, 'masks', f'masks_{i}_{j}.png')
            outputs['id_file'].file_slots[0].path = label_path
            bpy.ops.render.render(write_still=False)
            if not args.onlymasks:
                os.rename(depth_path + '0001.png', depth_path)
                os.rename(normal_path + '0001.png', normal_path)
            os.rename(label_path + '0001.png', label_path)

        if not load_metadata and i > 0 and i % 100 == 0:
            np.savez_compressed(backup_file, **metadata)
    if not load_metadata:
        np.savez_compressed(os.path.join(outpath, f'metadata_{start_idx}-{end_idx}.npz'), **metadata)


if __name__ == '__main__':
    create_dataset(start_idx=args.start_idx, end_idx=args.end_idx, load_metadata=args.load_metadata)



