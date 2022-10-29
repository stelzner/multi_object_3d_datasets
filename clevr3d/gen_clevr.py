import argparse, json

try:
    import bpy, bpy_extras
except ModuleNotFoundError:
    print("Blender libraries not found. Invoke via blender 2.78c via:")
    print("\t blender --background --python gen_clevr.py -- [args]")
    exit(1)
import numpy as np
from numpy.linalg import norm
import mathutils

import math
import random
import sys, random, os

# Utils:

def extract_args(input_argv=None):
    """
    Pull out command-line arguments after "--". Blender ignores command-line flags
    after --, so this lets us forward command line arguments from the blender
    invocation to our own script.
    """
    if input_argv is None:
        input_argv = sys.argv
    output_argv = []
    if '--' in input_argv:
        idx = input_argv.index('--')
        output_argv = input_argv[(idx + 1):]
    return output_argv


def parse_args(parser, argv=None):
    return parser.parse_args(extract_args(argv))


def delete_object(obj):
    """ Delete a specified blender object """
    if bpy.app.version < (2, 80, 0):
        for o in bpy.data.objects:
            o.select = False
        obj.select = True
    else:
        for o in bpy.data.objects:
            o.select_set(False)
        obj.select_set(True)
    bpy.ops.object.delete()


def get_camera_coords(cam, pos):
    """
    For a specified point, get both the 3D coordinates and 2D pixel-space
    coordinates of the point from the perspective of the camera.

    Inputs:
    - cam: Camera object
    - pos: Vector giving 3D world-space position

    Returns a tuple of:
    - (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
        in the range [-1, 1]
    """
    scene = bpy.context.scene
    x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
    scale = scene.render.resolution_percentage / 100.0
    w = int(scale * scene.render.resolution_x)
    h = int(scale * scene.render.resolution_y)
    px = int(round(x * w))
    py = int(round(h - y * h))
    return (px, py, z)


def set_layer(obj, layer_idx):
    """ Move an object to a particular layer """
    # Set the target layer to True first because an object must always be on
    # at least one layer.
    obj.layers[layer_idx] = True
    for i in range(len(obj.layers)):
        obj.layers[i] = (i == layer_idx)


def add_object(object_dir, name, scale, loc, theta=0):
    """
    Load an object from a file. We assume that in the directory object_dir, there
    is a file named "$name.blend" which contains a single object named "$name"
    that has unit size and is centered at the origin.

    - scale: scalar giving the size that the object should be in the scene
    - loc: tuple (x, y) giving the coordinates on the ground plane where the
        object should be placed.
    """
    # First figure out how many of this object are already in the scene so we can
    # give the new object a unique name
    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1

    filename = os.path.join(object_dir, '%s.blend' % name, 'Object', name)
    bpy.ops.wm.append(filename=filename)

    # Give it a new name to avoid conflicts
    new_name = '%s_%d' % (name, count)
    bpy.data.objects[name].name = new_name

    # Set the new object as active, then rotate, scale, and translate it
    x, y = loc

    o = bpy.data.objects[new_name]
    if bpy.app.version < (2, 80, 0):
        bpy.context.scene.objects.active = o
    else:
        o.select_set( state = True, view_layer = bpy.context.view_layer )
        bpy.context.view_layer.objects.active = o

    bpy.context.object.rotation_euler[2] = theta
    bpy.ops.transform.resize(value=(scale, scale, scale))
    bpy.ops.transform.translate(value=(x, y, scale))


def load_materials(material_dir):
    """
    Load materials from a directory. We assume that the directory contains .blend
    files with one material each. The file X.blend has a single NodeTree item named
    X; this NodeTree item must have a "Color" input that accepts an RGBA value.
    """
    for fn in os.listdir(material_dir):
        if not fn.endswith('.blend'): continue
        name = os.path.splitext(fn)[0]
        filepath = os.path.join(material_dir, fn, 'NodeTree', name)
        bpy.ops.wm.append(filename=filepath)


def add_material(name, **properties):
    """
    Create a new material and assign it to the active object. "name" should be the
    name of a material that has been previously loaded using load_materials.
    """
    # Figure out how many materials are already in the scene
    mat_count = len(bpy.data.materials)

    # Create a new material; it is not attached to anything and
    # it will be called "Material"
    bpy.ops.material.new()

    # Get a reference to the material we just created and rename it;
    # then the next time we make a new material it will still be called
    # "Material" and we will still be able to look it up by name
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % mat_count

    # Attach the new material to the active object
    # Make sure it doesn't already have materials
    obj = bpy.context.active_object
    assert len(obj.data.materials) == 0
    obj.data.materials.append(mat)

    # Find the output node of the new material
    output_node = None
    for n in mat.node_tree.nodes:
        if n.name == 'Material Output':
            output_node = n
            break

    # Add a new GroupNode to the node tree of the active material,
    # and copy the node tree from the preloaded node group to the
    # new group node. This copying seems to happen by-value, so
    # we can create multiple materials of the same type without them
    # clobbering each other
    group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
    group_node.node_tree = bpy.data.node_groups[name]

    # Find and set the "Color" input of the new group node
    for inp in group_node.inputs:
        if inp.name in properties:
            inp.default_value = properties[inp.name]

    # Wire the output of the new group node to the input of
    # the MaterialOutput node
    mat.node_tree.links.new(
        group_node.outputs['Shader'],
        output_node.inputs['Surface'],
    )


# End of utils


def set_camera_position(camera_pos):
    for i in range(3):
        bpy.data.objects['Camera'].location[i] = camera_pos[i]


def setup_renderer(fancy=False, filename=None):
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    if filename is not None:
        render_args.filepath = filename
    else:
        render_args.filepath = 'reconstructed_render.png'

    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    if fancy:
        if args.use_gpu:
            # Blender changed the API for enabling CUDA at some point
            if bpy.app.version < (2, 78, 0):
                bpy.context.user_preferences.system.compute_device_type = 'CUDA'
                bpy.context.user_preferences.system.compute_device = 'CUDA_0'
            else:
                print('Using CUDA')
                prefs = bpy.context.user_preferences
                addons = prefs.addons
                cyc = addons['cycles']
                cycles_prefs = cyc.preferences
                cycles_prefs.compute_device_type = 'CUDA'

        bpy.data.worlds['World'].cycles.sample_as_light = True
        bpy.context.scene.cycles.blur_glossy = 2.0
        bpy.context.scene.cycles.samples = args.render_num_samples
        bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
        bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    else:
        bpy.data.worlds['World'].cycles.sample_as_light = False
        bpy.context.scene.cycles.samples = 1
        bpy.context.scene.cycles.transparent_min_bounces = 0
        bpy.context.scene.cycles.transparent_max_bounces = 0

    if args.use_gpu:
        bpy.context.scene.cycles.device = 'GPU'


def get_extras(i=0, j=0, masks=True, depths=True):
    # switch on nodes
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    rl_node = tree.nodes.new('CompositorNodeRLayers')

    if depths:
        affine_node = tree.nodes.new('CompositorNodeMapValue')
        affine_node.size = [0.025,]

        # create output node
        out_node = tree.nodes.new('CompositorNodeOutputFile')
        out_node.base_path = './depths/'
        out_node.file_slots[0].path = 'depths_' + str(i) + '_' + str(j) + '_'
        out_node.format.color_depth = '16'
        out_node.format.color_mode = 'BW'
        out_node.format.file_format = 'PNG'

        # Links
        z_socket_out = rl_node.outputs['Z']
        affine_in_socket = affine_node.inputs[0]
        affine_out_socket = affine_node.outputs[0]

        file_socket_in = out_node.inputs[0]
        print('Connecting', z_socket_out.name, 'to', affine_in_socket.name)
        links.new(z_socket_out, affine_in_socket)  # link Image output to Viewer input
        print('Connecting', affine_out_socket.name, 'to', file_socket_in.name)
        links.new(affine_out_socket, file_socket_in)  # link Image output to Viewer input

    # Create id map output nodes
    if masks:
        bpy.context.scene.render.layers["RenderLayer"].use_pass_object_index = True
        id_file_output = tree.nodes.new("CompositorNodeOutputFile")
        id_file_output.label = 'ID Output'
        id_file_output.base_path = './masks/'
        id_file_output.file_slots[0].path = 'masks_' + str(i) + '_' + str(j) + '_'
        #id_file_output.file_slots[0].use_node_format = True
        id_file_output.format.file_format = 'PNG'
        id_file_output.format.color_mode = 'BW'
        id_file_output.format.color_depth = '8'

        divide_node = tree.nodes.new(type='CompositorNodeMath')
        divide_node.operation = 'DIVIDE'
        divide_node.use_clamp = False
        divide_node.inputs[1].default_value = 2**8

        links.new(rl_node.outputs['IndexOB'], divide_node.inputs[0])
        links.new(divide_node.outputs[0], id_file_output.inputs[0])
        print('setup link', rl_node.outputs['IndexOB'], id_file_output.inputs[0])

    setup_renderer()
    bpy.ops.render.render()

    bpy.context.scene.use_nodes = False

    # Get rid of the suffix _0001 on the filename, which Blender adds automatically
    if depths:
        os.rename(os.path.join(out_node.base_path, out_node.file_slots[0].path + '0001.png'),
                  os.path.join(out_node.base_path, out_node.file_slots[0].path[:-1] + '.png'))
    if masks:
        os.rename(os.path.join(id_file_output.base_path, id_file_output.file_slots[0].path + '0001.png'),
                  os.path.join(id_file_output.base_path, id_file_output.file_slots[0].path[:-1] + '.png'))




def sample_points(num_points=1000):
    xs = np.random.uniform(-4., 4., size=num_points)
    ys = np.random.uniform(-4., 4., size=num_points)
    zs = np.random.uniform(0., 1.5, size=num_points)
    points = np.stack((xs, ys, zs), -1)
    return points


def check_points(points, objects, num_objects):
    results = np.zeros((len(points), num_objects), dtype=np.uint8)
    direction = mathutils.Vector([0., 0., 1.])

    for j, obj in objects.items():
        world_matrix = obj.matrix_world
        inverse = world_matrix.copy()
        inverse.invert()
        for i, p in enumerate(points):
            loc = inverse * mathutils.Vector(p)
            #print('transforming world pos', p, 'to object pos', loc)
            num_intersections = 0
            #print('Checking point', loc, 'with object', obj)
            while True:
                #print('Shooting ray from', loc, 'in direction', direction)
                hit_something, loc, _, idx, = obj.ray_cast(loc, direction)
                #print('Result:', hit_something, loc, idx)
                if not hit_something:
                    break
                num_intersections += 1
                loc = loc + 0.0001 * direction
            #print('Found', num_intersections, 'intersections.')
            results[i, j] = num_intersections % 2 == 1
    #print(results)
    return results


class SceneLoader():
    def __init__(self, args):
        self.args = args
        with open(args.properties_json, 'r') as f:
            self.properties = json.load(f)

        self.object_mapping = ['Sphere', 'SmoothCylinder', 'SmoothCube_v2']
        self.material_mapping = ['Rubber', 'MyMetal']
        self.color_name_to_rgba = {}
        for name, rgb in self.properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            self.color_name_to_rgba[name] = rgba

        self.idx_to_color_name = ['red', 'cyan', 'green', 'blue', 'brown', 'gray', 'purple', 'yellow']

        self.metadata = np.load(args.metadata_file)

    def load_scene(self, i):
        bpy.ops.wm.open_mainfile(filepath=self.args.base_scene_blendfile)
        load_materials(self.args.material_dir)

        metadata = self.metadata
        num_imgs, num_objs = metadata['color'].shape[:2]

        objects_blender = {}

        for j in range(num_objs):
            if metadata['shape'][i, j] == 0:
                continue
            pos = metadata['3d_coords'][i, j]
            rot = metadata['rotation'][i, j]

            name = self.object_mapping[metadata['shape'][i, j] - 1]
            add_object(self.args.shape_dir, name, pos[2], (pos[0], pos[1]), theta=rot)
            objects_blender[j] = bpy.context.scene.objects.active
            objects_blender[j].pass_index = j

            mat_name = self.material_mapping[metadata['material'][i, j] - 1]
            rgba = self.color_name_to_rgba[self.idx_to_color_name[metadata['color'][i, j] - 1]]

            add_material(mat_name, Color=rgba)

        if 'light_jitter' in metadata:
            for j, light_name in enumerate(['Lamp_Key', 'Lamp_Back', 'Lamp_Fill']):
                for k in range(3):
                    bpy.data.objects[light_name].location[k] += metadata['light_jitter'][i, j, k]

    def num_objs(self, i):
        n = (self.metadata['shape'][i] > 0).sum()
        print(n)
        return n


def process_scenes(start_idx=0, end_idx=100000, render_images=True, render_depths=True, render_masks=True, max_objs=None):
    scene_loader = SceneLoader(args)

    camera_positions = scene_loader.metadata['camera_pos']
    thetas = [0, 120, 240]

    for i in range(start_idx, end_idx):
        if max_objs is not None and scene_loader.num_objs(i) > max_objs:
            continue
        scene_loader.load_scene(i)

        for c_idx, theta in enumerate(thetas):
            c_pos = camera_positions[i, c_idx]
            set_camera_position(c_pos)

            if render_images:
                # Output renders
                setup_renderer(fancy=True, filename='images/img_%d_%d.png' % (i, c_idx))
                bpy.ops.render.render(write_still=True)

            if render_depths or render_masks:
                get_extras(i, c_idx, depths=render_depths, masks=render_masks)


        # Generate point clouds. Not used or tested.
        if False:
            num_points = 10000
            points = sample_points(num_points)
            results = check_points(points, objects_blender, num_objs)
            print(points.shape, results.shape)
            in_obj = results.sum(1) > 0

            num_in_obj = in_obj.sum()
            print(num_in_obj, '/', num_points, 'points are in an object')
            np.savez_compressed(os.path.join('points', 'points_%d.npz' % i), pos=points, obj=results)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata_file', help='npy/json file specifying scenes')
    parser.add_argument('--start_idx', type=int, help='JSON file specifying scene', default=0)
    parser.add_argument('--end_idx', type=int, help='JSON file specifying scene', default=100000)
    parser.add_argument('--max_objs', type=int, help='Only render scenes with this maximum number of objects.')
    parser.add_argument('--base_scene_blendfile', default='data/base_scene_bg.blend',
        help="Base blender file on which all scenes are based; includes " +
             "ground plane, lights, and camera.")
    parser.add_argument('--material_dir', default='data/materials',
        help="Directory where .blend files for materials are stored")
    parser.add_argument('--shape_dir', default='data/shapes',
        help="Directory where .blend files for object models are stored")

    parser.add_argument('--width', default=320, type=int,
        help="The width (in pixels) for the rendered images")
    parser.add_argument('--height', default=240, type=int,
        help="The height (in pixels) for the rendered images")

    parser.add_argument('--render_images', action='store_true', help="Render RGB views")
    parser.add_argument('--render_depths', action='store_true', help="Render depth channel")
    parser.add_argument('--render_masks', action='store_true', help="Render object masks")

    parser.add_argument('--render_tile_size', default=256, type=int,
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
        help="Setting --use_gpu enables GPU-accelerated rendering using CUDA. " +
             "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
             "to work.")

    parser.add_argument('--properties_json', default='data/properties.json',
        help="JSON file defining objects, materials, sizes, and colors. " +
             "The \"colors\" field maps from CLEVR color names to RGB values; " +
             "The \"sizes\" field maps from CLEVR size names to scalars used to " +
             "rescale object models; the \"materials\" and \"shapes\" fields map " +
             "from CLEVR material and shape names to .blend files in the " +
             "--object_material_dir and --shape_dir directories respectively.")
    argv = extract_args()
    args = parser.parse_args(argv)

    process_scenes(start_idx=args.start_idx, end_idx=args.end_idx,
            render_images=args.render_images, render_depths=args.render_depths, render_masks=args.render_masks,
            max_objs=args.max_objs)

