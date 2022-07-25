# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter
import pprint
import numpy as np
import random as random
from PIL import Image
import simulation_utils.get_b_box as get_b_box
import png
import cv2


# SEED = 10
# np.random.seed(SEED)
# random.seed(SEED)
"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  print('could not import')
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.") 
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='../clevr-dataset-gen/image_generation/data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='../clevr-dataset-gen/image_generation/data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='../clevr-dataset-gen/image_generation/data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='../clevr-dataset-gen/image_generation/data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_objects', default=1, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=3, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--max_retries', default=50, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--output_image_dir', default='output_files/output_train_1d/images/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='output_files/output_train_1d/scenes/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='output_files/output_train_1d/CLEVR_scenes.json',
    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='output/blendfiles',
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_blendfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=320, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=240, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")




parser.add_argument('--loc', default=3, type=float,
  help="y coordinate of object")
parser.add_argument('--x_loc', default=2, type=float,
  help="x coordinate of object")
parser.add_argument('--obj_name', default='SmoothCylinder',
  help="name of object")
parser.add_argument('--obj_name_out', default='sphere',
  help="Codeword for object")
parser.add_argument('--output_folder', default='output_files/',
  help="location to save files")

parser.add_argument('--render_quality_random', default=0, type=int,
  help="if quality if randomly selected or not")
parser.add_argument('--render_quality_prob', default=0, type=float,
  help="probability to sample low quality images")

parser.add_argument('--render_material_random', default=1, type=int,
  help="if material if randomly selected or not")
parser.add_argument('--render_material_prob', default=0, type=float,
  help="probability to sample material 1 images")

parser.add_argument('--r', default=0.35, type=float,
  help="probability to sample material 1 images")

def rand(L):
    return 2.0 * L * (random.random() - 0.5)

def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

def main(args):
  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  label_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)


  args.output_image_dir = args.output_folder + '/images/'
  args.output_label_dir = args.output_folder + '/labels/'
  args.output_scene_dir = args.output_folder + '/scenes/'
  args.output_blend_dir = args.output_folder + '/blendfiles/'
  args.output_scene_file = args.output_folder + '/CLEVR_scenes.json'

  img_template = os.path.join(args.output_image_dir, img_template)
  label_template = os.path.join(args.output_label_dir, label_template)
  scene_template = os.path.join(args.output_scene_dir, scene_template)
  blend_template = os.path.join(args.output_blend_dir, blend_template)

  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_label_dir):
    os.makedirs(args.output_label_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)
  
  


  all_scene_paths = []

  for i in range(args.num_images):
    img_path = img_template % (i + args.start_idx)
    label_path = label_template % (i + args.start_idx)
    scene_path = scene_template % (i + args.start_idx)
    all_scene_paths.append(scene_path)
    blend_path = None
    if args.save_blendfiles == 1:
      blend_path = blend_template % (i + args.start_idx)

    ########################################################################################
    ########################################################################################
    ########################################################################################

    num_objects = random.randint(args.min_objects, args.max_objects)

    # Add random jitter to camera position
    camera_jitter = []
    if args.camera_jitter > 0:
      for i in range(3):
        camera_jitter = np.append(camera_jitter, rand(args.camera_jitter))

    # Add random jitter to lamp positions
    key_light_jitter, back_light_jitter, fill_light_jitter = [],[],[]
    if args.key_light_jitter > 0:
      for i in range(3):
        key_light_jitter = np.append(key_light_jitter, rand(args.key_light_jitter))
    if args.back_light_jitter > 0:
      for i in range(3):
        back_light_jitter = np.append(back_light_jitter, rand(args.back_light_jitter))
    if args.fill_light_jitter > 0:
      for i in range(3):
        fill_light_jitter = np.append(fill_light_jitter, rand(args.fill_light_jitter))

    if args.render_quality_random ==1:
      args.render_num_samples=random.choice([64, 128, 256, 512, 1024, 2048])
    else:
      quality_choices=[8,64]
      probabilities_quality_choices= [args.render_quality_prob, 1 - args.render_quality_prob]
      samples_n = np.random.multinomial(1, probabilities_quality_choices)
      samples_n = quality_choices[samples_n.tolist().index(1)]
      args.render_num_samples=samples_n

    ########################################################################################
    ###################                   OBJECT Info                    ###################
    # Load the property file
    with open(args.properties_json, 'r') as f:
      properties = json.load(f)
      color_name_to_rgba = {}
      for name, rgb in properties['colors'].items():
        rgba = [float(c) / 255.0 for c in rgb] + [1.0]
        color_name_to_rgba[name] = rgba
      material_mapping = [(v, k) for k, v in properties['materials'].items()]
      object_mapping = [(v, k) for k, v in properties['shapes'].items()]
      size_mapping = list(properties['sizes'].items())

    shape_color_combos = None
    if args.shape_color_combos_json is not None:
      with open(args.shape_color_combos_json, 'r') as f:
        shape_color_combos = list(json.load(f).items())

    objects_info = []
    for i in range(num_objects):
      # Choose a random size
      # size_name, r = random.choice(size_mapping)
      size_name, r = 'small', args.r

      # Try to place the object
      x = random.uniform(-3, 3)
      print(x)
      y = random.uniform(-2.8, 3.2)
      # x = random.uniform(-args.x_loc, args.x_loc)
      # x=args.x_loc
      # y=3
      # y=args.loc

      # Choose random color and shape
      # obj_name, obj_name_out = random.choice(object_mapping)
      # random.seed(SEED)
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))
      obj_name, obj_name_out = args.obj_name, args.obj_name_out
      # color_name, rgba = list(color_name_to_rgba.items())[0]

      # For cube, adjust the size a bit
      if obj_name == 'Cube':
        r /= math.sqrt(2)

      # Choose random orientation for the object.
      # random.seed(SEED)
      theta = 360.0 * random.random()
      # theta=60

      # Attach a random material
      # random.seed(SEED)
      material_mapping=[("Rubber", "rubber"), ("MyMetal", "metal")]
      if args.render_material_random ==1:
        mat_name, mat_name_out = random.choice(material_mapping)
      else:
        probabilities_material_choices= [args.render_material_prob, 1 - args.render_material_prob]
        samples_n = np.random.multinomial(1, probabilities_material_choices)
        samples_n = material_mapping[samples_n.tolist().index(1)]
        mat_name, mat_name_out=samples_n
      print(mat_name, mat_name_out)

      objects_info.append({
      'size_name': size_name,
      'r':r,
      'x':x,
      'y':y,
      'obj_name_out':obj_name_out,
      'obj_name':obj_name,
      'rgba':rgba,
      'color_name':color_name,
      'theta':theta,
      'mat_name':mat_name,
      'mat_name_out':mat_name_out,
      })
      print(objects_info)

    ########################################################################################
    ########################################################################################
    render_scene(args,
      num_objects=num_objects,
      output_index=(i + args.start_idx),
      output_split=args.split,
      output_image=img_path,
      output_label=label_path,
      output_scene=scene_path,
      output_blendfile=blend_path,
      camera_jitter=camera_jitter,
      key_light_jitter=key_light_jitter,
      back_light_jitter=back_light_jitter,
      fill_light_jitter=fill_light_jitter,
      objects_info=objects_info,
    )

  # After rendering all images, combine the JSON files for each scene into a
  # single JSON file.
  all_scenes = []
  # for scene_path in all_scene_paths:
  #   with open(scene_path, 'r') as f:
  #     all_scenes.append(json.load(f))
  output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': args.split,
      'license': args.license,
    },
    'scenes': all_scenes
  }
  # with open(args.output_scene_file, 'w') as f:
  #   json.dump(output, f)



def render_scene(args,num_objects=5,output_index=0,output_split='none',output_image='render.png',output_label='label.png',output_scene='render_json',output_blendfile=None,camera_jitter=None,key_light_jitter=None,back_light_jitter=None,fill_light_jitter=None,objects_info=None):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
  render_args = bpy.context.scene.render
  render_args.engine = "CYCLES"
  render_args.filepath = output_image
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100
  render_args.tile_x = args.render_tile_size
  render_args.tile_y = args.render_tile_size
  if args.use_gpu == 1:
    # Blender changed the API for enabling CUDA at some point
    if bpy.app.version < (2, 78, 0):
      bpy.context.user_preferences.system.compute_device_type = 'CUDA'
      bpy.context.user_preferences.system.compute_device = 'CUDA_0'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific stuff
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  # Add jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += camera_jitter[i]

  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  camera = bpy.data.objects['Camera']
  plane_normal = plane.data.vertices[0].normal
  cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
  cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  # Delete the plane; we only used it for normals anyway. The base scene file
  # contains the actual ground plane.
  utils.delete_object(plane)

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # Add jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += key_light_jitter[i]
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += back_light_jitter[i]
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += fill_light_jitter[i]

  # Now make some random objects
  objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera, objects_info[0]['size_name'], objects_info[0]['r'], objects_info[0]['x'], objects_info[0]['y'], objects_info[0]['obj_name_out'], objects_info[0]['obj_name'], objects_info[0]['rgba'], objects_info[0]['color_name'], objects_info[0]['theta'], objects_info[0]['mat_name'], objects_info[0]['mat_name_out'])


  ### get b_box
  box_dict, point_dict = get_b_box.main(bpy.context, blender_objects)
  label_image = np.zeros((args.height, args.width))
  for _id in box_dict:
    objects[_id]['bbox'] = box_dict[_id]
    objects[_id]['points'] = point_dict[_id]
    print(_id)
    # print(objects[_id]['points'])
    # print(len(objects[_id]['points']))
    for point in objects[_id]['points']:
      # print(point)
      # print(modif_point)
      label_image[args.height-int(clamp(point[1], 0.0, args.height-1)), int(clamp(point[0], 0.0, args.width-1))] = _id+1

  # print(label_image.unique())
  # print(np.unique(label_image))
  label_image=label_image.astype(np.uint8)
  label_image[label_image>0] = 255
  # figure(1)
  # imshow(A, interpolation='nearest')
  img = Image.fromarray(label_image, mode='L')
  img.save(output_label)

  # plt.imsave(output_label, label_image)
  # png.from_array(label_image, 'P').save(output_label)

  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  # with open(output_scene, 'w') as f:
  #   json.dump(scene_struct, f, indent=2)

  # if output_blendfile is not None:
  #   bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


def add_random_objects(scene_struct, num_objects, args, camera, size_name, r, x, y, obj_name_out, obj_name, rgba, color_name, theta, mat_name, mat_name_out):
  """
  Add random objects to the current blender scene
  """

  positions = []
  objects = []
  blender_objects = []
  for i in range(num_objects):
    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    
    utils.add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    print(x,y, obj.location, pixel_coords)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })

  return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
  
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships

if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')

