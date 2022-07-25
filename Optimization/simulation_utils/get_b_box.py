import bpy
from mathutils import Vector
# import matplotlib.pyplot as plt

class Box:

    dim_x = 1
    dim_y = 1

    def __init__(self, min_x, min_y, max_x, max_y, dim_x=dim_x, dim_y=dim_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.dim_x = dim_x
        self.dim_y = dim_y

    @property
    def x(self):
        return round(self.min_x * self.dim_x)

    @property
    def y(self):
        return round(self.dim_y - self.max_y * self.dim_y)

    @property
    def width(self):
        return round((self.max_x - self.min_x) * self.dim_x)

    @property
    def height(self):
        return round((self.max_y - self.min_y) * self.dim_y)

    def __str__(self):
        return "<Box, x=%i, y=%i, width=%i, height=%i>" % \
               (self.x, self.y, self.width, self.height)

    def to_tuple(self):
        if self.width == 0 or self.height == 0:
            return (0, 0, 0, 0)
        return (self.x, self.y, self.width, self.height)

# Copied from StackOverflow
# https://blender.stackexchange.com/questions/7198/save-the-2d-bounding-box-of-an-object-in-rendered-image-to-a-text-file
def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.

    Negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """

    mat = cam_ob.matrix_world.normalized().inverted()
    print(me_ob.data)
    me = me_ob.to_mesh(scene, True, 'RENDER')
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'

    lx = []
    ly = []
    lz = []
    # print(me)
    # input('')
    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)
        lz.append(-z)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    bpy.data.meshes.remove(me)

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    lxyz = list(zip([it*dim_x for it in lx], [it*dim_y for it in ly], lz))
    lxyz = [[x,y] for x,y in zip([it*dim_x for it in lx], [it*dim_y for it in ly])]

    return Box(min_x, min_y, max_x, max_y, dim_x, dim_y), lxyz 


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


def write_bounds_2d_return(scene, cam_ob, me_ob, frame_start, frame_end):
    box_dict = {}
    point_dict={}
    for frame in range(frame_start, frame_start + 1):
        bpy.context.scene.frame_set(frame)
        for _id, one_me_ob in enumerate(me_ob):
            _box, _points = camera_view_bounds_2d(scene, cam_ob, one_me_ob)
            curbox = _box.to_tuple()
            box_dict[_id] = curbox
            point_dict[_id] = _points
            # plt.imshow(X, cmap="gray")
            # plt.show()
    return box_dict, point_dict


def main(context, me_obs):

    scene = context.scene
    cam_ob = scene.camera
    me_ob = me_obs

    frame_current = scene.frame_current
    frame_start = scene.frame_start
    frame_end = scene.frame_end

    return write_bounds_2d_return(scene, cam_ob, me_ob, frame_start, frame_end)