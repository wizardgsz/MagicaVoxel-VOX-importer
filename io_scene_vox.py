"""
This script imports MagicaVoxel VOX files to Blender.

Usage:
Run this script from "File->Import" menu and then load the desired VOX file.
"""

# <pep8 compliant>

import os

import bpy
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, CollectionProperty
from bpy_extras.io_utils import ImportHelper

import struct

bl_info = {
    "name": "MagicaVoxel VOX format",
    "author": "Richard Spencer, Gabriele Scibilia",
    "version": (2, 1),
    "blender": (2, 80, 0),
    "location": "File > Import-Export",
    "description": "Import MagicaVoxel .vox files",
    "warning": "",
    "wiki_url": "",
    "support": 'TESTING',
    "category": "Import-Export"}


#
# =================================================================================================
#     MagicaVoxel - https://ephtracy.github.io/
#           It is a free, open source software package that provides simple, yet surprisingly 
#           powerful 3D modeling capability. 
#           Creator ephtracy explains: A free lightweight 8-bit voxel art editor and interactive 
#           path tracing renderer.
# =================================================================================================
#
# Default palette, given in .vox 150 specification format
DEFAULT_PALETTE = [0x00000000, 0xffffffff, 0xffccffff, 0xff99ffff, 0xff66ffff, 0xff33ffff, 0xff00ffff, 0xffffccff,
                   0xffccccff, 0xff99ccff, 0xff66ccff, 0xff33ccff, 0xff00ccff, 0xffff99ff, 0xffcc99ff, 0xff9999ff,
                   0xff6699ff, 0xff3399ff, 0xff0099ff, 0xffff66ff, 0xffcc66ff, 0xff9966ff, 0xff6666ff, 0xff3366ff,
                   0xff0066ff, 0xffff33ff, 0xffcc33ff, 0xff9933ff, 0xff6633ff, 0xff3333ff, 0xff0033ff, 0xffff00ff,
                   0xffcc00ff, 0xff9900ff, 0xff6600ff, 0xff3300ff, 0xff0000ff, 0xffffffcc, 0xffccffcc, 0xff99ffcc,
                   0xff66ffcc, 0xff33ffcc, 0xff00ffcc, 0xffffcccc, 0xffcccccc, 0xff99cccc, 0xff66cccc, 0xff33cccc,
                   0xff00cccc, 0xffff99cc, 0xffcc99cc, 0xff9999cc, 0xff6699cc, 0xff3399cc, 0xff0099cc, 0xffff66cc,
                   0xffcc66cc, 0xff9966cc, 0xff6666cc, 0xff3366cc, 0xff0066cc, 0xffff33cc, 0xffcc33cc, 0xff9933cc,
                   0xff6633cc, 0xff3333cc, 0xff0033cc, 0xffff00cc, 0xffcc00cc, 0xff9900cc, 0xff6600cc, 0xff3300cc,
                   0xff0000cc, 0xffffff99, 0xffccff99, 0xff99ff99, 0xff66ff99, 0xff33ff99, 0xff00ff99, 0xffffcc99,
                   0xffcccc99, 0xff99cc99, 0xff66cc99, 0xff33cc99, 0xff00cc99, 0xffff9999, 0xffcc9999, 0xff999999,
                   0xff669999, 0xff339999, 0xff009999, 0xffff6699, 0xffcc6699, 0xff996699, 0xff666699, 0xff336699,
                   0xff006699, 0xffff3399, 0xffcc3399, 0xff993399, 0xff663399, 0xff333399, 0xff003399, 0xffff0099,
                   0xffcc0099, 0xff990099, 0xff660099, 0xff330099, 0xff000099, 0xffffff66, 0xffccff66, 0xff99ff66,
                   0xff66ff66, 0xff33ff66, 0xff00ff66, 0xffffcc66, 0xffcccc66, 0xff99cc66, 0xff66cc66, 0xff33cc66,
                   0xff00cc66, 0xffff9966, 0xffcc9966, 0xff999966, 0xff669966, 0xff339966, 0xff009966, 0xffff6666,
                   0xffcc6666, 0xff996666, 0xff666666, 0xff336666, 0xff006666, 0xffff3366, 0xffcc3366, 0xff993366,
                   0xff663366, 0xff333366, 0xff003366, 0xffff0066, 0xffcc0066, 0xff990066, 0xff660066, 0xff330066,
                   0xff000066, 0xffffff33, 0xffccff33, 0xff99ff33, 0xff66ff33, 0xff33ff33, 0xff00ff33, 0xffffcc33,
                   0xffcccc33, 0xff99cc33, 0xff66cc33, 0xff33cc33, 0xff00cc33, 0xffff9933, 0xffcc9933, 0xff999933,
                   0xff669933, 0xff339933, 0xff009933, 0xffff6633, 0xffcc6633, 0xff996633, 0xff666633, 0xff336633,
                   0xff006633, 0xffff3333, 0xffcc3333, 0xff993333, 0xff663333, 0xff333333, 0xff003333, 0xffff0033,
                   0xffcc0033, 0xff990033, 0xff660033, 0xff330033, 0xff000033, 0xffffff00, 0xffccff00, 0xff99ff00,
                   0xff66ff00, 0xff33ff00, 0xff00ff00, 0xffffcc00, 0xffcccc00, 0xff99cc00, 0xff66cc00, 0xff33cc00,
                   0xff00cc00, 0xffff9900, 0xffcc9900, 0xff999900, 0xff669900, 0xff339900, 0xff009900, 0xffff6600,
                   0xffcc6600, 0xff996600, 0xff666600, 0xff336600, 0xff006600, 0xffff3300, 0xffcc3300, 0xff993300,
                   0xff663300, 0xff333300, 0xff003300, 0xffff0000, 0xffcc0000, 0xff990000, 0xff660000, 0xff330000,
                   0xff0000ee, 0xff0000dd, 0xff0000bb, 0xff0000aa, 0xff000088, 0xff000077, 0xff000055, 0xff000044,
                   0xff000022, 0xff000011, 0xff00ee00, 0xff00dd00, 0xff00bb00, 0xff00aa00, 0xff008800, 0xff007700,
                   0xff005500, 0xff004400, 0xff002200, 0xff001100, 0xffee0000, 0xffdd0000, 0xffbb0000, 0xffaa0000,
                   0xff880000, 0xff770000, 0xff550000, 0xff440000, 0xff220000, 0xff110000, 0xffeeeeee, 0xffdddddd,
                   0xffbbbbbb, 0xffaaaaaa, 0xff888888, 0xff777777, 0xff555555, 0xff444444, 0xff222222, 0xff111111]


#
# =================================================================================================
#     BLENDER UI Properties - IMPORT Operator
# =================================================================================================
#
class ImportVOX(bpy.types.Operator, ImportHelper):
    """Load a MagicaVoxel VOX File"""
    bl_idname = "import_scene.vox"
    bl_label = "Import VOX"
    bl_options = {'PRESET', 'UNDO'}

    files: CollectionProperty(name="File Path",
                              description="File path used for importing the VOX file",
                              type=bpy.types.OperatorFileListElement)

    directory: StringProperty()

    filename_ext = ".vox"
    filter_glob: StringProperty(
        default="*.vox",
        options={'HIDDEN'},
    )

    voxel_spacing: FloatProperty(name="Voxel Spacing", default=1.0)
    voxel_size: FloatProperty(name="Voxel Size", default=1.0)

    load_frame: IntProperty(name="Animation frame to load", default=0, min=0)

    use_bounds: BoolProperty(name="Use Voxel Bounds", default=False)

    start_voxel: IntProperty(name="Start Voxel", default=1, min=1)
    end_voxel: IntProperty(name="End Voxel", default=20, min=2)

    use_palette: BoolProperty(name="Use Palette Colors", description="Set voxels palette", default=True)

    gamma_correct: BoolProperty(name="Gamma Correct Colors", description="Apply gamma correction", default=True)
    gamma_value: FloatProperty(name="Gamma Correction Value", default=2.2, min=1, max=5)

    use_shadeless: BoolProperty(name="Use Shadeless Materials", default=False)

    use_materials: BoolProperty(name="Use MATT and MATL", description="Read and apply Matter properties to Materials", default=True)

    def execute(self, context):
        paths = [os.path.join(self.directory, name.name)
                 for name in self.files]
        if not paths:
            paths.append(self.filepath)

        keywords = self.as_keywords(ignore=("files", "filepath", "directory", "filter_glob",))

        for path in paths:
            import_vox(path, **keywords)
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "voxel_spacing")
        layout.prop(self, "voxel_size")
        layout.prop(self, "load_frame")
        layout.prop(self, "use_bounds")
        if self.use_bounds:
            layout.prop(self, "start_voxel")
            layout.prop(self, "end_voxel")
        layout.prop(self, "use_palette")
        layout.prop(self, "gamma_correct")
        if self.gamma_correct:
            layout.prop(self, "gamma_value")
        layout.prop(self, "use_shadeless")
        layout.prop(self, "use_materials")


# Defines material properties aka "Matter" in MagicaVoxel
from collections import namedtuple
#from recordtype import recordtype
material_spec = namedtuple('material_spec', [       # Material chunks, both MATT and MATL
                                'type_',            # Matter type: _diffuse, _metal, _glass, _emit (_plastic, _media for MATL)
                                'weight',           # 
                                'Plastic',          # Plastic for MATT only: MATT uses it as flag in _metal materials
                                'Roughness',        # Surface roughness
                                'Specular',         # Specular reflectivity
                                'IOR',              # Refractive index (glass specific)
                                'Attenuation',      # Attenuation (glass specific)
                                'Power',            # Power (emissive specific)
                                'Glow',             # Light glow (emissive specific)
                                'isTotalPower'      # True/False (emissive specific)
                                ], 
                                defaults = [        # TODO: Default values?
                                "_diffuse",         # type_
                                1.0,                # weight
                                0.0,                # Plastic
                                0.1,                # Roughness
                                0.5,                # Specular
                                1.30,               # IOR
                                0.0,                # Attenuation
                                0.0,                # Power
                                0.0,                # Glow
                                False               # isTotalPower
                                ])


def read_STRING(vox):
    size, = struct.unpack('<i', vox.read(4))
    str = vox.read(size)
    str = str.decode('utf-8')

    return str


def read_DICT(vox):
    num_of_key_value_pairs, = struct.unpack('<i', vox.read(4))

    dict = {}

    if num_of_key_value_pairs > 0:
        #print("\tkeys:", num_of_key_value_pairs)
        for i in range(num_of_key_value_pairs):
            str1 = read_STRING(vox)
            str2 = read_STRING(vox)
            dict[str1] = str2

            print("\t{}: {}".format(str1, str2))
            #print("\tkey-value: ('{}', '{}')".format(str1, str2))

    return dict


def read_MATT(vox, material_specs):
    matt_id, matt_type, weight = struct.unpack('<iif', vox.read(12))
    print("MATT({}):".format(matt_id))

    material_types = ["_diffuse", "_metal", "_glass", "_emit"]
    '''
    byte_value, = struct.unpack('<i', vox.read(4))
    reversed_bitstring = '{:032b}'.format(byte_value)[::-1]
    properties = ['Plastic', 'Roughness', 'Specular', 'IOR', 'Attenuation', 'Power', 'Glow', 'isTotalPower']
    property_bits = {}
    property_values = {}
    for pos, property_ in enumerate(properties):
        bit_value = reversed_bitstring[pos]
        property_bits.update({property_: bit_value})
        if bit_value == '1' and property_ != 'isTotalPower':  # isTotalPower doesn't get a value
            value,  = struct.unpack('<f', vox.read(4))
            property_values.update({property_: value})
        elif bit_value == '1' and property_ == 'isTotalPower':
            value,  = struct.unpack('<f', vox.read(4))
        #else:
        #    property_values.update({property_: '0'})
    #print("bits:", property_bits)
    #print("values:", property_values)
    # isTotalPower is never supplied by this
    material_specs.update({matt_id: material_spec(type_=material_types[matt_type], weight=weight, **property_values)})
    '''
    prop_bits, = struct.unpack('<i', vox.read(4))
    binary = bin(prop_bits)

    spec = material_spec(type_=material_types[matt_type], weight=weight)

    print("\t_type:", spec.type_)
    print("\t_weight:", spec.weight)

    if prop_bits & 1:
        value, = struct.unpack('<f', vox.read(4))
        spec = spec._replace(Plastic = value)
        if value > 0:                               # Transform to _plastic material (new in MATL, but not in MATT)
            spec = spec._replace(type_= '_plastic')
        print("\t_plastic: %f" % value)
    if prop_bits & 2:
        value, = struct.unpack('<f', vox.read(4))
        spec = spec._replace(Roughness = value)
        print("\t_rough: %f" % value)
    if prop_bits & 4:
        value, = struct.unpack('<f', vox.read(4))
        spec = spec._replace(Specular = value)
        print("\t_spec: %f" % value)
    if prop_bits & 8:
        value, = struct.unpack('<f', vox.read(4))   # _glass materials: [0.0, 2,0] range
        value = (value * 3.75) + 0.25               # Map to 0.25 -> 4 range
        spec = spec._replace(IOR = value)
        print("\t_ior: %f" % value)
    if prop_bits & 16:
        value, = struct.unpack('<f', vox.read(4))   # _glass materials
        spec = spec._replace(Attenuation = value)
        print("\t_att: %f" % value)
    if prop_bits & 32:
        value, = struct.unpack('<f', vox.read(4))   # _emit materials only: [0, 4] range
        spec = spec._replace(Power = value)
        print("\tPower: %f" % value)
    if prop_bits & 64:
        value, = struct.unpack('<f', vox.read(4))
        spec = spec._replace(Glow = value)
        print("\tGlow: %f" % value)
    if prop_bits & 128:
        spec = spec._replace(isTotalPower = True)
        print("\tisTotalPower: True")

    # isTotalPower is never supplied by this
    material_specs.update({matt_id: spec})

    return spec


def read_MATL(vox, material_specs):
    matl_id, = struct.unpack('<i', vox.read(4))
    print("MATL({}):".format(matl_id))

    properties = read_DICT(vox)

    matl_type = properties['_type']
    weight = float(properties['_weight'])

    spec = material_spec(type_=matl_type, weight=weight)

    #if '_plastic' in properties:
    #    value = float(properties['_plastic'])      # Plastic is _weight in MATL
    #    spec = spec._replace(Plastic = value)
    if '_rough' in properties:
        value = float(properties['_rough'])
        spec = spec._replace(Roughness = value)
    if '_spec' in properties:
        value = float(properties['_spec'])
        spec = spec._replace(Specular = value)
    if '_ior' in properties:
        value = float(properties['_ior'])           # _glass materials: [0.0, 2,0] range
        value = (value * 3.75) + 0.25               # Map to 0.25 -> 4 range
        spec = spec._replace(IOR = value)
    if '_att' in properties:
        value = float(properties['_att'])           # _glass materials
        spec = spec._replace(Attenuation = value)
    if '_glow' in properties:
        value = float(properties['_glow'])
        spec = spec._replace(Glow = value)
    if '_flux' in properties:
        value = float(properties['_flux'])          # _emit materials only: [0, 4] range
        spec = spec._replace(Power = value)

    # isTotalPower is never supplied by this
    material_specs.update({matl_id: spec})

    return spec


def import_vox(path, *, voxel_spacing=1, voxel_size=1, load_frame=0,
               use_bounds=False, start_voxel=None, end_voxel=None,
               use_palette=True, gamma_correct=True, gamma_value=2.2,
               use_shadeless=False, use_materials=True):
    os.system("cls")

    print("\nImporting voxel file {}\n".format(path))

    import time
    time_start = time.time()

    with open(path, 'rb') as vox:

        voxels = []
        palette = {}
        current_frame = 0
        material_specs = {}

        # assert is VOX 150 file
        assert (struct.unpack('<4ci', vox.read(8)) == (b'V', b'O', b'X', b' ', 150))

        # MAIN chunk
        assert (struct.unpack('<4c', vox.read(4)) == (b'M', b'A', b'I', b'N'))
        N, M = struct.unpack('<ii', vox.read(8))
        assert (N == 0)  # MAIN chunk should have no content

        # M is remaining # of bytes in file

        while True:
            try:
                *name, s_self, s_child = struct.unpack('<4cii', vox.read(12))
                assert (s_child == 0)  # sanity check
                name = b''.join(name).decode('utf-8')  # unsure of encoding..
            except struct.error:
                # end of file
                break
            if name == 'PACK':
                # number of models
                num_models, = struct.unpack('<i', vox.read(4))
                # clamp load_frame to total number of frames
                load_frame = min(load_frame, num_models)
            elif name == 'SIZE':
                # model size
                # x, y, z = struct.unpack('<3i', vox.read(12))
                vox.read(12)
            elif name == 'XYZI':
                # voxel data
                if current_frame == load_frame:
                    num_voxels, = struct.unpack('<i', vox.read(4))
                    for voxel in range(num_voxels):
                        voxel_data = struct.unpack('<4B', vox.read(4))
                        voxels.append(voxel_data)
                else:
                    print("Skipping voxels in frame #{}".format(current_frame))
                    vox.read(s_self)
                current_frame += 1
            elif name == 'RGBA':
                # palette
                for col in range(256):
                    palette.update({col + 1: struct.unpack('<4B', vox.read(4))})
            elif name == 'MATT':
                # material for old file format
                if use_materials:
                    material_spec = read_MATT(vox, material_specs)
                    print(material_spec)
                else:
                    vox.read(s_self)
            elif name == 'MATL':
                # material for new file format
                if use_materials:
                    material_spec = read_MATL(vox, material_specs)
                    print(material_spec)
                else:
                    vox.read(s_self)
            elif name == 'LAYR':
                # layer
                vox.read(s_self)
            elif name == 'nTRN':
                # trasform node
                vox.read(s_self)
            elif name == 'nGRP':
                # group node
                vox.read(s_self)
            elif name == 'nSHP':
                # shape node
                vox.read(s_self)
            elif name == 'rOBJ':
                # global settings
                vox.read(s_self)
            else:
                # Any other chunk, we don't know how to handle
                # This puts us out-of-step
                print('Unknown Chunk id {}'.format(name))
                return {'CANCELLED'}

    if use_bounds:
        # clamp end_voxel to size of model
        end = min([end_voxel, len(voxels)])
        voxels = voxels[start_voxel - 1:end]

    if not palette:  # no palette provided, use default
        for col in range(256):
            palette.update({col + 1: struct.unpack('<4B', struct.pack('<I', DEFAULT_PALETTE[col]))})

    if use_palette:
        used_palette_indices = set()
        for voxel in voxels:
            # This is done here, so to avoid adding materials for voxels not in bounds
            used_palette_indices.add(voxel[3])  # record the pallette entry is used

        if not gamma_correct:
            gamma_value = 1
        mat_palette = {}

        print("len(material_specs):", len(material_specs))
        print("len(used_palette_indices):", len(used_palette_indices))

        for index in used_palette_indices:
            palette_entry = palette[index]
            gamma_corrected = [pow(col / 255, gamma_value) for col in palette_entry[:3]]
            gamma_corrected.append(palette_entry[3])
            material = bpy.data.materials.new("Voxel_mat{}".format(index))
            material.diffuse_color = gamma_corrected

            # TODO: add specular, if materials being used
            # TODO: refactor so that we work on assumption of diffuse, but stay DRY
            if use_materials and index in material_specs:
                spec = material_specs[index]
                mat_type = spec.type_
                material.use_nodes = True
                nodes = material.node_tree.nodes

                # https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html
                # https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/glass.html
                # https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/emission.html
                # https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/glossy.html
                # https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/volume_absorption.html
                # https://piziadas.com/it/2014/04/volume-shaders-volume-absorption-blender-cycles.html

                if mat_type == '_diffuse':
                    # add Principled BSDF settings
                    node = nodes['Principled BSDF']
                    node.inputs['Specular'].default_value = spec.Specular

                elif mat_type == '_metal':
                    # add Principled BSDF settings
                    node = nodes['Principled BSDF']
                    node.inputs['Metallic'].default_value = spec.weight
                    node.inputs['Specular'].default_value = spec.Specular
                    node.inputs['Roughness'].default_value = spec.Roughness

                elif mat_type == '_glass':
                    # add Principled BSDF settings
                    node = nodes['Principled BSDF']
                    node.inputs['Transmission'].default_value = spec.weight
                    node.inputs['Specular'].default_value = spec.Specular
                    node.inputs['Roughness'].default_value = spec.Roughness
                    node.inputs['IOR'].default_value = spec.IOR
                    # add Volume Absorption shader to set Attenuation (if > 0.0)
                    if spec.Attenuation > 0.0:
                        # get specific node
                        node_output = nodes.get('Material Output')
                        # create volume absorption node
                        node_absorption = nodes.new(type='ShaderNodeVolumeAbsorption')
                        node_absorption.location = 10,450
                        node_absorption.inputs['Density'].default_value = spec.Attenuation
                        # link nodes
                        links = material.node_tree.links
                        link = links.new(node_absorption.outputs[0], node_output.inputs['Volume'])
                    # TODO: enable screen space refraction in material options
                    # TODO: enable screen space reflections in render options + refraction

                elif mat_type == '_emit':
                    # emissive
                    material_to_shader(material, spec)
                    # TODO: enable BLOOM in render options

                elif mat_type ==  '_plastic':
                    # glossy plastic
                    material_to_shader(material, spec)

                # TODO: add '_media' material for MATL (e.g. the cloud matter)
                #elif mat_type == '_media':

                else:
                    print("Unknown '{}' material type at index {}".format(mat_type, index))
            #else:
            #    material.diffuse_intensity = 1.0

            if use_shadeless:
                material.use_nodes = True
                material_to_shader(material, "_emit")
            mat_palette.update({index: material})

    # Using primitive_cube_add once here, to give us a template cube
    bpy.ops.mesh.primitive_cube_add(size=voxel_size)
    base_voxel = bpy.context.object

    # create new collection and link it to the scene
    name = os.path.basename(path)
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)

    print("Loaded voxels:", len(voxels))

    to_link = []

    for voxel in voxels:
        copy = base_voxel.copy()
        copy.data = base_voxel.data.copy()
        copy.location = [float(coord) * voxel_spacing for coord in voxel[:3]]
        to_link.append(copy)
        if use_palette:
            copy.active_material = mat_palette[voxel[3]]

    for object_ in to_link:
        collection.objects.link(object_)

    print("Linked voxels:", len(to_link))

    # Delete the template cube
    bpy.ops.object.delete({"selected_objects": [base_voxel]})

    layer = bpy.context.view_layer
    layer.update()

    print("\nSuccessfully imported {} in {:.3f} sec".format(path, time.time() - time_start))
    return {'FINISHED'}


#
# =================================================================================================
#     Change "default" shader to Emission shader without affecting its color
#     http://web.purplefrog.com/~thoth/blender/python-cookbook/change-diffuse-to-emission-node.html
# =================================================================================================
#
def replace_with_shader(node, node_tree, shader_type):
    new_node = node_tree.nodes.new(shader_type)
    connected_sockets_out = []
    sock = node.inputs[0]
    if len(sock.links) > 0:
        color_link = sock.links[0].from_socket
    else:
        color_link = None
    defaults_in = sock.default_value[:]

    for sock in node.outputs:
        if len(sock.links) > 0:
            connected_sockets_out.append( sock.links[0].to_socket)
        else:
            connected_sockets_out.append(None)

    new_node.location = (node.location.x, node.location.y)

    if color_link is not None:
        node_tree.links.new(new_node.inputs[0], color_link)
    new_node.inputs[0].default_value = defaults_in

    if connected_sockets_out[0] is not None:
        node_tree.links.new(connected_sockets_out[0], new_node.outputs[0])

    return new_node


def material_to_shader(mat, matter):

    doomed = []

    # accept both 'material_spec' or 'str' types
    if isinstance(matter, material_spec):
        shader_type = matter.type_
    elif isinstance(matter, str):
        shader_type = matter

    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            if shader_type == '_diffuse':
                new_node = replace_with_shader(node, mat.node_tree, 'ShaderNodeBsdfDiffuse')
                if isinstance(matter, material_spec):
                    new_node.inputs["Roughness"].default_value = matter.Roughness
            elif shader_type == '_glass':
                new_node = replace_with_shader(node, mat.node_tree, 'ShaderNodeBsdfGlass')
                if isinstance(matter, material_spec):
                    new_node.inputs["Roughness"].default_value = matter.Roughness
                    new_node.inputs["IOR"].default_value = matter.IOR
            elif shader_type == '_metal':
                new_node = replace_with_shader(node, mat.node_tree, 'ShaderNodeBsdfGlossy')
                if isinstance(matter, material_spec):
                    new_node.inputs["Roughness"].default_value = matter.Roughness
            elif shader_type == '_emit':
                new_node = replace_with_shader(node, mat.node_tree, 'ShaderNodeEmission')
                if isinstance(matter, material_spec):
                    new_node.inputs["Strength"].default_value = (matter.Power * 2.0) + 1.0
            elif shader_type == '_plastic':
                new_node = replace_with_shader(node, mat.node_tree, 'ShaderNodeBsdfGlossy')
                if isinstance(matter, material_spec):
                    new_node.inputs["Roughness"].default_value = matter.Roughness
            doomed.append(node)
        #else:
        #    print("Skipping node '%s':" % node.name, node.type)

    # wait until we are done iterating and adding before we start wrecking things
    for node in doomed:
        mat.node_tree.nodes.remove(node)


#
# =================================================================================================
#     Register - Unregister - MAIN
# =================================================================================================
#
classes = (
    ImportVOX,
)


def menu_func_import(self, context):
    self.layout.operator(ImportVOX.bl_idname, text="MagicaVoxel (.vox)")


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    register()

    # Test call
    bpy.ops.import_scene.vox('INVOKE_DEFAULT')
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
