import os
from shutil import copyfile
import numpy as np


def read_off(filepath):
    """
    read a standard .off file

    Parameters
    -------------------------
    file : path to a '.off'-format file

    Output
    -------------------------
    vertices,faces : (n,3), (m,3) array of vertices coordinates
                    and indices for triangular faces
    """
    with open(filepath, 'r') as f:
        if f.readline().strip() != 'OFF':
            raise TypeError('Not a valid OFF header')
        n_verts, n_faces, _ = [int(x) for x in f.readline().strip().split(' ')]
        vertices = [[float(x) for x in f.readline().strip().split()] for _ in range(n_verts)]
        if n_faces > 0:
            faces = [[int(x) for x in f.readline().strip().split()][1:4] for _ in range(n_faces)]
            faces = np.asarray(faces)
        else:
            faces = None

    return np.asarray(vertices), faces


def read_obj(filepath):
    """
    read a standard .obj file

    Parameters
    -------------------------
    file : path to a '.off'-format file

    Output
    -------------------------
    vertices,faces : (n,3), (m,3) array of vertices coordinates
                    and indices for triangular faces
    """
    with open(filepath, 'r') as f:

        vertices = []
        faces = []

        for line in f:
            line = line.strip()
            if line == '' or line[0] == '#':
                continue

            line = line.split()
            if line[0] == 'v':
                vertices.append([float(x) for x in line[1:]])
            elif line[0] == 'f':
                faces.append([int(x.split('/')[0]) - 1 for x in line[1:]])

    return np.asarray(vertices), np.asarray(faces)


def write_off(filepath, vertices, faces, precision=None, face_colors=None):
    """
    Writes a .off file

    Parameters
    --------------------------
    filepath  : path to file to write
    vertices  : (n,3) array of vertices coordinates
    faces     : (m,3) array of indices of face vertices
    precision : int - number of significant digits to write for each float
    """
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0] if faces is not None else 0
    precision = precision if precision is not None else 16

    if face_colors is not None:
        assert face_colors.shape[0] == faces.shape[0], "PB"
        if face_colors.max() <= 1:
            face_colors = (256 * face_colors).astype(int)

    with open(filepath, 'w') as f:
        f.write('OFF\n')
        f.write(f'{n_vertices} {n_faces} 0\n')
        for i in range(n_vertices):
            f.write(f'{" ".join([f"{coord:.{precision}f}" for coord in vertices[i]])}\n')

        if n_faces != 0:
            for j in range(n_faces):
                if face_colors is None:
                    f.write(f'3 {" ".join([str(tri) for tri in faces[j]])}\n')
                else:
                    f.write(f'4 {" ".join([str(tri) for tri in faces[j]])} ')
                    f.write(f'{" ".join([str(tri_c) for tri_c in face_colors[j]])}\n')


def read_vert(filepath):
    """
    Read a .vert file from TOSCA dataset

    Parameters
    ----------------------
    filepath : path to file

    Output
    ----------------------
    vertices : (n,3) array of vertices coordinates
    """
    vertices = [[float(x) for x in line.strip().split()] for line in open(filepath, 'r')]
    return np.asarray(vertices)


def read_tri(filepath, from_matlab=True):
    """
    Read a .tri file from TOSCA dataset

    Parameters
    ----------------------
    filepath    : path to file
    from_matlab : whether file indexing starts at 1

    Output
    ----------------------
    faces : (m,3) array of vertices indices to define faces
    """
    faces = [[int(x) for x in line.strip().split()] for line in open(filepath,'r')]
    faces = np.asarray(faces)
    if from_matlab and np.min(faces) > 0:
        raise ValueError("Indexing starts at 0, can't set the from_matlab argument to True ")
    return faces - int(from_matlab)


def write_mtl(filepath, texture_im='texture_1.jpg'):
    """
    Writes a .mtl file for a .obj mesh

    Parameters
    ----------------------
    filepath   : path to file
    texture_im : name of the image of texture
    """
    with open(filepath, 'w') as f:
        f.write('newmtl material_0\n')
        f.write(f'Ka  {0.2:.6f} {0.2:.6f} {0.2:.6f}\n')
        f.write(f'Kd  {1.:.6f} {1.:.6f} {1.:.6f}\n')
        f.write(f'Ks  {1.:.6f} {1.:.6f} {1.:.6f}\n')
        f.write(f'Tr  {1:d}\n')
        f.write(f'Ns  {0:d}\n')
        f.write(f'illum {2:d}\n')
        f.write(f'map_Kd {texture_im}')


def _get_data_dir():
    """
    Return the directory where texture data is savec

    Output
    ---------------------
    data_dir : str - directory of texture data
    """
    curr_dir = os.path.dirname(__file__)
    return os.path.join(curr_dir,'data')


def get_uv(vertices, ind1, ind2, mult_const=1):
    """
    Extracts UV coordinates for a mesh for a .obj file

    Parameters
    -----------------------------
    vertices   : (n,3) coordinates of vertices
    ind1       : int - column index to use as first coordinate
    ind2       : int - column index to use as second coordinate
    mult_const : float - number of time to repeat the pattern

    Output
    ------------------------------
    uv : (n,2) UV coordinates of each vertex
    """
    vt = vertices[:,[ind1,ind2]]
    vt -= np.min(vt)
    vt = mult_const * vt / np.max(vt)
    return vt


def write_obj(filepath, vertices, faces, uv=None, mtl_file='material.mtl', texture_im='texture_1.jpg',
              precision=6, verbose=False):
    """
    Writes a .obj file with texture.
    Writes the necessary material and texture files.

    Parameters
    -------------------------
    filepath   : str - path to the .obj file to write
    vertices   : (n,3) coordinates of vertices
    faces      : (m,3) faces defined by vertex indices
    uv         : uv map for each vertex. If not specified no texture is used
    mtl_file   : str - name of the .mtl file
    texture_im : str - name of the .jpg file definig texture
    """
    use_texture = uv is not None
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0]
    precision = 16 if precision is None else precision

    dir_name = os.path.dirname(filepath)

    if use_texture:
        # Remove useless part of the path if written by mistake
        mtl_file = os.path.basename(mtl_file)
        texture_file = os.path.basename(texture_im)

        # Add extensions if forgotten
        if os.path.splitext(mtl_file)[1] != '.mtl':
            mtl_file += '.mtl'
        if os.path.splitext(texture_file)[1] != '.jpg':
            texture_file += '.jpg'

        # Write .mtl and .jpg files if necessary
        mtl_path = os.path.join(dir_name, mtl_file)
        texture_path = os.path.join(dir_name, texture_file)

        if not os.path.isfile(texture_path):
            data_texture = os.path.join(_get_data_dir(), texture_im)
            if not os.path.isfile(data_texture):
                raise ValueError(f"Texture {texture_im} does not exist")
            copyfile(data_texture, texture_path)
            print(f'Copy texture at {texture_path}')

        if not os.path.isfile(mtl_path):
            write_mtl(mtl_path,texture_im=texture_im)
            if verbose:
                print(f'Write material at {mtl_path}')

        # Write the .obj file
        mtl_name = os.path.splitext(mtl_file)[0]

    with open(filepath,'w') as f:
        if use_texture:
            f.write(f'mtllib ./{mtl_name}.mtl\ng\n')

        f.write(f'# {n_vertices} vertices - {n_faces} faces\n')
        for i in range(n_vertices):
            f.write(f'v {" ".join([f"{coord:.{precision}f}" for coord in vertices[i]])}\n')

        if use_texture and n_faces > 0:
            f.write(f'g {mtl_name}_export\n')
            f.write('usemtl material_0\n')

            for j in range(n_faces):
                f.write(f'f {" ".join([f"{1+tri:d}/{1+tri:d}" for tri in faces[j]])}\n')

            for k in range(n_vertices):
                f.write(f'vt {" ".join([str(coord) for coord in uv[k]])}\n')

        elif n_faces > 0:
            for j in range(n_faces):
                f.write(f'{" ".join(["f"] + [str(1+tri) for tri in faces[j]])}\n')

    if verbose:
        print(f'Write .obj file at {filepath}')
