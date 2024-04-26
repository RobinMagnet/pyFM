import os
from shutil import copyfile
import numpy as np


def read_off(filepath, read_colors=False):
    """
    read a standard .off file

    Read a .off file containing vertex and face information, and possibly face colors.

    Parameters
    ----------
    file : str
        path to a '.off'-format file
    read_colors : bool, optional
        bool - whether to read colors if present

    Returns
    -------
    vertices : np.ndarray
        (n,3) array of vertices coordinates
    faces : np.ndarray
        (m,3) array of indices of face vertices
    colors : np.ndarray, optional
        (m,3) Only if read_colors is True. Array of colors for each face. None if not found.
    """
    with open(filepath, 'r') as f:
        if f.readline().strip() != 'OFF':
            raise TypeError('Not a valid OFF header')

        header_line = f.readline().strip().split(' ')
        while header_line[0].startswith('#'):
            header_line = f.readline().strip().split(' ')
        n_verts, n_faces, _ = [int(x) for x in header_line]

        vertices = [[float(x) for x in f.readline().strip().split()[:3]] for _ in range(n_verts)]

        if n_faces > 0:
            face_elements = [ [int(x) for x in f.readline().strip().split()[1:] if not x.startswith('#')] for _ in range(n_faces)]
            face_elements = np.asarray(face_elements)
            faces = face_elements[:, :3]
            if read_colors:
                colors = face_elements[:, 3:6] if face_elements.shape[1] == 6 else None
        else:
            faces = None

    if read_colors:
        return np.asarray(vertices), faces, colors

    return np.asarray(vertices), faces


def read_obj(filepath, load_normals=False, load_texture=False, load_texture_normals=False):
    """
    Read a .obj file containing a mesh.

    Parameters
    -------------------------
    filepath : str
        path to the .obj file
    load_normals : bool, optional
        whether to load vertex normals if present
    load_texture : bool, optional
        whether to load texture coordinates if present. Reads both uv coordinates and face to texture vertex indices
    load_texture_normals : bool, optional
        whether to load texture normals if present. Reads face to texture normal indices

    Returns
    -------
    vertices : np.ndarray
        (n,3) array of vertices coordinates
    faces    : np.ndarray
        (m,3) array of indices of face vertices, None if not present
    normals  : np.ndarray, optional
        Only if load_normals is True. (n,3) array of vertex normals, None if not present
    uv       : np.ndarray, optional
        Only if load_texture is True (n,2) array of uv coordinates, None if not present
    fvt      : np.ndarray, optional
        Only if load_texture is True (m,3) array of indices of face to vertex texture (vt) indices, None if not present
    fnt      : np.ndarray, optional
        Only if load_texture_normals is True (m,3) array of indices of face to texture normal indices, None if not present

    """

    with open(filepath, 'r') as f:
        vertices = []
        faces = []
        normals = []

        uv = []
        fvt = []
        fnt = []

        for line in f:
            line = line.strip()
            if line == '' or line[0].startswith('#'):
                continue

            line = line.split()
            if line[0] == 'v':
                vertices.append([float(x) for x in line[1:4]])

            elif load_texture and line[0] == 'vt':
                uv.append([float(x) for x in line[1:3]] if len(line)>=3 else [float(line[0]), 0])

            elif line[0] == 'f':
                faces.append([int(x.split('/')[0]) - 1 for x in line[1:]])

                if load_texture and line[1].count('/') > 0:
                    if line[1].split('/')[1] != '':
                        fvt.append([int(x.split('/')[1]) - 1 for x in line[1:]])

                if load_normals and line[1].count('/') == 2:
                    if line[1].split('/')[2] != '':
                        fnt.append([int(x.split('/')[2]) - 1 for x in line[1:]])

            elif load_normals and line[0] == 'vn':
                normals.append([float(x) for x in line[1:]])

    vertices = np.asarray(vertices)
    faces = np.asarray(faces) if len(faces) > 0 else None
    normals = np.asarray(normals) if len(normals) > 0 else None
    uv = np.asarray(uv) if len(uv) > 0 else None
    fvt = np.asarray(fvt) if len(fvt) > 0 else None
    fnt = np.asarray(fnt) if len(fnt) > 0 else None

    output = [vertices, faces]
    if load_normals:
        output.append(normals)

    if load_texture:
        output.append(uv)
        output.append(fvt)

    if load_texture_normals:
        output.append(fnt)

    return output


def write_off(filepath, vertices, faces, precision=None, face_colors=None):
    """
    Writes a mesh to a .off file

    The number of significant digit to use can be specified for memory saving.

    Parameters
    --------------------------
    filepath: str
        path to the .off file to write
    vertices: np.ndarray
        (n,3) array of vertices coordinates
    faces: np.ndarray
        (m,3) array of indices of face vertices
    precision: int, optional
        number of significant digits to write for each float. Defaults to 16
    face_colors: np.ndarray, optional
        (m,3) array of colors for each face
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


def write_obj(filepath, vertices, faces=None, uv=None, fvt=None, fnt=None, vertex_normals=None, mtl_path=None, mtl_name="material_0", precision=None):
    """
    Writes a .obj file with texture.

    If material is used, the .mtl file will be copied to the same directory as the .obj file.

    Parameters
    -------------------------
    filepath : str
        path to the .obj file to write
    vertices : np.ndarray
        (n,3) array of vertices coordinates
    faces : np.ndarray, optional
        (m,3) array of indices of face vertices
    uv : np.ndarray, optional
        (n,2) array of uv coordinates
    fvt : np.ndarray, optional
        (m,3) array of indices of face to vertex texture indices
    fnt : np.ndarray, optional
        (m,3) array of indices of face to texture normal indices
    vertex_normals : np.ndarray, optional
        (n,3) array of vertex normals
    mtl_path : str, optional
        path to the .mtl file defining the material
    mtl_name : str, optional
        name of the material in the .mtl file
    precision : int, optional
        number of significant digits to write for each float
    """

    n_vertices = len(vertices)
    n_faces = len(faces) if faces is not None else 0
    n_vt = len(uv) if uv is not None else 0
    precision = precision if precision is not None else 16

    if (mtl_path is not None) and (uv is not None) and (fvt is None):
        print('WARNING: Material and uv provided, but no face texture index')

    if mtl_path is not None and n_faces == 0:
        print('WARNING: Material provided, but no face. Ignoring material.')

    with open(filepath,'w') as f:
        if n_faces > 0 and mtl_path is not None:
            mtl_filename = os.path.splitext(os.path.basename(mtl_path))[0]
            f.write(f'mtllib {mtl_path}\ng\n')

        f.write(f'# {n_vertices} vertices - {n_faces} faces - {n_vt} vertex textures\n')

        for i in range(n_vertices):
            f.write(f'v {" ".join([f"{coord:.{precision}f}" for coord in vertices[i]])}\n')

        if vertex_normals is not None:
            for i in range(len(vertex_normals)):
                f.write(f'vn {" ".join([f"{coord:.{precision}f}" for coord in vertex_normals[i]])}\n')

        if uv is not None:
            for i in range(len(uv)):
                f.write(f'vt {" ".join([f"{coord:.{precision}f}" for coord in uv[i]])}\n')

        if n_faces > 0:
            if mtl_path is not None:
                f.write(f'g {mtl_filename}_export\n')
                f.write(f'usemtl {mtl_name}\n')

            for j in range(n_faces):
                if fvt is not None and fnt is not None:
                    f.write(f'f {" ".join([f"{1+faces[j][k]:d}/{1+fvt[j][k]:d}/{1+fnt[j][k]:d}" for k in range(3)])}\n')

                elif fvt is not None:
                    f.write(f'f {" ".join([f"{1+faces[j][k]:d}/{1+fvt[j][k]:d}" for k in range(3)])}\n')

                elif fnt is not None:
                    f.write(f'f {" ".join([f"{1+faces[j][k]:d}//{1+fnt[j][k]:d}" for k in range(3)])}\n')

                else:
                    f.write(f'f {" ".join([str(1+tri) for tri in faces[j]])}\n')


def read_vert(filepath):
    """
    Read a .vert file from TOSCA dataset

    Parameters
    ----------------------
    filepath : str
        path to file

    Returns
    ----------------------
        vertices : np.ndarray
            (n,3) array of vertices coordinates
    """
    vertices = [[float(x) for x in line.strip().split()] for line in open(filepath, 'r')]
    return np.asarray(vertices)


def read_tri(filepath, from_matlab=True):
    """
    Read a .tri file from TOSCA dataset

    Parameters
    ----------------------
    filepath    : str
        path to file
    from_matlab : bool, optional
        If True, file indexing starts at 1

    Returns
    ----------------------
    faces : np.ndarray
        (m,3) array of vertices indices to define faces
    """
    faces = [[int(x) for x in line.strip().split()] for line in open(filepath,'r')]
    faces = np.asarray(faces)
    if from_matlab and np.min(faces) > 0:
        raise ValueError("Indexing starts at 0, can't set the from_matlab argument to True ")
    return faces - int(from_matlab)


def write_mtl(filepath, texture_im='texture_1.jpg'):
    """
    Writes a .mtl file for a .obj mesh.

    Use the name of a texture image to define the material.

    Parameters
    ----------------------
    filepath   : str
        path to file
    texture_im : str, optional
        name of the image of texture. Default to 'texture_1.jpg', included in the package
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
    Return the directory where texture data is saved.

    Looks in the package directory.

    Returns
    ---------------------
    data_dir : str
        directory of texture data
    """
    curr_dir = os.path.dirname(__file__)
    return os.path.join(curr_dir,'data')


def get_uv(vertices, ind1, ind2, mult_const=1):
    """Extracts UV coordinates for a mesh for a .obj file

    Parameters
    ----------
    vertices   :
        (n,3) coordinates of vertices
    ind1       : int
        column index to use as first coordinate
    ind2       : int
        column index to use as second coordinate
    mult_const : float
        number of time to repeat the pattern

    Returns
    -------
    uv : float
        (n,2) UV coordinates of each vertex

    """



    vt = vertices[:,[ind1,ind2]]
    vt -= np.min(vt)
    vt = mult_const * vt / np.max(vt)
    return vt


def write_obj_texture(filepath, vertices, faces, uv=None, mtl_file='material.mtl', texture_im='texture_1.jpg',
                      mtl_name=None, precision=6, vertex_normals=None, verbose=False):
    """
    Writes a .obj file with texture, with a simpler interface than `write_obj`.

    This function writes mtl files and copy textures if necessary

    Parameters
    -------------------------
    filepath   : str
        path to the .obj file to write
    vertices   : np.ndarray
        (n,3) coordinates of vertices
    faces      : np.ndarray
        (m,3) faces defined by vertex indices
    uv: np.ndarray, optional
        (n,2) UV coordinates of each vertex
    mtl_file   : str, optional
        name or path of the .mtl file. If just a name, a default material will be created.
    texture_im : str, optional
        name or path of the .jpg file defining texture
    mtl_name   : str, optional
        name of the material in the .mtl file
    precision  : int, optional
        number of significant digits to write for each float
    vertex_normals : np.ndarray, optional
        (n,3) array of vertex normals
    verbose    : bool, optional
        whether to print information

    """
    assert filepath.endswith('.obj'), f"Filepath must end with .obj. Current filepath: {filepath}"

    use_texture = uv is not None
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0] if faces is not None else 0
    precision = 16 if precision is None else precision

    out_dir_name = os.path.abspath(os.path.dirname(filepath))

    if use_texture:
        # if texture_im = /path/to/texture.jpg
        if os.path.isfile(texture_im):
            texture_name = os.path.basename(texture_im)  # texture.jpg
            texture_abspath = os.path.abspath(os.path.join(out_dir_name,texture_name))  # /outdir/texture.jpg
            texture_relpath = os.path.join('./', texture_name)  # ./texture.jpg
            if not os.path.isfile(texture_abspath):
                copyfile(texture_im, texture_abspath)

        else:
            # texture_im is texture.jpg or just texture
            if os.path.splitext(texture_im)[1] != '.jpg':
                texture_im = texture_im + '.jpg'

            texture_im  = os.path.join(os.path.dirname(__file__), 'data', texture_im)
            texture_name = os.path.basename(texture_im)

            texture_abspath = os.path.abspath(os.path.join(out_dir_name,texture_name))
            texture_relpath = os.path.join('./', texture_name)
            if not os.path.isfile(texture_abspath):
                copyfile(texture_im, texture_abspath)
                if verbose:
                    print(f'Copy texure at {texture_abspath}')


        if os.path.isfile(mtl_file):
            # mtl_file = /path/to/material.mtl
            mtl_name = os.path.basename(mtl_file)  # material.mtl
            mtl_abspath = os.path.abspath(os.path.join(out_dir_name,mtl_name))  # /outdir/material.mtl
            mtl_relpath = os.path.join('./', mtl_name)  # ./material.mtl
            if not os.path.isfile(mtl_abspath):
                copyfile(mtl_file, mtl_abspath)
                if verbose:
                    print(f'Copy material at {mtl_abspath}')

        else:
            # mtl_file is material.mtl or just material
            if os.path.splitext(mtl_file)[1] != '.mtl':
                mtl_file = mtl_file + '.mtl'

            mtl_abspath = os.path.abspath(os.path.join(out_dir_name,mtl_file))  # /outdir/material.mtl
            mtl_relpath = os.path.join('./', os.path.basename(mtl_file))  # ./material.mtl
            if os.path.isfile(mtl_abspath):
                os.remove(mtl_abspath)
            write_mtl(mtl_abspath, texture_im=texture_relpath)
            mtl_name = 'material_0'
            if verbose:
                print(f'Write material at {mtl_abspath}')

            if mtl_name is None:
                mtl_name = 'material_0'

    else:
        mtl_relpath = None
        texture_relpath = None

    write_obj(filepath, vertices=vertices, faces=faces, uv=uv, fvt=faces, mtl_path=mtl_relpath,
              mtl_name=mtl_name, precision=precision, vertex_normals=vertex_normals)

    if verbose:
        print(f'Write .obj file at {filepath}')
