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
    with open(filepath,'r') as f:
        if f.readline().strip() != 'OFF':
            raise TypeError('Not a valid OFF header')
        n_verts, n_faces, _ = [int(x) for x in f.readline().strip().split(' ')]
        vertices = [ [float(x) for x in f.readline().strip().split()] for _ in range(n_verts)]
        faces = [[int(x) for x in f.readline().strip().split()][1:] for _ in range(n_faces)]
    
    return np.asarray(vertices), np.asarray(faces)    

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
    vertices = [ [float(x) for x in line.strip().split()] for line in open(filepath,'r')]
    return np.asarray(vertices)

def read_tri(filepath, from_matlab = True):
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
    return np.asarray(faces) - int(from_matlab)