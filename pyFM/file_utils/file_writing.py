import numpy as np
def write_off(filepath, vertices, faces):
    """
    Writes a .off file

    Parameters
    --------------------------
    filepath : path to file to write
    vertices : (n,3) array of vertices coordinates
    faces    : (m,3) array of indices of face vertices
    """
    n_vertices = vertices.shape[0]
    n_faces = faces.shape[0]

    with open(filepath,'w') as f:
        f.write('OFF\n')
        f.write(f'{n_vertices} {n_faces} 0\n')
        for i in range(n_vertices):
            f.write(f'{" ".join([str(coord) for coord in vertices[i]])}\n')
        
        for j in range(n_faces):
            f.write(f'3 {" ".join([str(tri) for tri in faces[j]])}\n')