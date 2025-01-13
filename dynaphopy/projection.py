import numpy as np
import numba   # numba is imported to accelerate projection

# parallel enables the numba.prange directive
@numba.njit(fastmath=True,parallel=True, nogil=True,cache=True)
def project_wv(number_of_atoms, number_of_dimensions, velocity_projected_temp, velocity, q_vector, coordinates,unique_atom_type,atom_type,velocity_projected):
    """project atomic velocity onto wavevector

    The parallelization is over atoms.

    Parameters
    ----------
    number_of_atoms
        number of atoms
    number_of_dimensions
        dimensions
    velocity_projected
        the projected atomic velocity as a float array with shape (``Nx``, 3)
    velocity
        the projected atomic velocity as a float array with shape (``Nx``, 3)
    q_vector
        the wave vector which is projected
    coordinates
        atomic coordinates

    """

    #Projection into wave vector

    for i in numba.prange(number_of_atoms):
        # Projection on atom
        #if project_on_atom > -1:
        #    if atom_type[i] != project_on_atom:
        #        continue

        exp_factor = np.exp(-1j*(q_vector[0]*coordinates[i,0] + q_vector[1]*coordinates[i,1] +q_vector[2]*coordinates[i,2]))            # np.dot(q_vector, coordinates[i,:]))

        for k in range(number_of_dimensions):
            velocity_projected_temp[:, i, k] = velocity[:,i,k]*exp_factor

    N_atom_type = unique_atom_type.shape[0]

    #for i in range(len(atom_type)):
    #    velocity_projected[:,atom_type[i],:] += velocity_projected_temp[:,i,:]

    for i in numba.prange(N_atom_type):
        temp_id_pri = unique_atom_type[i]
        index = np.where(atom_type==temp_id_pri)[0]
        #velocity_projected[:,temp_id_pri,:] = np.sum(velocity_projected_temp[:,index,:], axis = 1)

        for j in range(len(index)):
            velocity_projected[:,temp_id_pri,:] += velocity_projected_temp[:,index[j],:]

# parallel enables the numba.prange directive
@numba.njit(parallel=True, nogil=True,fastmath=True,cache=True)
def project_phonon(number_of_frequencies, number_of_cell_atoms, vc, eigenvectors,velocity_projected_temp):
    """project atomic velocity onto wavevector

    The parallelization is over atoms.

    Parameters
    ----------
    number_of_frequencies
        number of frequencies
    number_of_cell_atoms
        number of cell atoms
    velocity_projected
        the projected atomic velocity onto phonon as a float array with shape (``Nx``, 3)
    vc
        the projected atomic velocity onto wavevector as a float array with shape (``Nx``, 3)
    eigenvectors
        the eigenvectors which will be projected

    """

    #Projection into wave vector
        #Projection into wave vector
    eigenvectors_conj = eigenvectors.conj()

    for k in numba.prange(number_of_frequencies):
        for i in range(number_of_cell_atoms):
            #velocity_projected_temp[:, k,i] = np.dot(vc[:, i, :], eigenvectors[k, i, :].conj())
            velocity_projected_temp[:, k,i] =  vc[:, i, 0]*eigenvectors_conj[k, i, 0] + vc[:, i, 1]*eigenvectors_conj[k, i, 1] +vc[:, i, 2]*eigenvectors_conj[k, i, 2]  #  np.dot(vc[:, i, :], eigenvectors[k, i, :].conj())
            #velocity_projected_temp[:, k,i] = sum(vc[:, i, d] * eigenvectors_conj[k, i, d] for d in range(3))

    #for k in numba.prange(number_of_frequencies):
    #    for i in numba.prange(number_of_cell_atoms):
    #        velocity_projected[:, k] += np.dot(vc[:, i, :], eigenvectors[k, i, :].conj())


# parallel enables the numba.prange directive
@numba.njit(parallel=True, nogil=True,fastmath=True,cache=True)
def project_phonon_amorphous(number_of_cell_atoms, vc, eigenvectors,velocity_projected_temp):
    """project atomic velocity onto wavevector

    The parallelization is over atoms.

    Parameters
    ----------
    number_of_frequencies
        number of frequencies
    number_of_cell_atoms
        number of cell atoms
    velocity_projected
        the projected atomic velocity onto phonon as a float array with shape (``Nx``, 3)
    vc
        the projected atomic velocity onto wavevector as a float array with shape (``Nx``, 3)
    eigenvectors
        the eigenvectors which will be projected

    """

    #Projection into wave vector
        #Projection into wave vector
    eigenvectors_conj = eigenvectors.conj()


    for i in numba.prange(number_of_cell_atoms):
        #velocity_projected_temp[:, k,i] = np.dot(vc[:, i, :], eigenvectors[k, i, :].conj())
        velocity_projected_temp[:, i] =  vc[:, i, 0]*eigenvectors_conj[i, 0] + vc[:, i, 1]*eigenvectors_conj[i, 1] +vc[:, i, 2]*eigenvectors_conj[i, 2]  #  np.dot(vc[:, i, :], eigenvectors[k, i, :].conj())



def project_onto_wave_vector(trajectory, q_vector, project_on_atom=-1):

    number_of_primitive_atoms = np.int16(trajectory.structure.get_number_of_primitive_atoms())
    velocity = trajectory.get_velocity_mass_average()
#    velocity = trajectory.velocity   # (use the velocity without mass average, just for testing)

    number_of_atoms = np.int32(velocity.shape[1])
    number_of_dimensions = np.int32(velocity.shape[2])
    supercell = trajectory.get_supercell_matrix()

    coordinates = trajectory.structure.get_positions(supercell)
    atom_type = np.array(trajectory.structure.get_atom_type_index(supercell=supercell))

    velocity_projected = np.zeros((velocity.shape[0], number_of_primitive_atoms, number_of_dimensions), dtype=complex)

    project_on_atom = np.int8(project_on_atom)

    if q_vector.shape[0] != coordinates.shape[1]:
        print("Warning!! Q-vector and coordinates dimension do not match")
        exit()

    #print("data type used in project_onto_wave_vector:")
    #print("number_of_atoms:" + str(type(number_of_atoms)))
    #print("number_of_atoms:" + str(number_of_atoms.dtype))
    #np.save("number_of_atoms.npy",number_of_atoms)

    #print("number_of_dimensions:" + str(type(number_of_dimensions)))
    #print("number_of_dimensions:" + str(number_of_dimensions.dtype))
    #np.save("number_of_dimensions.npy",number_of_dimensions)

    #print("velocity_projected:" + str(type(velocity_projected)))
    #print("velocity_projected:" + str(velocity_projected.dtype))
    #np.save("velocity_projected.npy",velocity_projected)

    #print("velocity:" + str(type(velocity)))
    #print("velocity:" + str(velocity.dtype))
    #np.save("velocity.npy",velocity)

    #print("q_vector:" + str(type(q_vector)))
    #print("q_vector:" + str(q_vector.dtype))
    #np.save("q_vector.npy",q_vector)

    #print("coordinates:" + str(type(coordinates)))
    #print("coordinates:" + str(coordinates.dtype))
    #np.save("coordinates.npy",coordinates)

    #print("atom_type:" + str(type(atom_type)))
    #print("atom_type:" + str(atom_type.dtype))
    #np.save("atom_type.npy",atom_type)

    #print("project_on_atom:" + str(type(project_on_atom)))
    #print("project_on_atom:" + str(project_on_atom.dtype))
    #np.save("project_on_atom.npy",project_on_atom)

    #Projection into wave vector
    velocity_projected_temp = np.zeros_like(velocity,dtype=complex)
    unique_atom_type = np.unique(atom_type)
    project_wv(number_of_atoms, number_of_dimensions, velocity_projected_temp, velocity, q_vector, coordinates,unique_atom_type,atom_type,velocity_projected)

    del velocity_projected_temp
    #project_wv(number_of_atoms, number_of_dimensions, velocity_projected, velocity, q_vector, coordinates, atom_type,project_on_atom)
    #for i in range(number_of_atoms):
        # Projection on atom
    #    if project_on_atom > -1:
    #        if atom_type[i] != project_on_atom:
    #            continue

    #    for k in range(number_of_dimensions):
    #        velocity_projected[:, atom_type[i], k] += velocity[:,i,k]*np.exp(-1j*np.dot(q_vector, coordinates[i,:]))

   #Normalize velocities (method 1)
  #  for i in range(velocity_projected.shape[1]):
  #      velocity_projected[:,i,:] /= atom_type.count(i)

   #Normalize velocities (method 2)
    number_of_primitive_cells = number_of_atoms/number_of_primitive_atoms
    velocity_projected /= np.sqrt(number_of_primitive_cells)
    return velocity_projected


def project_onto_phonon(vc, eigenvectors):

    number_of_cell_atoms = np.int32(vc.shape[1])
    number_of_frequencies = np.int32(eigenvectors.shape[0])

    #Projection in phonon coordinate
    #velocity_projected=np.zeros((vc.shape[0], number_of_frequencies), dtype=complex)
    velocity_projected_temp=np.zeros((vc.shape[0], number_of_frequencies,number_of_cell_atoms), dtype=complex)

    #print("data type used in project_onto_phonon:")
    #print("number_of_frequencies:" + str(type(number_of_frequencies)))
    #print("number_of_frequencies:" + str(number_of_frequencies.dtype))
    #np.save("number_of_frequencies.npy",number_of_frequencies)

    #print("number_of_cell_atoms:" + str(type(number_of_cell_atoms)))
    #print("number_of_cell_atoms:" + str(number_of_cell_atoms.dtype))
    #np.save("number_of_cell_atoms.npy",number_of_cell_atoms)

    #print("velocity_projected:" + str(type(velocity_projected)))
    #print("velocity_projected:" + str(velocity_projected.dtype))
    #np.save("velocity_projected_phonon.npy",velocity_projected)

    #print("vc:" + str(type(vc)))
    #print("vc:" + str(vc.dtype))
    #np.save("vc.npy",vc)

    #print("eigenvectors:" + str(type(eigenvectors)))
    #print("eigenvectors:" + str(eigenvectors.dtype))
    #np.save("eigenvectors.npy",eigenvectors)
    project_phonon(number_of_frequencies, number_of_cell_atoms, vc, eigenvectors,velocity_projected_temp)
    #project_phonon(number_of_frequencies, number_of_cell_atoms, vc, eigenvectors,velocity_projected)

    velocity_projected = np.sum(velocity_projected_temp, axis = 2)

    del velocity_projected_temp

    #for k in range(number_of_frequencies):
    #    for i in range(number_of_cell_atoms):
    #        velocity_projected[:, k] += np.dot(vc[:, i, :], eigenvectors[k, i, :].conj())

    return velocity_projected


def project_onto_phonon_amorphous(vc, eigenvectors):

    number_of_cell_atoms = np.int32(vc.shape[1])
    number_of_frequencies = np.int32(eigenvectors.shape[0])

    #Projection in phonon coordinate
    #velocity_projected=np.zeros((vc.shape[0], number_of_frequencies), dtype=complex)
    velocity_projected_temp=np.zeros((vc.shape[0], number_of_cell_atoms), dtype=complex)

    velocity_projected=np.zeros((vc.shape[0], number_of_frequencies), dtype=complex)

    #print("eigenvectors:" + str(type(eigenvectors)))
    #print("eigenvectors:" + str(eigenvectors.dtype))
    #np.save("eigenvectors.npy",eigenvectors)
    for i in range(number_of_frequencies):
        temp_eigenvector = eigenvectors[i,:,:]
        project_phonon_amorphous(number_of_cell_atoms, vc, temp_eigenvector, velocity_projected_temp)
        velocity_projected[:,i] = np.sum(velocity_projected_temp, axis = 1)
        velocity_projected_temp=np.zeros((vc.shape[0], number_of_cell_atoms), dtype=complex)
    #project_phonon(number_of_frequencies, number_of_cell_atoms, vc, eigenvectors,velocity_projected)

    del velocity_projected_temp

    #for k in range(number_of_frequencies):
    #    for i in range(number_of_cell_atoms):
    #        velocity_projected[:, k] += np.dot(vc[:, i, :], eigenvectors[k, i, :].conj())

    return velocity_projected


#Just for testing (slower implementation) [but equivalent]
def project_onto_phonon2(vc,eigenvectors):

    number_of_cell_atoms = vc.shape[1]
    number_of_frequencies = eigenvectors.shape[0]

    #Projection in phonon coordinate
    velocity_projected=np.zeros((vc.shape[0],number_of_frequencies),dtype=complex)

    for i in range (vc.shape[0]):
        for k in range(number_of_frequencies):
            velocity_projected[i,k] = np.trace(np.dot(vc[i,:,:], eigenvectors[k,:,:].T.conj()))
#            velocity_projected[i,k] = np.sum(np.linalg.eigvals(np.dot(vc[i,:,:],eigenvectors[k,:,:].T.conj())))
    return velocity_projected




"""
def project_onto_wave_vector(trajectory, q_vector, project_on_atom=-1):

    number_of_primitive_atoms = trajectory.structure.get_number_of_primitive_atoms()
    velocity = trajectory.get_velocity_mass_average()
#    velocity = trajectory.velocity   # (use the velocity without mass average, just for testing)

    number_of_atoms = velocity.shape[1]
    number_of_dimensions = velocity.shape[2]
    supercell = trajectory.get_supercell_matrix()

    coordinates = trajectory.structure.get_positions(supercell)
    atom_type = trajectory.structure.get_atom_type_index(supercell=supercell)

    velocity_projected = np.zeros((velocity.shape[0], number_of_primitive_atoms, number_of_dimensions), dtype=complex)

    if q_vector.shape[0] != coordinates.shape[1]:
        print("Warning!! Q-vector and coordinates dimension do not match")
        exit()

    #Projection into wave vector
    for i in range(number_of_atoms):
        # Projection on atom
        if project_on_atom > -1:
            if atom_type[i] != project_on_atom:
                continue

        for k in range(number_of_dimensions):
            velocity_projected[:, atom_type[i], k] += velocity[:,i,k]*np.exp(-1j*np.dot(q_vector, coordinates[i,:]))

   #Normalize velocities (method 1)
  #  for i in range(velocity_projected.shape[1]):
  #      velocity_projected[:,i,:] /= atom_type.count(i)

   #Normalize velocities (method 2)
    number_of_primitive_cells = number_of_atoms/number_of_primitive_atoms
    velocity_projected /= np.sqrt(number_of_primitive_cells)
    return velocity_projected


def project_onto_phonon(vc, eigenvectors):

    number_of_cell_atoms = vc.shape[1]
    number_of_frequencies = eigenvectors.shape[0]

    #Projection in phonon coordinate
    velocity_projected=np.zeros((vc.shape[0], number_of_frequencies), dtype=complex)
    for k in range(number_of_frequencies):
        for i in range(number_of_cell_atoms):
            velocity_projected[:, k] += np.dot(vc[:, i, :], eigenvectors[k, i, :].conj())

    return velocity_projected


#Just for testing (slower implementation) [but equivalent]
def project_onto_phonon2(vc,eigenvectors):

    number_of_cell_atoms = vc.shape[1]
    number_of_frequencies = eigenvectors.shape[0]

    #Projection in phonon coordinate
    velocity_projected=np.zeros((vc.shape[0],number_of_frequencies),dtype=complex)

    for i in range (vc.shape[0]):
        for k in range(number_of_frequencies):
            velocity_projected[i,k] = np.trace(np.dot(vc[i,:,:], eigenvectors[k,:,:].T.conj()))
#            velocity_projected[i,k] = np.sum(np.linalg.eigvals(np.dot(vc[i,:,:],eigenvectors[k,:,:].T.conj())))
    return velocity_projected
"""
