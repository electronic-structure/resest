import numpy as np
import math
import upf_to_json as upf

def get_atom_type_info(fname):
    """
    Get description of atom type from UPF file.

    Parameters:
      fname - UPF file name

    Return:
      Dictionary with atom type description.

    """
    jd = upf.parse_upf_from_file(fname)

    result = {}

    result["z"] = jd["pseudo_potential"]["header"]["z_valence"]

    nbeta = 0
    for beta in jd["pseudo_potential"]["beta_projectors"]:
        l = beta["angular_momentum"]
        nbeta += (2 * l + 1)

    result["nbeta"] = nbeta

    if "augmentation" in jd["pseudo_potential"]:
        result["augment"] = True
    else:
        result["augment"] = False

    return result


def get_fft_grid_size(b1, b2, b3, cutoff):
    """
    Estimate size of the FFT grid by three reciprocal lattice vectors and a cutoff.
    Vectors and cutoff are provided in the reciprocal atomic units of length.

    Parameters:
      b1 - first reciprocal lattice vector (in a.u.^-1)
      b2 - second reciprocal lattice vector (in a.u.^-1)
      b3 - third reciprocal lattice vector (in a.u.^-1)
      cutoff - plane-wave cutoff (in a.u.^-1)

    Return:
      Array of FFT grid dimensions.
    """
    det = abs(np.linalg.det([b1, b2, b3]))
    l1 = int(2 * cutoff * np.linalg.norm(np.cross(b2, b3)) / det) + 1
    l2 = int(2 * cutoff * np.linalg.norm(np.cross(b3, b1)) / det) + 1
    l3 = int(2 * cutoff * np.linalg.norm(np.cross(b1, b2)) / det) + 1

    return [l1, l2, l3]

# TODO: better estimation of atom-related parameters (augmentation, number of beta projectors)
#       estimate number of bands, for this total pseudo-nuclear charge of system is needed
def estimate_memory(atom_files, num_atoms, a1, a2, a3, gk_cutoff, pw_cutoff, gamma, ndmag, **kwargs):
    """
    Estimate the memory consumption of the SIRIUS library. All input parameters are in atomic units.

    Parameters:
      a1 - first lattice vector (in a.u.)
      a2 - second lattice vector (in a.u.)
      a3 - third lattice vector (in a.u.)
      gk_cutoff - cutoff for G+k vectors or, in other words, for wave-functions (in a.u.^-1)
      pw_cutoff - cutoff for density and potential (in a.u.^-1)
      gamma - True if this is a Gamma-point calculation
      ndmag - number of magnetic dimensions (or type of spin tratement): 0 - non-magnetic, 1 - spin collinear, 3 - non-collinear cases

      kwargs - additional parameters:
        nbnd - number of bands; in spin-collinear case this is the number of bands per spin channel
        num_ranks_per_node - try to estimate memory with this number of ranks per node

    Return:
      m
    """

    nbnd = kwargs.get('nbnd', 0)
    num_ranks_per_node = kwargs.get('num_ranks_per_node', 1)

    # list of dictionaries describing each atom type
    atom_types = [get_atom_type_info(f) for f in atom_files]

    # get total number of atoms
    num_atoms_tot = np.sum(num_atoms)

    # get total nuclear charge and total number of beta-projectors
    z = 0
    nbeta = 0
    na_max = 0
    for i in range(len(atom_types)):
        z += atom_types[i]["z"] * num_atoms[i]
        nbeta += atom_types[i]["nbeta"] * num_atoms[i];
        na_max = max(na_max, num_atoms[i])

    # estimate number of bands
    if nbnd == 0:
        if ndmag in [0, 1]:
            nbnd = int(z / 2) + 10
        else:
            nbnd = int(z / 2) + 10

    # volume of the unit cell
    omega = abs(np.linalg.det([a1, a2, a3]))

    # volume of the Brillouin zone
    v0 = pow(2 * math.pi, 3) / omega

    # volume of the cutoff sphere for wave-functions
    v1 = 4 * math.pi * pow(gk_cutoff, 3) / 3

    # volume of the cutoff sphere for density and potential
    v2 = 4 * math.pi * pow(pw_cutoff, 3) / 3

    # approximate number of G+k vectors
    ngk = int(v1 / v0)
    if gamma: ngk /= 2

    # approximate number G vectors
    ng = int(v2 / v0)

    # by default, SIRIUS reduces G-vectors by half even if Gamma switch is not set;
    # this is because density and potential are real functions and we can use rho(G)=conj(rho(-G)) symmetry;
    # leave it like this
    if True: ng /= 2

    # number of spins (1 or 2)
    nsp = 1 if ndmag == 0 else 2

    # number of independent spin components (1 or 2)
    nsc = 2 if ndmag == 2 else 1

    # number of auxiliary basis functions
    # by default it is 4x number of bands
    nphi = 4 * nbnd

    # estimate size of augmentation operator
    nt_aug = 0
    m_aug = 0
    for a in atom_types:
        if a["augment"]:
            nt_aug += 1
            m_aug = max(m_aug, (a["nbeta"] + 1) * a["nbeta"] / 2)


    # get reciprocal lattice vectors
    M = np.linalg.inv(np.array([a1, a2, a3]))
    b1 = 2 * math.pi * M[:,0]
    b2 = 2 * math.pi * M[:,1]
    b3 = 2 * math.pi * M[:,2]

    # estimate size of the fine FFT grid
    fft_grid = get_fft_grid_size(b1, b2, b3, pw_cutoff)

    # estimate size of the coarse FFT grid
    fft_grid_coarse = get_fft_grid_size(b1, b2, b3, 2 * gk_cutoff)

    # Comments about FFT:
    #   1) FFT on the coarse grid is not parallelized in SIRIUS by default
    #   2) FFT buffers are allocated for the entire run


    gpu_max_mem = 16 * pow(2, 30) / num_ranks_per_node
    #cpu_max_mem = 64 * pow(2, 30)


    # estimate the minimum number of MPI ranks needed to fit FFT buffers and augmentation operator in GPU
    Np = 1
    while True:
        # memory consumption for each of Np ranks
        M = min(nt_aug, 2) * m_aug * ng / Np * 16 + min(m_aug * ng / Np * 16, pow(2, 30)) + \
            min(nt_aug, 1) * min(na_max * ng / Np * 16, pow(2, 30)) + \
            3 * fft_grid_coarse[0] * fft_grid_coarse[1] * fft_grid_coarse[2] * 16 + \
            3 * fft_grid[0] * fft_grid[1] * (1 + fft_grid[2] / Np) * 16
        if M < gpu_max_mem: break
        Np += 1


    # estimate the minimum number of MPI ranks for Davidson solver
    Nb = 1
    while True:
        # consumption of the Davison solver for each of Nb ranks
        M = (nbnd * nsp + 3 * nbnd * nsc + 3 * nphi * nsc + nbeta) * ngk / Nb * 16 + \
            3 * fft_grid_coarse[0] * fft_grid_coarse[1] * fft_grid_coarse[2] * 16 + \
            3 * fft_grid[0] * fft_grid[1] * (1 + fft_grid[2] / max(Np, Nb))
        # let's assume we do MAGMA diagonalization; in this case H and S matrices are allocated on GPU
        if True:
            if gamma:
                M += 2 * nphi * nphi * 8
            else:
                M += 2 * nphi * nphi * 16

        if M < gpu_max_mem: break
        Nb += 1

    # adjust total number of ranks
    Np = max(Np, Nb)

    print("approximate number of G vectors   : %i"%ng)
    print("approximate number of G+k vectors : %i"%ngk)
    print("total nuclear charge              : %f"%z)
    print("number of bands                   : %i"%nbnd)
    print("number of beta-projectors         : %i"%nbeta)

    print("estimated total number of MPI ranks: %i"%Np)
    print("estimated number of MPI ranks for band parallelisation : %i"%Nb)

    return Np, Nb

# list of UPF files (atom types)
atom_files = ['Si.pbe-n-rrkjus_psl.1.0.0.UPF', 'ge_pbe_v1.4.uspp.F.UPF']
# number of atoms for eash type
num_atoms = [511, 1]


a = 5.13 * 8
print(estimate_memory(atom_files, num_atoms, [0, a, a], [a, 0, a], [a, a, 0], 6, 20, False, 0, num_ranks_per_node=1))



