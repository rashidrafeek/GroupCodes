import numpy as np
from guesthost.analysis.functions_modules import (
        proj2plane, self_correlation_P1, self_correlation_P2,
        distance_of_point_on_plane
)
from pymatgen.util.coord_cython import pbc_shortest_vectors
from pymatgen.core import Lattice as PmgLattice

class Motif:
    """Class representing one unit of a repeating collection of fragments"""
    
    def __init__(self, frg_list):
        self.fragments = frg_list
        self.nfrg = len(frg_list)

class Lattice:
    """Class representing a lattice composed of several motifs with its associated cell"""

    def __init__(self, motif_list, cell, nx, ny, nz, pbc=True):
        motif_grid = np.array(motif_list).reshape(nx, ny, nz)

        self.motif_grid = motif_grid
        self.cell = cell
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.ncells = (nx, ny, nz)
        self.pbc = pbc

    def compute_for_all(self, fn, *args, dtype=float, **kwargs):
        all_vals = np.zeros((self.nx, self.ny, self.nz), dtype=dtype)

        for ix, iy, iz in np.ndindex(self.nx, self.ny, self.nz):
            val = fn(self, (ix, iy, iz), *args, **kwargs)

            all_vals[ix, iy, iz] = val

        return all_vals


class HPLattice(Lattice):
    """Class representing a hybrid perovskite lattice"""

    def __init__(self, motif_list, cell, nx, ny, nz):
        super().__init__(motif_list, cell, nx, ny, nz)

    def ma_torsion(self, ind, frg_ref, htyp="N", atan=False):
        ix, iy, iz = ind
        frg = self.motif_grid[ix, iy, iz].fragments[0]
        if htyp == "N":
            v1 = frg_ref.HN
        elif htyp == "C":
            v1 = frg_ref.HC

        if atan:
            val = frg.caltorsion_atan(v1, htyp=htyp)
        else:
            val = frg.caltorsion(v1, htyp=htyp)

        return val

    def all_ma_torsions(self, lattice_ref, htyp="N", atan=False):
        torsions = np.zeros((self.nx, self.ny, self.nz))
        for ix, iy, iz in np.ndindex(self.nx, self.ny, self.nz):
            frg_ref = lattice_ref.motif_grid[ix,iy,iz].fragments[0]
            torsion = self.ma_torsion((ix, iy, iz), frg_ref, htyp=htyp, atan=atan)

            torsions[ix,iy,iz] = torsion

        return torsions

    def ma_torsion_removetilts(self, ind, frg_ref, htyp="N", match_rows=False, return_tilt=False):
        ix, iy, iz = ind
        frg = self.motif_grid[ix, iy, iz].fragments[0]
        if htyp == "N":
            v1 = frg_ref.HN
        elif htyp == "C":
            v1 = frg_ref.HC

        val = frg.caltorsion_removetilts(
            v1, htyp=htyp, match_rows=match_rows, 
            return_tilt=return_tilt
        )

        return val

    def all_ma_torsions_removetilts(self, lattice_ref, htyp="N", match_rows=False):
        torsions = np.zeros((self.nx, self.ny, self.nz))

        for ix, iy, iz in np.ndindex(self.nx, self.ny, self.nz):
            frg_ref = lattice_ref.motif_grid[ix,iy,iz].fragments[0]
            torsion = self.ma_torsion_removetilts(
                    (ix, iy, iz), frg_ref, htyp=htyp, 
                    match_rows=match_rows, return_tilt=False
                    )

            torsions[ix,iy,iz] = torsion

        return torsions

    def all_ma_torsions_withtilts(self, lattice_ref, htyp="N", match_rows=False):
        torsions = np.zeros((self.nx, self.ny, self.nz))
        tilts = np.zeros((self.nx, self.ny, self.nz))

        for ix, iy, iz in np.ndindex(self.nx, self.ny, self.nz):
            frg_ref = lattice_ref.motif_grid[ix,iy,iz].fragments[0]
            torsion, tilt = self.ma_torsion_removetilts(
                    (ix, iy, iz), frg_ref, htyp=htyp, 
                    match_rows=match_rows, return_tilt=True
                    )

            torsions[ix,iy,iz] = torsion
            tilts[ix,iy,iz] = tilt

        return torsions, tilts

    def ucell_localcellvecs(self, sys, pb_axis):
        """Compute local cell vectors using Pb axes"""
        or_ind, ax_inds = pb_axis
        or_pos = sys.positions[or_ind]
        ax_posvecs = sys.positions[ax_inds]

        pmglattice = PmgLattice(self.cell)
        or_fr = pmglattice.get_fractional_coords(or_pos)

        vecs = []
        for ax_pos in ax_posvecs:
            ax_fr = pmglattice.get_fractional_coords(ax_pos)
            svec = pbc_shortest_vectors(pmglattice, or_fr, ax_fr)[0]
            vecs.append(svec[0])  # Extract the 3D vector

        return vecs

    def ucell_localcellparameters(self, sys, pb_axis):
        """Compute local cell parameters: lengths, angles, volume"""
        av, bv, cv = self.ucell_localcellvecs(sys, pb_axis)
        a, b, c = [np.linalg.norm(v) for v in [av, bv, cv]]

        α = np.degrees(np.arccos(np.dot(bv, cv) / (b * c)))
        β = np.degrees(np.arccos(np.dot(av, cv) / (a * c)))
        γ = np.degrees(np.arccos(np.dot(av, bv) / (a * b)))

        cellmat = np.column_stack([av, bv, cv])
        vol = np.linalg.det(cellmat)

        return {'a': a, 'b': b, 'c': c, 'α': α, 'β': β, 'γ': γ, 'vol': vol}

    def reciprocalvecs(self, h):
        """Compute reciprocal vectors for cell matrix h"""
        v = np.cross(h[1], h[2])
        vol = np.dot(h[0], v)
        g = np.array([np.cross(h[j], h[k]) / vol for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]])
        return g

    def ucell_theta_phi(self, sys, c_ind, n_ind, pb_axis, dir):
        """Compute theta and phi for MA CN vector relative to local axes"""
        lst = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        i, j, k = lst[dir]

        orig_pos = sys.positions[pb_axis[0]]
        ax_positions = sys.positions[pb_axis[1]]

        pmglat = PmgLattice(self.cell)
        orig_fr = pmglat.get_fractional_coords(orig_pos)
        lcell = np.array([pbc_shortest_vectors(pmglat, orig_fr, pmglat.get_fractional_coords(ax_pos))[0][0] for ax_pos in ax_positions])

        r = sys.positions[n_ind] - sys.positions[c_ind]

        # theta
        theta = np.degrees(np.arccos(np.dot(r, lcell[i]) / (np.linalg.norm(r) * np.linalg.norm(lcell[i]))))

        # phi
        g = self.reciprocalvecs(lcell)
        vprp = np.dot(r, g[i]) * lcell[i]
        vpl = r - vprp
        angb = np.dot(vpl, lcell[j]) / (np.linalg.norm(vpl) * np.linalg.norm(lcell[j]))
        angc = np.dot(vpl, g[k])
        phase = 1.0 if angc >= 0 else -1.0
        phi = phase * np.degrees(np.arccos(np.clip(angb, -1, 1)))
        if np.isnan(phi):
            phi = 90.0

        return theta, phi

    def ucell_br_distance_from_plane(self, sys, pb_axis, br_axis, dir_pl, dir_br):
        """Compute Br distance from plane defined by Pb axes"""
        o_pb, ax_pb = pb_axis
        pos_pb_orig = sys.positions[o_pb]
        br_move_dir = list(set([0, 1, 2]) - set(dir_pl))[0]

        pmglat = PmgLattice(self.cell)
        orig_fr = pmglat.get_fractional_coords(pos_pb_orig)

        ax_vecs = []
        for d in dir_pl:
            ax_pos = sys.positions[ax_pb[d]]
            ax_fr = pmglat.get_fractional_coords(ax_pos)
            vec = pbc_shortest_vectors(pmglat, orig_fr, ax_fr)[0]
            ax_vecs.append(vec)

        pos_br = sys.positions[br_axis[dir_br]]
        br_fr = pmglat.get_fractional_coords(pos_br)
        pb_br_vec = pbc_shortest_vectors(pmglat, orig_fr, br_fr)[0]

        pln_perp_vec = np.cross(ax_vecs[0], ax_vecs[1])
        norm_pln_perp = np.linalg.norm(pln_perp_vec)
        if norm_pln_perp == 0:
            return 0.0
        pb_br_plnpperp_vec = (np.dot(pb_br_vec, pln_perp_vec) / norm_pln_perp**2) * pln_perp_vec
        pb_br_plnparallelvec = pb_br_vec - pb_br_plnpperp_vec
        pos_br_projection_pln = pos_pb_orig + pb_br_plnparallelvec

        proj_fr = pmglat.get_fractional_coords(pos_br_projection_pln)
        br_pln_vec = pbc_shortest_vectors(pmglat, proj_fr, br_fr)[0]

        gl_dist = np.linalg.norm(br_pln_vec) * np.sign(br_pln_vec[br_move_dir])
        return gl_dist

    def ucell_pb_br_pb_angle(self, sys, pb_axis, br_axis, dir):
        """Compute Pb-Br-Pb angle"""
        orig_pb = pb_axis[0]
        mid_brs = br_axis
        end_pbs = pb_axis[1]
        ang = sys.get_angle(orig_pb, mid_brs[dir], end_pbs[dir])
        return ang

    def ucell_pb_br_pb_angle_hostrelative(self, sys, pb_axis, br_axis, dir, dir_coup):
        """Compute Pb-Br-Pb angle with host-relative adjustment"""
        orig_ang = self.ucell_pb_br_pb_angle(sys, pb_axis, br_axis, dir)
        dir_pl = [dir, dir_coup]
        pl_dist = self.ucell_br_distance_from_plane(sys, pb_axis, br_axis, dir_pl, dir)
        if pl_dist > 0.0:
            return 360.0 - orig_ang
        else:
            return orig_ang

    def ucell_η_lat(self, sys, pb_axis, br_axis, dir_coup, ref=180.0, N=1.0):
        """Compute lattice order parameter η"""
        br_dirs = [(dir_coup + 1) % 3, (dir_coup + 2) % 3]
        θ_v = [self.ucell_pb_br_pb_angle_hostrelative(sys, pb_axis, br_axis, d, dir_coup) for d in br_dirs]
        V = [(θ - ref) / N for θ in θ_v]
        return V

    def U_r(self, R, k_vec, MA=False):
        """Rotation matrix for order parameters"""
        k_dot_R = np.dot(k_vec, R)
        if MA:
            k_dot_R = (np.pi / 4) * (1.0 - np.cos(k_dot_R))
        c = np.cos(k_dot_R)
        s = np.sin(k_dot_R)
        U = np.array([[c, s], [-s, c]])
        return U

    def ucell_ω(self, η, R, MA=False, orig_cell=(0, 0)):
        """Transform η to ω using rotation matrix"""
        k_vec = np.array([0.5, 0.5]) * 2 * np.pi
        U = self.U_r(np.array(R) + np.array(orig_cell), k_vec, MA)
        ω = U @ η
        return ω

    def _change_ϕ_domain(self, ϕ, orig_domain=(-180, 180), final_domain=(-45, 135), eps=None):
        """Change phi domain"""
        oi, of_ = orig_domain
        fi, ff = final_domain
        period = ff - fi
        if eps is not None and abs(ϕ) < eps + fi:
            ϕ = fi
        elif eps is not None and abs(ϕ) > ff - eps:
            ϕ = ff
        else:
            if oi <= ϕ <= fi:
                ϕ += period
            elif ff <= ϕ <= of_:
                ϕ -= period
        return ϕ

    def ucell_η_MA_static(self, θ, ϕ, norm=False, change_phi_domain=True, ϕ_domain=(-45, 135), eps=None):
        """Compute MA η from θ, ϕ"""
        if change_phi_domain:
            ϕ = self._change_ϕ_domain(ϕ, final_domain=ϕ_domain, eps=eps)
        vb = np.sin(np.radians(θ)) * np.cos(np.radians(ϕ))
        vc = np.sin(np.radians(θ)) * np.sin(np.radians(ϕ))
        V = np.array([vb, vc])
        if norm:
            V /= np.linalg.norm(V)
        return V

    def ucell_η_MA(self, sys, c_ind, n_ind, pb_axis, dir, **kwargs):
        """Compute MA η for unit cell"""
        θ, ϕ = self.ucell_theta_phi(sys, c_ind, n_ind, pb_axis, dir)
        return self.ucell_η_MA_static(θ, ϕ, **kwargs)

    def ma_orientation(self, ind, ax):
        "MA orientations for unitcell specified by `ind` with respect to axis, `ax`"""

        ix, iy, iz = ind
        cn_vec = self.motif_grid[ix, iy, iz].fragments[0].CN
        lcell = self.pb_environment_cell(ind)
        vpl, t, p = proj2plane(ax, cn_vec, lcell)

        return t, p

    def all_ma_orientations(self, ax):
        theta_vals = np.zeros((self.nx, self.ny, self.nz))
        phi_vals = np.zeros((self.nx, self.ny, self.nz))

        for ix, iy, iz in np.ndindex(self.nx, self.ny, self.nz):
            t, p = self.ma_orientation((ix, iy, iz), ax)

            theta_vals[ix, iy, iz] = t
            phi_vals[ix, iy, iz] = p

        return theta_vals, phi_vals

    def shift_index(self, ind, ax, n=1):
        ind = list(ind)
        ind[ax] = (ind[ax]+n) % self.ncells[ax]

        return ind

    def get_motif(self, ind):
        ix, iy, iz = ind

        return self.motif_grid[ix, iy, iz]

    def pb_environment_coords(self, ind):
        ax_positions = np.array([
                self.get_motif(self.shift_index(ind, 0)).fragments[1].B,
                self.get_motif(self.shift_index(ind, 1)).fragments[1].B,
                self.get_motif(self.shift_index(ind, 2)).fragments[1].B
            ])

        return ax_positions

    def pb_environment_cell(self, ind):
        ix, iy, iz = ind
        o_host_frg = self.get_motif(ind).fragments[1]
        o_pos = o_host_frg.B
        
        ax_positions = self.pb_environment_coords(ind)

        if self.pbc:
            lcell = pbc_vectors(self.cell, o_pos, ax_positions).T
        else:
            lcell = (ax_positions - o_pos).T

        return lcell

    def pb_br_pb_angle(self, ind, ax):
        ix, iy, iz = ind
        hostfrg_orig = self.get_motif(ind).fragments[1]
        hostfrg_ax = self.get_motif(self.shift_index(ind, ax)).fragments[1]

        atoms_orig = hostfrg_orig.atoms
        atoms_ax = hostfrg_ax.atoms
        atoms = atoms_orig + atoms_ax

        x_ind = ax+1
        ang = atoms.get_angle(0, x_ind, 4, mic=self.pbc)

        return ang

    def pb_br_pb_angle_hostrelative(self, ind, ax, coup_dir):
        orig_ang = self.pb_br_pb_angle(ind, ax)
        pln = (ax, coup_dir)
        pln_dist = self.br_distance_from_plane(ind, pln, ax)
        sign = np.sign(pln_dist)

        if sign == 1:
            ang = 360-orig_ang
        else:
            ang = orig_ang
        
        return ang

    def br_environment_coords(self, ind):
        o_host_frg = self.get_motif(ind).fragments[1]
        
        ax_positions = np.array([
                o_host_frg.X[0],
                self.get_motif(self.shift_index(ind, 0, -1)).fragments[1].X[0],
                o_host_frg.X[1],
                self.get_motif(self.shift_index(ind, 1, -1)).fragments[1].X[1],
                o_host_frg.X[2],
                self.get_motif(self.shift_index(ind, 2, -1)).fragments[1].X[2],
            ])

        return ax_positions

    def br_distance_from_plane(self, ind, pln, br_dir):
        o_host_frg = self.get_motif(ind).fragments[1]
        o_pos = o_host_frg.B
        pb_coords = self.pb_environment_coords(ind)
        br_coords = self.br_environment_coords(ind)

        all_coords = np.concatenate([pb_coords, br_coords])
        if self.pbc:
            all_coords = pbc_vectors(self.cell, o_pos, all_coords, relative=False)

        br_ind = 3+(2*br_dir)
        dist_dat = distance_of_point_on_plane(
                o_pos, all_coords[pln[0]], all_coords[pln[1]], all_coords[br_ind]
        )
        (ax_sign,) = {0,1,2} - set(pln)
        dist = dist_dat[0]*np.sign(dist_dat[1][ax_sign])

        return dist

def pbc_vectors(cell, v1, v2, orig_ind=0, relative=True):
    """Obtain shortest vectors with PBC for the given vectors"""

    pmglattice = PmgLattice(cell)
    v1_fr = pmglattice.get_fractional_coords(v1)
    v2_fr = pmglattice.get_fractional_coords(v2)
    
    # The first vector in v1 is taken as origin
    svecs = pbc_shortest_vectors(pmglattice, v1_fr, v2_fr)[orig_ind]
    if not relative:
        svecs = svecs+v1

    return svecs

class LatticeTrajectory:
    """A trajectory of lattice objects for each structure in the original trajectory"""

    def __init__(self, lattice_list):
        self.lattices = lattice_list
        self.nt = len(lattice_list)
        
        init_lat = lattice_list[0]
        self.nx = init_lat.nx
        self.ny = init_lat.ny
        self.nz = init_lat.nz
        self.pbc = init_lat.pbc
    
    def compute_for_all(self, fn, *args, **kwargs):
        vals = []
        for lat in self.lattices:
            val = fn(lat, *args, **kwargs)
            vals.append(val)

        return vals
    
    def compute_for_all_cells(self, fn, *args, dtype=float, **kwargs):
        vals = []
        for lat in self.lattices:
            val = lat.compute_for_all(fn, *args, dtype=dtype, **kwargs)
            vals.append(val)

        return vals

    def compute_for_all_with_sys(self, sys_list, fn, *args, **kwargs):
        """Compute for all lattices with corresponding systems"""
        vals = []
        for lat, sys in zip(self.lattices, sys_list):
            val = fn(lat, sys, *args, **kwargs)
            vals.append(val)

        return vals

    def ma_self_correlation_P1(self):
        """MA self correlation with first order legendre polynomial P1"""

        fn_get_cn = lambda x, ind: x.motif_grid[ind[0],ind[1],ind[2]].fragments[0].CN
        all_corr = []
        for ix, iy, iz in np.ndindex(self.nx, self.ny, self.nz):
            all_cn = [fn_get_cn(self.lattices[t], (ix, iy, iz)) for t in range(self.nt)]

            corr = self_correlation_P1(all_cn)
            all_corr.append(corr)

        return np.array(all_corr)

    def ma_self_correlation_P2(self):
        """MA self correlation with first order legendre polynomial P2"""

        fn_get_cn = lambda x, ind: x.motif_grid[ind[0],ind[1],ind[2]].fragments[0].CN
        all_corr = []
        for ix, iy, iz in np.ndindex(self.nx, self.ny, self.nz):
            all_cn = [fn_get_cn(self.lattices[t], (ix, iy, iz)) for t in range(self.nt)]

            corr = self_correlation_P2(all_cn)
            all_corr.append(corr)

        return np.array(all_corr)
