import numpy as np
import unittest
import guesthost as gh
import ase.io

class TestUnitcellFunctions(unittest.TestCase):
    def setUp(self):
        # Load the test structure, following test_lattice_orientations.py pattern
        self.mpb_sys_ini_path = "structures/mpb_trajectory.xyz"
        self.trj = gh.Trajectory(self.mpb_sys_ini_path, order=True)
        self.n = 4  # Number of unitcells in each direction
        self.all_inds = [320, 384, 385, 386, 256, 387, 388, 389, 0, 64, 65, 66]
        self.ma_inds = [0, 1, 2, 3, 4, 5, 6, 7]
        self.host_inds = [8, 9, 10, 11]

        # Fragmentize and create lattices following test_lattice_orientations.py
        ma_lists = self.trj.fragmentize(64, self.ma_inds, gh.MethylAmmonium, ordering="unitcell")
        host_lists = self.trj.fragmentize(64, self.host_inds, gh.Host, ordering="unitcell")
        lattices = self.trj.create_lattice(
            64, [self.ma_inds, self.host_inds], [gh.MethylAmmonium, gh.Host],
            (self.n, self.n, self.n), gh.HPLattice, ordering="type", unit_order=self.all_inds
        )
        self.lattraj = gh.LatticeTrajectory(lattices)

        # Get the first system and lattice for testing
        self.sys = self.trj.atoms_list[0]
        self.lattice = lattices[0]

    def test_ucell_localcellvecs(self):
        """Test local cell vectors computation"""
        # Mock pb_axis: Pb at 8, I at 9,10,11
        pb_axis = (8, [9, 10, 11])
        vecs = self.lattice.ucell_localcellvecs(self.sys, pb_axis)
        self.assertEqual(len(vecs), 3)
        for v in vecs:
            self.assertEqual(len(v), 3)
            self.assertTrue(np.all(np.isfinite(v)))

    def test_ucell_localcellparameters(self):
        """Test local cell parameters computation"""
        pb_axis = (8, [9, 10, 11])
        params = self.lattice.ucell_localcellparameters(self.sys, pb_axis)
        required_keys = ['a', 'b', 'c', 'α', 'β', 'γ', 'vol']
        for key in required_keys:
            self.assertIn(key, params)
            self.assertTrue(np.isfinite(params[key]))

    def test_ucell_theta_phi(self):
        """Test theta phi computation"""
        c_ind, n_ind = 0, 4  # C and N indices
        pb_axis = (8, [9, 10, 11])
        theta, phi = self.lattice.ucell_theta_phi(self.sys, c_ind, n_ind, pb_axis, 0)
        self.assertTrue(np.isfinite(theta))
        self.assertTrue(np.isfinite(phi))
        self.assertGreaterEqual(theta, 0)
        self.assertLessEqual(theta, 180)

    def test_ucell_η_MA(self):
        """Test MA order parameter computation"""
        c_ind, n_ind = 0, 4
        pb_axis = (8, [9, 10, 11])
        eta = self.lattice.ucell_η_MA(self.sys, c_ind, n_ind, pb_axis, 0)
        self.assertEqual(len(eta), 2)
        self.assertTrue(np.all(np.isfinite(eta)))

    def test_compute_for_all_with_sys(self):
        """Test computing unitcell functions for all frames using LatticeTrajectory"""
        # Compute theta phi for all frames
        results = self.lattraj.compute_for_all_with_sys(
            self.trj.atoms_list,
            lambda lat, sys: lat.ucell_theta_phi(sys, 0, 4, (8, [9, 10, 11]), 0)
        )
        self.assertEqual(len(results), self.trj.nt)
        for res in results:
            theta, phi = res
            self.assertTrue(np.isfinite(theta))
            self.assertTrue(np.isfinite(phi))
            self.assertGreaterEqual(theta, 0)
            self.assertLessEqual(theta, 180)

    def test_U_r_and_ucell_ω(self):
        """Test rotation matrix and ω computation"""
        eta = np.array([0.7, 0.3])
        U = self.lattice.U_r([0, 0], [np.pi, np.pi])
        self.assertEqual(U.shape, (2, 2))
        omega = self.lattice.ucell_ω(eta, [0, 0])
        self.assertEqual(len(omega), 2)
        self.assertTrue(np.allclose(omega, eta, atol=1e-10))  # For R=[0,0], U=I

if __name__ == "__main__":
    unittest.main()