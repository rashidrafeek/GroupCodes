import guesthost.lattice as lat                                        
import ase.io                                                          
                                                                       
# Load system
sys = ase.io.read('structure.xyz')                                     
                                                                       
# Create lattice                                                       
lattice = lat.HPLattice(motifs, cell, nx, ny, nz)                      
                                                                       
# Compute local cell parameters                                        
pb_axis = (pb_orig_ind, [pb_ax1, pb_ax2, pb_ax3])                      
params = lattice.ucell_localcellparameters(sys, pb_axis)               
                                                                       
# Compute MA orientation                                               
c_ind, n_ind = 0, 4  # Example indices                                 
theta, phi = lattice.ucell_theta_phi(sys, c_ind, n_ind, pb_axis, dir=0)
                                                                       
# Compute order parameters                                             
eta_lat = lattice.ucell_η_lat(sys, pb_axis, br_axis, dir_coup=0)       
eta_ma = lattice.ucell_η_MA(sys, c_ind, n_ind, pb_axis, dir=0)         
