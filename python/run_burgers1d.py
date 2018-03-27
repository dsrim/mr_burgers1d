

from burgers1d import Rom
import numpy as np
import matplotlib.pyplot as pl

Rom0 = Rom()

Rom0.sample_hfm(N=250,dt=0.0125,xl=0.,xr=100.,M=4000)
#Rom0.sample_hfm(N=250,xl=0.,xr=200.,dt=0.006125/2.,M=8000)
#Rom0.sample_hfm(N=500,xl=0.,xr=200.,dt=0.0125,M=8000)
Rom0.plot_hfm_solution()

if 1:
    Rom0.build_bases(tol=1e-10,max_basis_size=200,Mfinal=100,t_interval=20,\
                   nalpha=4,P=4)
    Rom0.save()

    mu0 = [7.2,0.04]
    Rom0.run_rom(mu=mu0,evaluate=True,M0=50)
    #Rom0.plot_triangulation()
    Rom0.plot_rom_solution()
    Rom0.compare_rom_solution(mu0,interval=5)


def random_mu():
    mu = np.random.rand(2)
    mu[0] = mu[0]*(9. - 3.) + 3.
    mu[1] = mu[1]*(0.075 - 0.02) + 0.02
    return mu


def test_mu():

    mu0 = random_mu()
    print('mu = (' + str(mu0[0]) + ', ' + str(mu0[1]) + ')')

    simplex0 = Rom0._rom_tri.find_simplex(mu0)
    print('simplex no. = ' + str(simplex0))
    
    Rom0.run_rom(mu=mu0,evaluate=True)
    Rom0.compare_rom_solution(mu0,interval=10)


