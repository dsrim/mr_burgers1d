
import pickle as pkl
import numpy as np
import matplotlib.pyplot as pl


print('loading pickled ROM...')
with open('_output/ROM_object.pkl',mode='r') as input_file:
    Rom0 = pkl.load(input_file)

print('= done.')


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

tri = Rom0._rom_tri

if 0:
    c = Rom0._get_uniform_samples(4)
    c = 0.98*(c - 0.01) + 0.01
    
    n = 1
    dt = Rom0._hfm_dt
    for n in range(tri.nsimplex):
        print('sampling simplex no. ' + str(n))
        f1,ax1 = pl.subplots(nrows=1,ncols=1,figsize=(10,3))
        
        for j in range(c.shape[1]):
            mu0 = Rom0._get_spatial_coords(tri,n,c[:,j])
            Rom0.run_rom(mu=mu0,evaluate=True,M0=4000)
        
            ra_list = Rom0._rom_ra_list
            L = len(ra_list)
            x = Rom0._x
            dpi = Rom0._dpi
        
            u = Rom0.run_hfm(mu0)
            e4t = []
            for k in range(L):
                ra = ra_list[k]
                #l2error = np.sqrt(dt)*np.linalg.norm(ra - u[:,k])
                
                max_error = np.max(np.abs(ra - u[:,k]))
                e4t.append(max_error)
        
            ax1.semilogy(e4t,'royalblue')
        
            # TODO save to _plots subdir?
        
        fig_name = 'rom_error_' + str(n) + '.png'
        f1.savefig(fig_name,dpi=dpi)
        pl.close(f1)


if 0:
    ## print *actual* basis size:
    test_mu()
    
    r_list = Rom0._rom_r_list
    athresh = 0.
    for k in range(len(r_list)):
        r0 = r_list[k]
        ar0 = np.abs(r0)
        ar0 = ar0 / ar0.max()
        nthresh = np.sum(ar0 >= 1e-3)  
        athresh += float(nthresh) / float(len(r_list))
        print('above threshold = ' + str(nthresh))

    
# 

mu_list = []

for j in range(100000):
    
    Rom2.run_rom(mu=mu1,M0=4000)
