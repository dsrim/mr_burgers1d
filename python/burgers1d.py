# See LICENSE
# Donsub Rim, Columbia U. (dr2965@columbia.edu) 2018-08-16
#   * cleaned up to make repository public

import os,sys
import numpy as np
import matplotlib.pyplot as pl
from scipy.spatial import Delaunay
from scipy.interpolate import InterpolatedUnivariateSpline
from copy import copy

# import displacement interpolation routines
dinterp_path = '../../dinterp/dinterp'
sys.path.insert(0,dinterp_path)
import dinterp


class Rom(object):
    r"""
        Basic class representing reduced-order models

    """


    def __init__(self):

        self._N = None              # HFM dof 
        self._x = None              # HFM domain 
        self._h = None              # HFM mesh-width for HFM
        self._hfm_mu = None         # current set of parameters
        self._hfm_M = None          # HFM number of time-steps
        self._hfm_dt = None         # HFM time-step size 
        self._hfm_D = None          # HFM stiffness
        self._hfm_src = None        # HFM source-term 
        self._hfm_k2mu_list = None  # list of HFM parameters (mu)
        self._hfm_u_list = None     # list of HFM solutions

        self._rom_tri = None        # ROM Delaunay triangular of domain
        self._rom_P = None
        self._rom_r1 = None
        self._rom_ra = None
        self._rom_ra_list = []
        self._rom_i0 = None
        self._rom_r_list = []       # ROM solution list
        self._rom_basis = []        # ROM basis functions
        self._rom_basis_raw = []    # ROM basis functions
        self._rom_F = []            # ROM flux functions
        self._rom_bc = []           # ROM BC constraints
        self._rom_src = []          # ROM source functions
        self._rom_M = []            # ROM basis numbers
        self._rom_T = []            # ROM transition matrices
        self._time_index_list = []  # ROM grid-points in time

        self._rom_reduced = False   # ROM basis reduction flag
        self._rom_set_up = False    # ROM basis reduction flag

        self._dpi = 200             # dpi for savefigs

    @property
    def hfm_D(self):
        r"""differentation matrix"""
        if self._hfm_D is None:
            self._compute_diff()
        return self._hfm_D

    @property
    def hfm_src(self):
        r"""source term"""
        self._compute_hfm_src()
        return self._hfm_src

    def _compute_hfm_src(self):
        r"""compute exponential source term"""
        mu1 = self._hfm_mu[1]
        x = self._x
        self._hfm_src = 0.02*np.exp(mu1*x)


    def _compute_diff(self):
        r"""

        assemble differentiation matrix

        """
        N = self._N
        h = self._h
        
        D = np.zeros((N,N))
        D += (np.diag([.5]*N,k=0) - np.diag([.5]*(N-1),k=-1)) / h
        D[ 0, 0] = -.5 /h
        D[ 0, 1] =  .5 /h
        D[-1,-2] = -.5 /h
        D[-1,-1] =  .5 /h

        self._hfm_D = D

        
    def _hfm_time_step(self,u):
        r"""
        
         Take one time-step of the high-fidelity model(HFM)
        
        """
        
        D = self.hfm_D
        src = self.hfm_src
        dt = self._hfm_dt
        
        u = np.array(u)
        u = u - np.dot(dt*D,u*u) + dt*src
        
        return u


    def run_hfm(self,mu,N=250,xl=0.,xr=100.,dt=0.0125,M=4000):
        r"""
        Run High Fidelity Model (HFM)

        :Inputs:
            mu: (2,)-array designating parameters

        :Outputs:
            u: (N,)-array representing finite volume solution
        
        """

        # set up domain
        self._N = N
        x = np.linspace(xl,xr,N)
        h = (xr - xl)/N
        
        self._x = x
        self._h = h
        self._hfm_dt = dt
        self._hfm_M  = M
        self._hfm_mu = mu
        
        # solve high-dimensional problem for given mu
        u = np.zeros((N,M+1))    
        
        for j in range(M):
            u[0:2] = float(mu[0])    # set incoming BC
            u[:,j+1] = self._hfm_time_step(u[:,j])
        
        return u


        
    def sample_hfm(self,N=250,dt=0.0125,M=4000,xl=0.,xr=100., \
                   mu0_list = [3.  , 6.  , 9.   ],\
                   mu1_list = [0.02, 0.05, 0.075]):
        r"""

        Run high fidelity model (HFM) for given two parameter values for
        the Burgers' equation. The solution and the given parameter values are 
        stored in ``self._hfm_k2mu_list = k2mu`` and  ``self._hfm_u_list``.
        Also saves to a file 

        then save solution to file

        :HFM:
        - Finite volume method on uniform grid
        - 1st order Godunov flux with no entropy fix 
        
        :Input:
        - N: number of grid-points
        - M: total number of time-steps to take 
        - dt: time-step size
        - xl: left boundary
        - xr: right boundary
        - mu0_list: points to evaluate for first parameter mu0 designating 
          left incoming boundary
        - mu0_list: points to evaluate for second parameter mu1 designating
          the strength of the source term
        
        """


        self._hfm_mu0_list = mu0_list    # incoming left BC
        self._hfm_mu1_list = mu1_list    # parameter for the source term

        mu = np.zeros(2)    # parameter values
        k2mu   = []
        u_list = []
        
        print('running HFM ...')
        k = 0   # run number
        for i in range(len(mu0_list)):
            for j in range(len(mu1_list)):
                
                mu[0] = mu0_list[i]
                mu[1] = mu1_list[j]
                k2mu.append((mu[0],mu[1]))
                
                sys.stdout.write('\r= running HFM for parameter values: ' \
                    + 'mu0 = {:1.4f}, mu1 = {:1.4f} '.format(mu[0],mu[1]))
                sys.stdout.flush()

                # run HFM
                uk = self.run_hfm(mu,N=N,xl=xl,xr=xr,dt=dt,M=M)
                u_list.append(uk)

                fname = os.path.join('_output','hfm_sol_{:02d}.npy'.format(k))
                np.save(fname,uk)
                k += 1
        
        self._hfm_k2mu_list = k2mu
        self._hfm_u_list = u_list
        
        
    def _diffp(self,a):
        """
        differentiate with padding: (output vector length unchanged)
        
        """
        return np.diff(np.concatenate((a[:1],a)))


    def _supports(self,a):
        """
        return begin/ending indices of connected supports of the input vector 
        
        """
        b = 1.*(a > 0.)
        dpb = np.diff(np.concatenate(([0.],b),axis=0))
        I = np.arange(len(a))
        ibegin = I[dpb > 0.]
        iend   = I[dpb < 0.]
        if len(ibegin) > len(iend):
            iend = np.concatenate((iend, [len(a)]))
        return (ibegin,iend)


    def _get_parts(self,v):
        """
        return positive / negative parts of a vector
        
        """
        vp = v*(v >= 0.)
        vm = v*(v <  0.)
        return vp,vm

    
    def _get_spos(self,v):
        """
        return mean position of the supports  
        
        """
        
        supp = np.abs(v) > 0.
        return np.mean(np.arange(len(v))[supp])

    def _get_pieces(self,v):
        r"""
        obtain pieces (connected supports) of a vector where the other pieces
        have been zero'ed out by multiplying a characteristic function

        """
        ibegin,iend = self._supports(v)
        pieces_list = []
        
        for k in range(len(ibegin)):
            ii = np.zeros(v.shape,dtype=bool)
            ii[ibegin[k]:iend[k]] = True
            w = v*ii
            pieces_list.append(w)
        return pieces_list


    def _add_gc(self,v,n=(2,2)):
        r"""
        pre- and post-pend ghost cells

        """
        return np.concatenate((np.array([0.]*n[0]),v,np.array([0.]*n[1])),axis=0)


    def _del_gc(self,v,n=(2,2)):
        r"""
        remove ghost cells

        """
        if (n[0] > 0) and (n[1] > 0):
            return v[n[0]:-n[1]]
        elif (n[0] == 0) and (n[1] > 0):
            return v[:-n[1]]
        elif (n[0] > 0) and (n[1] == 0):
            return v[n[0]:]


    def plot_all_parts(self,v0,v1):
        r"""
        plot all positive / negative parts

        """
        diffp = self._diffp
        get_parts = self._get_parts

        dv0 = diffp(v0)
        dv1 = diffp(v1)
    
        dv0p,dv0m = get_parts(dv0)
        dv1p,dv1m = get_parts(dv1)
    
        fig,axes = pl.subplots(nrows=3,ncols=2,figsize=(12,6))
        axes[0,0].plot(v0);   axes[0,0].set_title('u0');
        axes[0,1].plot(v1);   axes[0,1].set_title('u1');
        axes[1,0].plot(dv0p); axes[1,0].set_title('+ part of Du0');
        axes[2,0].plot(dv0m); axes[2,0].set_title('- part of Du0');
        axes[1,1].plot(dv1p); axes[1,1].set_title('+ part of Du1');
        axes[2,1].plot(dv1m); axes[2,1].set_title('- part of Du1');
        
        fig.tight_layout()

        return fig, axes


    def _plot_2times(self,j0=3,j1=200,k=2):
        r"""
        plot some positive and negative parts for illustration

        :Inputs:
    
        - k: snapshot number
        - j0,j1: time-steps to compare

        """
        j0 = 3
        j1 = 200
        k  = 2
        
        print(u2mu[k])
        
        mu = np.zeros(2)
        mu[0] = u2mu[k][0]
        mu[1] = u2mu[k][1]
        
        u0   = u_list[k]
        u00  = np.array(u0)[:, j0]
        u01  = np.array(u0)[:, j1]
        
        du00 = diffp(u00)
        du01 = diffp(u01)
        
        du00p,du00m = get_parts(du00)
        du01p,du01m = get_parts(du01)
        
        plot_all_parts(u00,u01)


    def _dinterp_parts(self,v0,v1,alpha,tol=1e-12,verbose=False):
        r"""

        displacement interpolation by parts

        1. Take two 1D-arrays and for each: differentiate and obtain *pieces*
           (fctn times char. fctn of non-zero connected supports)
        2. Match pieces that are closer together in the domain
        3. Apply displacement interpolation between the corresponding pieces

        :Input:
        - v0,v1: two 1D arrays to interpolate
        - alpha: interpolation parameter (between 0 and 1)
        - tol: tolerance for assessing if two vectors are identical
        - verbose: display messages (for debugging)

        :Output:
        - a list containing individually interpolated pieces

        """
        
        # initialize some variables
        restrict = False
        add_gc = self._add_gc
        del_gc = self._del_gc
        get_pieces = self._get_pieces
        get_spos = self._get_spos
        
        ## extrapolate right BC

        # get given values
        y0 = v0[-4:]
        y1 = v1[-4:]
    
        # add ghost cells of size 2
        v0 = add_gc(v0,n=(0,2))
        v1 = add_gc(v1,n=(0,2))
        
        order = 3    # use cubic splines
        x0 = np.array([1,2,3,4])
        s0 = InterpolatedUnivariateSpline(x0, y0, k=order)
        s1 = InterpolatedUnivariateSpline(x0, y1, k=order)
        
        # set ghost cells
        v0[-2] = s0(5)
        v0[-1] = s0(6)
     
        v1[-2] = s1(5)
        v1[-1] = s1(6)
        
        # get pieces
        w0_list = get_pieces(v0)
        w1_list = get_pieces(v1)
        
        n0 = len(w0_list)    
        n1 = len(w1_list)

        count = 0
        restrict = 0      # a flag to remove ghost cells later
        gl = 0
        gr = 0
        
        while (n0 != n1):
            # IF the number of pieces are different
            # match the odd one out with the support in the ghost cells
            # (should not happen if signature condition is satisfied)
            
            if verbose:
                print(\
                '= no. of pieces: (n0,n1)= ({:2d},{:2d})'.format(n0,n1))
            
            if (verbose and (n1 != n0)):
                print('*** WARNING: signature condition violated ***')
            
            if n1 < n0:
                if verbose:
                    print('= swapping v0 and v1..')
                
                n2 = n0
                n0 = n1
                n1 = n2
                
                w2_list = w0_list
                w0_list = w1_list
                w1_list = w2_list
                
                v2 = v0
                v0 = v1
                v1 = v2
                
                alpha = 1. - alpha
                if verbose:
                    print(\
                   '= new no. of pieces: (n0,n1)= ({:2d},{:2d})'.format(n0,n1))
            
            # get medians of the supports
            spos0 = np.zeros(n0)
            spos1 = np.zeros(n1)
            
            for j,w in enumerate(w0_list):
                spos0[j] = get_spos(w)
            
            for j,w in enumerate(w1_list):
                spos1[j] = get_spos(w)
            
            # compute distance matrix
            dist_mat = np.zeros((n0,n1))
            for j0 in range(n0):
                for j1 in range(n1):
                    dist_mat[j0,j1] = np.abs(spos0[j0] - spos1[j1])
                    
            # match supports that are closer
            adj_mat  = np.zeros((n0,n1),dtype='bool')
            for j0 in range(n0):
                i = int(np.argmin(dist_mat[j0,:]))
                adj_mat[j0,i] = True
            
            ax0 = 0
            vn0 = v0
            vn1 = v1
            spos = spos1
            
            ii = ~adj_mat.sum(axis=ax0,dtype='bool')
            
            if verbose:
                print(dist_mat)
                print(adj_mat)
                print(ii)     
            
            S = np.mean(vn0)
            if S < 1e-5:
                S = 1e-5
            
            if n0 > 0:
                spos_ii = spos[ii]
                if len(spos_ii) > 1:
                    spos_ii = spos_ii[0]
            else:
                # edge case when one vector is null
                if verbose:
                    print('*** WARNING: one vector is null ! ***')
                spos_ii = spos[0]
            
            if verbose:
                print(spos_ii)            

            if float(spos_ii) < (float(len(w)) / 2.):
                vn0 = add_gc(vn0,n=(2,0))
                vn0[0] = 1.
                vn1 = add_gc(vn1,n=(2,0))
                gl += 1
            else:
                vn0 = add_gc(vn0,n=(0,2))
                vn0[-1] = 1.
                vn1 = add_gc(vn1,n=(0,2))
                gr += 1
        
            restrict += 1
            
            # get parts
            v0 = vn0
            v1 = vn1

            w0_list = get_pieces(v0)
            w1_list = get_pieces(v1)

            n0 = len(w0_list)    
            n1 = len(w1_list)

            if verbose:
                print('added no of pieces: n0,n1 = {:2d},{:2d}'.format(n0,n1) \
                  + '\n                    gl,gr = {:2d},{:2d}'.format(gl,gr))

            count += 1
            if count > 5:
                print('*** WARNING: missing-piece unresolved ***')
                break
    
        g_list = []
        s_list = []
        
        for j in range(len(w0_list)):
            
            w0 = w0_list[j]
            w1 = w1_list[j]
                
            if np.linalg.norm(w0 - w1) < tol:
                # if the two vectors are same: do nothing.
                h = w0
                
            else:
                x = np.linspace(0.,1.,len(w0))  # dummy domain
                
                # compute mass
                s0 = np.sum(w0)
                s1 = np.sum(w1)
                    
                # perform displacement interpolation
                # DR: need not be computed every time... can be stored
                x1,x2,Fv = dinterp.computeCDF(x,w0/s0,w1/s1);    
                xx,g = dinterp.dinterp(x,x1,x2,Fv,alpha);
                    
                s = (1. - alpha)*s0 + alpha*s1
                if np.sum(g) > 0.:
                    h = s*g/np.sum(g)
                else:
                    h = s*w0
                
            # remove ghost cells
            if restrict > 0:
                h = del_gc(h,n=(2*gl,2*gr))
    
            # remove right ghost cells
            h = del_gc(h,n=(0,2))
            if 0:
                f,axes = pl.subplots(nrows=3,figsize=(10,6))
                axes[0].plot(w0)
                axes[1].plot(w1)   
                axes[2].plot(h)
                axes[2].set_title(str(s))
                f.tight_layout()
            
            g_list.append(h)
    
        return g_list

    
    def _plot1(self):
        """
        TODO
        
        """
        
        # plot each piece
        fig,axes = pl.subplots(nrows=2,ncols=1,figsize=(8,4))
        supports_list = []
        
        p_list = get_pieces(du00p)
        axes[0].set_title('+ part')
        for w in p_list:
            axes[0].plot(x,w)
        
        for w in p_list:
            supports_list.append(w > 0.)
            x0 = x[w > 0.]
            zz = x0*0.
            axes[0].plot(x0,zz,'sr-',linewidth=2.,markersize=2.)
        
        p_list = get_pieces(-du00m)
        axes[1].set_title('- part')
        for w in p_list:
            supports_list.append(w > 0.)
            axes[1].plot(x,-w)
            x0 = x[w > 0.]
            zz = x0*0.
            axes[1].plot(x0,zz,'sr-',linewidth=2.,markersize=2.)
        
        fig.tight_layout()


    def _dinterp_all(self,v0p,v1p,v0m,v1m,alpha):
        r"""
        deprecated? (TODO)

        """

        dinterp_parts = self._dinterp_parts
        
        gp_list = dinterp_parts( v0p, v1p,alpha)
        gm_list = dinterp_parts(-v0m,-v1m,alpha)
        
        for k,gm in enumerate(gm_list):
            gm_list[k] *= -1.
        
        return gp_list + gm_list


    def _get_cell(self,t,ts):
        r"""
        return array-index with the largest entry
        
        """
        N = len(ts)
        for j in range(N):
            if t < ts[j]:
                return j-1
        return 0

    
    def _get_W(self,v0,v1,alphas,offsets=(0.,0.),\
               return_pieces=True,return_svd=True,plot_parts=False):
        r"""
        construct local basis using displacement interpolation and then 
        performing SVD on the snapshot matrix
 
        :Inputs:
        - v0,v1: two 1D arrays
        - alphas: list of doubles bw 0 & 1, disp. interpolation parameters
        - offsets: offsets to use for disp. interpolation
        - return_pieces: return *pieces* of the displacement interpolation
        - return_svd: return the SVD of the interpolated results
        - plot_parts: plot individual parts (for debugging)

        """
    
        dinterp_alphas = self._dinterp_alphas
        get_parts = self._get_parts
        get_pieces = self._get_pieces
        diffp = self._diffp
        
        # perform displacement interpolation by parts
        us = dinterp_alphas(v0,v1,alphas,offsets=offsets)

        
        if return_pieces:
            # return individual parts by post-processing us
            us_pieces = []      # list to contain interpolated pieces
            for k,u in enumerate(us):
                dup,dum = get_parts(diffp(u))   # obtain parts
                dup_list = get_pieces(dup)      # positive pieces
                dum_list = get_pieces(-dum)     # negative pieces
                
                for dup in dup_list:
                    supp = 1.*(np.abs(dup) > 0.)
                    us_pieces.append(np.cumsum(dup)*supp)
                    if plot_parts:
                        pl.plot(np.cumsum(dup)*supp)
                        pl.savefig('_plots/dup_' + str(k))
                        pl.close()
                
                for dum in dum_list:
                    supp = 1.*(np.abs(dum) > 0.)
                    us_pieces.append(-np.cumsum(dum)*supp)
                    if plot_parts:
                        pl.plot(np.cumsum(dum)*supp)
                        pl.savefig('_plots/dum_' + str(k))
                        pl.close()
            dim = v0.shape
            us_pieces.append(np.ones(dim))    # append constant fctns
            for k in range(1,8):
                ubc0 = np.zeros(dim)
                ubc0[:k] = 1.
                us_pieces.append(ubc0.copy())  # append fctns for left-BCs
            
            us_pieces.append(np.ones(v0.shape))
            us = np.array(us_pieces).T     
            
        else:
            dim = v0.shape
            us.append(np.ones(dim))    # append constant fctns
            for k in range(1,8):
                ubc0 = np.zeros(dim)
                ubc0[:k] = 1.
                us.append(ubc0.copy())  # append fctns for left-BCs
            us = np.array(us).T
            
        if return_svd:
            U,s,V = np.linalg.svd(us,full_matrices=0)
            return U,s,V,us
        else:
            return us


    def _dinterp_alphas(self,v0,v1,alphas,\
                       offsets=(0.,0.),return_pieces=True,return_int=True):
        r"""

        Perform disp. interpolation by parts in bulk for a given list of
        interpolation parameters alphas

        :Inputs:
        - v0,v1: two 1D arrays to use for disp. interpolation
        - alphas: list of doubles bw 0 & 1, disp. interpolation parameters
        - offsets: add a weighted constant before cumulative sum
        - return_pieces: return individual pieces of the interpolation

        """

        get_parts = self._get_parts
        dinterp_all = self._dinterp_all
        
        v0p,v0m = get_parts(v0)
        v1p,v1m = get_parts(v1)
        us = []
        
        for alpha in alphas:
            g_list = dinterp_all(v0p,v1p,v0m,v1m,alpha)    # can be made faster
            offset = (1-alpha)*offsets[0] + alpha*offsets[1]
            h = np.zeros(v0.shape)
            if return_pieces:
                for g0 in g_list:
                    if return_int:
                        us.append(np.cumsum(g0))
                    else:
                        us.append(g0)
            else:
                for g0 in g_list:
                    if return_int:
                        h += np.cumsum(g0)
                    else:
                        h += g0
                    h[:2] += offset
                us.append(h)
        return us


    def _precompute_flux_rom(self,U):
        r"""
        Using local basis, precompute flux difference 
        
        """
        # pre-compute flux
        N = U.shape[0]
        m = U.shape[1]
        F = - .5*(np.einsum('ij,ik,il->jkl',U[1:,:],U[1:,:],U[1:,:])  \
                -  np.einsum('ij,ik,il->jkl',U[1:,:],U[:(N-1),:],U[:(N-1),:])) \
             - .5*(np.einsum('j,k,l->jkl',U[0,:],U[1,:],U[1,:]) \
                -  np.einsum('j,k,l->jkl',U[0,:],U[0,:],U[0,:])) 
        return F


    def _precompute_src_rom(self,U,p=20):
        r"""
        Using local basis, precompute sources
        
        """

        s0 = 0.02    
        N = U.shape[0]      # HFM dof
        m = U.shape[1]      # ROM dof
        
        z = np.linspace(0.,100.,N) 
        V = np.zeros((m,p))
        w = np.ones(N)
    
        for j in range(1,p):
            w *= z / float(j)
            V[:,j] = s0*np.dot(U.T,w)
    
        V[:,0] = s0*np.dot(U.T,np.ones(N))
        
        return V


    def _rom_time_step(self,r,F,src,dt,mu1,bc):
        r"""
        update time-step in Galerkin-ROM

        """
        
        h = self._h
        p = src.shape[1]
        mu1_array = np.array([mu1 ** j for j in range(p)])
        r = np.array(r).flatten()
        dr = dt/h*np.dot(np.dot(F,r),r) + dt*np.dot(src,mu1_array)
        bc = bc / np.linalg.norm(bc)
        dr = dr - np.dot(dr,bc)*bc  # apply bc
        r = r + dr
        
        return r


    def plot_triangulation(self):
        r"""
        Plot Delaunay triangulation in the parameter space

        """

        tri = self._rom_tri
        dpi = self._dpi
        
        nsimplex= tri.nsimplex        # no of triangles
        simplices = tri.simplices     # nodes of each triangle
        points = tri.points           # coordinates of vertices
        f, ax0 = pl.subplots(nrows=1,ncols=1)
        
        for j in range(nsimplex):
            
            inodes = simplices[j,[0,1,2,0]]
            
            xcoords = points[inodes,0]
            ycoords = points[inodes,1]
            ax0.plot(xcoords, ycoords,'-b.')
    
        ax0.set_xlabel('$\mu_1$')
        ax0.set_ylabel('$\mu_2$')

        f.savefig('_plots/triangulation.png',dpi=dpi)
        pl.close(f)



    def _get_barycentric(self,tri,x):
        r"""
        Obtain Barycentric coordinates in tri for the given point

        :Input:
        tri: Deluanay triangulation (scipy)
        x: (2,)-array of doubles
        
        :Output:
        v: (2,)-array of doubles containing Barycentric coordinates

        """
        
        ndim = tri.ndim
        n = int(tri.find_simplex(x))    # find simplex number
        T = tri.transform[n,:ndim,:ndim]
        r = tri.transform[n,ndim,:]
            
        c = np.dot(T, x-r)
        
        return c


    def _get_spatial_coords(self,tri,n,c):
        r"""
        Obtain spatial coordinates in tri for the given Barycentric coord

        :Input:
        tri: Deluanay triangulation (scipy)
        n: element number
        c: (2,)-array of doubles containing Barycentric coordinate

        :Output:
        x: (2,)-array of doubles containing spatial coordinate

        """
    
        ndim = tri.ndim
        T = tri.transform[n,:ndim,:ndim]
        r = tri.transform[n,ndim,:]
        
        x = np.linalg.solve(T,c) + r

        return x

    
    def _get_uniform_samples(self,n):
        
        x = np.linspace(0.,1.,n)
        y = np.linspace(0.,1.,n)
        x0,y0 = np.meshgrid(x,y)
        x0 = x0.flatten()
        y0 = y0.flatten()
        
        return np.array([x0[(x0 + y0) <= 1.],y0[(x0 + y0) <= 1.]])    

    
    def _get_sides(self,tri,n):
        
        simplex = tri.simplices[n,:]
        side0 = [simplex[2], simplex[0]]
        side1 = [simplex[2], simplex[1]]
        
        return side0,side1

    
    def _get_time_index(self,j,times_list):
        l0 = next((i for i,i0 in enumerate(times_list) if j <i0),\
                   len(times_list)) - 1
        if 0:
            print(str(j) + ' , interval [' + str(times_list[l0]) + ', ' + str(times_list[l0+1]) + ']')
        return max([l0,0])
    
    
    
    def _dinterp2D(self,u10,u01,u00,c,offset=None,return_pieces=False):
        r'''
        displacement interpolation in 2D parameter space, computed by
        performing 2 disp. interpolations
        
        :Input:
        u10,u01,u00: nodes (vertices) of the parameter-triangle
        c: is a (2,)-array with barycentric coordinates
        
        '''

        diffp = self._diffp                     # differentiate 
        dinterp_alphas = self._dinterp_alphas   # disp. interp in bulk

        v00 = diffp(u00)
        v10 = diffp(u10)
        v01 = diffp(u01)
        
        if offset == None:
            offset = (1. - c[0] - c[1])*u00[0] + c[0]*u10[0] + c[1]*u01[0]
        
        alphas = [c[0]]
        g_list = dinterp_alphas(v00,v10,alphas,return_pieces=False,\
                                               return_int=False)
        va0 = g_list[0]
        
        g_list = dinterp_alphas(v01,v10,alphas,return_pieces=False,\
                                               return_int=False)
        va1 = g_list[0]
        
        if c[0] < 1.:
            alphas = [c[1] *(1. - c[0])]
        else:
            # trivial case
            alphas = [0.]
        g_list = dinterp_alphas(va0,va1,alphas,return_int=True,\
                                return_pieces=return_pieces)
        if not return_pieces:
            uaa = g_list[0]
            uaa += offset
            return uaa
        else:
            return g_list



    def _dinterp2D_tri(self,tri,u_list,tm_list,mu):
        r"""
        Perform displacement interpolation in 2D parameter space
        over a triangle over give time

        :Inputs:
        - tri: Delunay triangulation
        - u_list: HFM solutions corresponding to the given triangle
        - tm_list: list of time-steps to interpolate over

        """
        
        uaa_snapshot = []

        get_barycentric = self._get_barycentric
        get_sides = self._get_sides
        
        n = tri.find_simplex(mu)
        c = get_barycentric(tri,mu)
        c = np.abs(c)
        side0,side1 = get_sides(tri,n)
    
        for j in range(len(tm_list)):
    
            tm = tm_list[j]
            
            u00 = np.array(u_list[side0[0]][:,tm]).flatten()
            u10 = np.array(u_list[side0[1]][:,tm]).flatten()
            u01 = np.array(u_list[side1[1]][:,tm]).flatten()
                
            uaa = self._dinterp2D(u10,u01,u00,c,offset=0.,return_pieces=False)
            uaa_snapshot.append(uaa)
        
        uaa_snapshot = np.array(uaa_snapshot).T
        return uaa_snapshot


    
    def build_basis(self,tol=1e-14,Tfinal=50,max_basis_size=200,\
                    t_interval=5,P=5,Pt=5,p=40,save_snapshot=False,\
                    verbose=False):
        r'''
        Construct local basis, by applying disp. interpolation over each
        local parameter-time space element and performing SVD

        :Input:

        - tol: tolerance level for truncating singular values in percentage
        - p: no. of terms to keep in the Taylor series expansion of source term
        - P: no. of unformly-spaced samples per element in parameter
        - Pt: no. of unformly-spaced samples per element in time
        - t_interval: interval in the parameter-time element
        - max_basis_size: threshold the maximum number of basis
        - save_snapshot: store snapshot matrix (pre-SVD)
        - Tfinal: final time-step

        '''


        # import internal functions
        diffp = self._diffp
        dinterp2D_tri = self._dinterp2D_tri
        get_W = self._get_W
        get_uniform_samples = self._get_uniform_samples
        get_barycentric = self._get_barycentric
        get_spatial_coords = self._get_spatial_coords
        precompute_flux_rom = self._precompute_flux_rom
        precompute_src_rom = self._precompute_src_rom

        # set up Delaunay triangulation
        mu0_list = self._hfm_mu0_list
        mu1_list = self._hfm_mu1_list
        
        Mu1,Mu0 = np.meshgrid(mu1_list,mu0_list)
        train_pts = np.concatenate(([Mu0.flatten()],[Mu1.flatten()]),axis=0).T
        tri = Delaunay(train_pts)
        self._rom_tri = tri
        
        self._rom_P = P
        z = get_uniform_samples(P)    # get uniformly-spaced samples

        # set up time-intervals for sampling
        time_index_list = [1,2] + range(3,Tfinal,t_interval)
        self._time_index_list = time_index_list
        
        if verbose:
            # plot barycentric grid-points z
            fig,axes = pl.subplots(ncols=2,figsize=(10,4));
            axes[0].plot(z[0,:],z[1,:],'.');
            # plot parameter points in mu-plane
            for mu0 in mu_list:
                axes[1].plot(mu0[0],mu0[1],'r.');
            fig.savefig('_plots/sampled_parameter_pts.png')
        
        u_list = self._hfm_u_list      # get HFM solutions
        alphas = np.linspace(0.,1.,Pt)  # number of points sampled in time

        for n in range(tri.nsimplex):
            
            # get mu coordinates
            mu_list = []    # mu values for each sample points in z
            for j in range(z.shape[1]):
                c = np.array([z[0,j],z[1,j]])
                mu0 = get_spatial_coords(tri,n,c)
                mu_list.append(mu0)
            
            basis_list = []         # left singular vectors 
                                    # saved to external file unless specified
            basis_raw_list = []     # raw basis (not stored unless specified)
            s_list = []             # singular values at each slab
            M_list   = []           # number of basis kept at each slab
            basis_all = []          # all basis for the param-time-slab

            for j in range(len(time_index_list)-1):
                
                basis_j = []    # basis for time-slice
                sys.stdout.write('\r({:1d}) constructing basis,  '.format(n)\
                      +'     time-interval: {:4d}'.format(j) + ' '*25)
                sys.stdout.flush()
                
                for mu in mu_list:
                    
                    # TODO interp2D_tri also searches for the triangle for mu
                    uaa_snapshot = dinterp2D_tri(tri,u_list,time_index_list[j:(j+2)],mu)
                    uaa_j0 = uaa_snapshot[:,0]
                    uaa_j1 = uaa_snapshot[:,1]
                    
                    v0 = diffp(uaa_j0)
                    v1 = diffp(uaa_j1)
                    
                    us = get_W(v0,v1,alphas,\
                               return_pieces=True, return_svd=False)
                    basis_j.append(us)
                
                basis_all.append(basis_j)

            M0 = len(time_index_list)-1
            for j in range(M0):
            
                if j == 0:
                    basis_j3 = basis_all[j] + basis_all[j+1]
                elif ((j > 0) and (j < M0-1)):
                    basis_j3 = basis_all[j-1] + basis_all[j] + basis_all[j+1]
                elif j == M0-1:
                    basis_j3 = basis_all[j-1] + basis_all[j] 
                else:
                    basis_j3 = basis_all[j] 
                
                basis_array_j = np.concatenate(basis_j3, axis=1)
                U,s,V = np.linalg.svd(basis_array_j,full_matrices=0)
                s_list.append(s)
                M_k = \
                    next((i for i,si in enumerate(s/s[0]) if si < tol),\
                          max_basis_size)
                M_list.append(M_k)
                Usmall = U[:,:M_k] 
                basis_list.append(Usmall)
                
                if save_snapshot:
                    basis_raw_list.append(basis_array_j)

            self._rom_basis.append(basis_list)
            if save_snapshot:
                self._rom_basis_raw.append(basis_raw_list)

            # off-line computations for flux and source
            F_list   = []
            src_list = []
            bc_list  = []
            
            K = len(basis_list)-1
            maxM = 1
            for k,U in enumerate(basis_list):
                
                M_k = M_list[k]
                if M_k > maxM:
                    maxM = M_k
                sys.stdout.write('\r({:1d}) constructing flux,'.format(n)\
                         + 'src     time-interval: {:4d}/{:4d}'.format(k,K)\
                                + ' [d {:3d}]'.format(maxM))
                sys.stdout.flush()
                
                # compute reduced basis 
                F = precompute_flux_rom(U[:,:M_k])
                F_list.append(F)
            
                # compute source term
                src = precompute_src_rom(U[:,:M_k],p=p)
                src_list.append(src)

                # compute boundary conditions
                bc_list.append( U[0,:M_k] )

            #TODO: following 4 lines necessary?
            self._rom_F.append(F_list)
            self._rom_M.append(M_list)
            self._rom_bc.append(bc_list)
            self._rom_src.append(src_list)

            # compute transition matrix
            transition_list = []
            
            for k in range(len(basis_list)-1):
            
                s0 = s_list[k]
            
                M_k   = M_list[k]
                M_kp1 = M_list[k+1]
            
                U_k   = basis_list[k]
                U_kp1 = basis_list[k+1]
            
                transition_list.append(\
                             np.dot(U_kp1[:,:M_kp1].T,U_k[:,:M_k]) )

            self._rom_T.append(transition_list)

            # save generated lists to output files    
            L = len(basis_list)
            for k in range(L):
                m = L - 1 - k
                
                Usmall = basis_list.pop()
                F = F_list.pop()
                src = src_list.pop()
                bc = bc_list.pop()
                
                basis_fname = \
                        '_output/basis_' +str(n)+ '_' +str(m)+ '.npy'
                F_fname = \
                        '_output/F_' +str(n)+ '_' +str(m)+ '.npy'
                src_fname = \
                        '_output/src_' +str(n)+ '_' +str(m)+ '.npy'
                bc_fname = \
                        '_output/bc_' +str(n)+ '_' +str(m)+ '.npy'
                
                np.save(basis_fname, Usmall)
                np.save(F_fname, F)
                np.save(src_fname, src)
                np.save(bc_fname, bc)
            
            for k in range(L-1):
                m = L-1 - k
                T = transition_list.pop()
                T_fname = \
                        '_output/transition_'+str(n)+'_'+str(m) + '.npy'
                np.save(T_fname, T)

            M_fname = '_output/M_' + str(n) + '.npy'
            np.save(M_fname,M_list)

        sys.stdout.write('\r= done.\n')
        sys.stdout.flush()

    def _mc_sample(self,nsols=1000,M0=4000):
        r"""
        
        further reduction of the model based on MC sampling

        :Inputs:
        nsols: number of solutions for each triangular element
        M0:    final time-step
        
        """

        np.random.seed(12345)

        M0 = len(self._time_index_list)-1

        # generate samples
        for n in range(self._rom_tri.nsimplex):
            j = 0
            random_mu_list = []
            random_mu = self.random_mu
            rs_list = []

            while(len(random_mu_list) < nsols):
                # sample random variable
                mu0 = random_mu()
                
                n0 = int(self._rom_tri.find_simplex(mu0))
                if n0 == n:
                    random_mu_list.append(mu0)
            
                    self.run_rom(mu=mu0,frugal=True,M0=M0,evaluate=False)
                    rs_list.append(copy(self._rom_r_list))
                    sys.stdout.write('\r({:1d})'.format(n)\
                                   + ' running ROM, sample {:5d}..'.format(j+1))
                    sys.stdout.flush()
                    j += 1
            
            mu_fname = '_output/sampled_mu_' + str(n) + '.npy'
            sols_fname = '_output/sampled_rom_sols_' + str(n) + '.npy'
            np.save(mu_fname,random_mu_list)
            np.save(sols_fname,rs_list)
            self._rom_set_up = False


    def print_prog(text):

        sys.stdout.write('\r')
        sys.stdout.write(text)
        sys.stdout.flush()

    
    def _dot3d(self,A,B,C,D):
        r"""
        compute A_{i,j,k} B_{i,I} C_{j,J} D_{k,K}
        resulting in: an array E of size {I,J,K}
        
        """

        E = np.dot(A,D)
        E = np.transpose(np.dot(np.transpose(E,(0,2,1)),C),(0,2,1))
        E = np.transpose(np.dot(np.transpose(E,(2,1,0)),C),(2,1,0))

        return E

    
    def reduce_basis(self,tol=1e-6,\
                     max_nbasis=150,M0=400,nsols=1000):
        r"""
        compute POD reduced basis for each local element using sampled basis
        functions

        :Inputs:
        - M0: maximum time-intervals
        - max_time-step: maximum no. of time-steps allowed
        - max_nbasis: maximum no. of basis allowed
        - tol: tolerance for truncating singular values (%)
        - nsols: number of solutions to sample for each parameter-triangle 
        
        """

        self._mc_sample(nsols=nsols,M0=M0)
        N = self._rom_tri.nsimplex
        til = self._time_index_list.tolist()
        M0 = len(til)-1
        
        for n in range(N):
            # compute reduction
            mu_fname = '_output/sampled_mu_'+str(n)+'.npy'
            sols_fname = '_output/sampled_rom_sols_'+str(n)+'.npy'

            M_fname = '_output/M_'+str(n)+'.npy'
            M_list = np.load(M_fname)
            snapshot_list = np.load(sols_fname)
            #M = snapshot_list.shape[1]      # number of time intervals
            rM_list = []
            sys.stdout.write('\r({:1d}) computing reduced basis ...'.format(n)\
                             + ' '*20)
            sys.stdout.flush()
            
            #print('max_time_step = ' + str(max_time_step))
            
            for m in range(M0):
                
                m1 = M_list[m]
                sampled_sols_list = []
                
                # gather snapshots in the time interval 
                for i in range(til[m],til[m+1]):
                    for k in range(len(snapshot_list)):
                        if ((i-1) < len(snapshot_list[k])):
                            new_array = np.array([snapshot_list[k][i-1].T]).T
                            sampled_sols_list.append(new_array)
                
                if len(sampled_sols_list) == 0:
                    break
                #TODO rm print([a.shape for a in sampled_sols_list])
                sampled_sols = np.concatenate(sampled_sols_list,axis=1)
                
                #M0 = sampled_sols.shape[1]      # number of "generated" bases
                W,s,V = np.linalg.svd(sampled_sols,full_matrices=0)

                # compute reduced basis by truncating singular values,
                # store no. of reduced basis
                m0 = \
                    next((i for i,si in enumerate(s/s[0]) if si < tol),\
                         max_nbasis)
                rM_list.append(m0)  
                W = W[:,:m0]
                
                # compute basis
                basis_fname = \
                        '_output/basis_' +str(n)+ '_' +str(m)+ '.npy'
                basis = np.load(basis_fname)
                rbasis = np.dot(basis[:,:m1],W)
                rbasis_fname = \
                        '_output/rbasis_' +str(n)+ '_' +str(m)+ '.npy'
                np.save(rbasis_fname,rbasis)

                # reduce flux
                sys.stdout.write('\r({:1d}) constructing flux '.format(n)\
                         + '        time-interval: {:4d}/{:4d}'.format(m,M0)\
                                + ' [d {:3d}]'.format(m0))
                sys.stdout.flush()
                F_fname = \
                        '_output/F_' +str(n)+ '_' +str(m)+ '.npy'
                F = np.load(F_fname)
                rF = self._dot3d(F,W,W,W)   # faster than einsum...
                
                rF_fname = \
                        '_output/rF_' +str(n)+ '_' +str(m)+ '.npy'
                np.save(rF_fname,rF)
                
                # reduce source
                sys.stdout.write('\r({:1d}) constructing src  '.format(n)\
                         + '        time-interval: {:4d}/{:4d}'.format(m,M0)\
                                + ' [d {:3d}]'.format(m0))
                sys.stdout.flush()
                src_fname = \
                        '_output/src_' +str(n)+ '_' +str(m)+ '.npy'
                src = np.load(src_fname)
                p = src.shape[1]
                rsrc = np.dot(W.T,src)
                rsrc_fname = \
                        '_output/rsrc_' +str(n)+ '_' +str(m)+ '.npy'
                np.save(rsrc_fname,rsrc)
                
                # reduce BCs
                sys.stdout.write('\r({:1d}) constructing BC   '.format(n)\
                         + '        time-interval: {:4d}/{:4d}'.format(m,M0)\
                                + ' [d {:3d}]'.format(m0))
                sys.stdout.flush()
                rbc = rbasis[0,:m0]
                rbc /= np.linalg.norm(rbc)
                rbc_fname = \
                        '_output/rbc_' +str(n)+ '_' +str(m)+ '.npy'
                np.save(rbc_fname,rbc)

                # reduce transition list
                if (m > 0):
                    sys.stdout.write('\r({:1d}) constructing trans'.format(n)\
                             + 'ition   time-interval: {:4d}/{:4d}'.format(m,M0)\
                                    + ' [d {:3d}]'.format(m0))
                    sys.stdout.flush()
                    T_fname =  \
                        '_output/transition_'+str(n)+'_'+str(m) + '.npy'
                    T = np.load(T_fname)
                    Tw = np.dot(T,W_old)
                    rT = np.dot(W.T,Tw)
                    rT_fname =  \
                        '_output/rtransition_'+str(n)+'_'+str(m) + '.npy'
                    np.save(rT_fname,rT)
                W_old = W                    
                sys.stdout.flush()
        
            rM_fname = \
                       '_output/rM_' + str(n) + '.npy'
            np.save(rM_fname, rM_list)
        self._rom_reduced = True
    

    def random_mu(self):
        mu = np.random.rand(2)
        mu[0] = mu[0]*(9. - 3.) + 3.
        mu[1] = mu[1]*(0.075 - 0.02) + 0.02
        return mu


    def _set_up_rom(self,reset=False,load_basis=False,\
                         simplex_list=None, time_interval_list=None, \
                         M0=100,verbose=False):
        r"""
        
        Load bases precomputed flux, src, transition_list into memory

        :Inputs:
        - reset: force reload basis functions and store into memory
        - load_basis: load the high-dimensional basis vectors in to memory
        - time_interval_list: restrict to given time interval
        - M0: final time-step
    
        """

        if reset:
            self._rom_set_up = False

        if not self._rom_set_up:
            
            if self._rom_reduced:
                prefix = 'r'
            else:
                prefix = ''
            
            K = self._rom_tri.nsimplex
            M0 = len(self._time_index_list)-1

            if simplex_list == None:
                k_list = range(K)
            else:
                k_list = simplex_list

            if time_interval_list == None:
                # not to be confused with usual time_index_list
                m_list = range(M0)
            else:
                m_list = time_interval_list

            self._rom_basis = []
            self._rom_F = []
            self._rom_src = []
            self._rom_bc = []
            self._rom_T = []
            self._rom_M = []
            
            for k in k_list:
                M_fname = '_output/' + prefix + \
                        'M_' + str(k) + '.npy'
                M_list = np.load(M_fname)
                transition_list = []
                F_list = []
                src_list = []
                basis_list = []
                bc_list = []
                for m in m_list:
                    basis_fname = '_output/' + prefix + \
                            'basis_' +str(k)+ '_' +str(m)+ '.npy'
                    F_fname = '_output/' + prefix + \
                            'F_' +str(k)+ '_' +str(m)+ '.npy'
                    src_fname = '_output/' + prefix + \
                            'src_' +str(k)+ '_' +str(m)+ '.npy'
                    bc_fname = '_output/' + prefix + \
                            'bc_' +str(k)+ '_' +str(m)+ '.npy'
                    T_fname = '_output/' + prefix + \
                            'transition_'+str(k)+'_'+str(m+1) + '.npy'
    
                    F = np.load(F_fname)
                    src = np.load(src_fname)
                    bc = np.load(bc_fname)
                    
                    # load basis only if told to evaluate
                    # TODO: .copy() necessary?
                    if (m == 0) or ((m > 0) and load_basis):
                        basis = np.load(basis_fname)
                        basis_list.append(basis.copy())
                    F_list.append(F.copy())
                    src_list.append(src.copy())
                    bc_list.append(bc.copy())
                    
                    # transition list of length M-2 
                    if (m < M0-1):
                        T = np.load(T_fname)
                        transition_list.append(T.copy())
                
                self._rom_M.append(M_list)
                self._rom_F.append(F_list)
                self._rom_src.append(src_list)
                self._rom_basis.append(basis_list)
                self._rom_bc.append(bc_list)
                self._rom_T.append(transition_list)
                
                self._rom_set_up = True
        else:
            if verbose:
                print('ROM already set up')
            pass


    
    def run_rom(self,mu=[7.5, 0.035],M0=None,\
                     evaluate=False,verbose=False,reread=False,\
                     frugal=False):
        r"""
        run reduced order model
        
        :Input:
        - mu: 2D-array [mu1,mu2]
        - M0: maximal number of time-intervals
        - evaluate: represent in HFM basis and store in self._rom_ra_list
        - frugal: set up and run rom for just one element
        
        """
        
        show_plots = False
        get_time_index = self._get_time_index        

        tri = self._rom_tri
        
        k = tri.find_simplex(mu)        # triangle number

        time_index_list = self._time_index_list
        L = len(time_index_list)
        
        if frugal:
            self._set_up_rom(simplex_list=[k],load_basis=evaluate)
            
            M_list = self._rom_M[0]
            F_list = self._rom_F[0]
            src_list = self._rom_src[0]
            basis_list = self._rom_basis[0]
            bc_list = self._rom_bc[0]
            transition_list = self._rom_T[0]
        else:
            self._set_up_rom(reset=reread,load_basis=evaluate,M0=M0)
    
            M_list = self._rom_M[k]
            F_list = self._rom_F[k]
            src_list = self._rom_src[k]
            basis_list = self._rom_basis[k]
            bc_list = self._rom_bc[k]
            transition_list = self._rom_T[k]
        
        
        rom_time_step = self._rom_time_step

        M_0 = M_list[0]
        U = basis_list[0][:,:M_0]

        # input parameters
        mu0 = mu[0]
        mu1 = mu[1]

        # set up initial condition
        x = self._x
        u0a = np.zeros(x.shape)
        u0a[0:2] = mu0   # mu0
        r0 = np.dot(U.T,u0a)
        r1 = np.array(r0).copy()       # redundant..?
        r_list = []

        sc = 1                  # TODO: remove
        dt = self._hfm_dt
        dt1 = dt/sc
        if M0 == None:
            M0 = len(time_index_list) - 1
            tfinal = time_index_list[M0]
        else:
            tfinal = time_index_list[M0]

        if evaluate:
            self._rom_ra_list = []
            self._rom_ra_list.append(np.dot(U,r1))
        
        i0_old = get_time_index(0, time_index_list)

        for j in range(1,tfinal):
            if verbose:
                sys.stdout.write('\r' + ' '*79)
                sys.stdout.write('\r({:1d}) running ROM,'.format(int(k))
                            + ' time-interval : {:04d}.. '.format(i0_old))
                sys.stdout.flush()
            i0 = get_time_index(j, time_index_list)
            if (i0_old < i0):
                i0_old = i0
                r1 = np.dot(transition_list[i0-1],r1)
            
            F   =   F_list[i0_old]
            src = src_list[i0_old]
            bc = bc_list[i0_old]
            r1 = rom_time_step(r1,F,src,dt,mu1,bc)
            r1 = np.array(r1).flatten() 
            
            r_list.append(r1.copy())
            
            if evaluate:
                if verbose:
                    sys.stdout.write('.. reconstruction ..')
                    sys.stdout.flush()
                M_k = M_list[i0_old]
                if self._rom_reduced:
                    fname = '_output/rbasis_'+str(k)+'_'+str(i0_old)+'.npy'
                else:
                    fname = '_output/basis_'+str(k)+'_'+str(i0_old)+'.npy'
                Usmall = basis_list[i0_old]
                ua = np.dot(Usmall,r1)
                self._rom_ra_list.append(ua)
                self._rom_ra = ua
                if verbose:
                    sys.stdout.write('.. done')
                    sys.stdout.flush()
            sys.stdout.flush()
        
        self._rom_r1 = r1
        self._rom_r_list = r_list
        self._rom_i0 = i0_old
        #if verbose:
            #print(' .. run complete.')


    def plot_rom_solution(self):

        ra_list = self._rom_ra_list
        L = len(ra_list)
        x = self._x
        dpi = self._dpi

        f,ax0 = pl.subplots(nrows=1,ncols=1,figsize=(10,3))
        for j in range(0,L,10):
            ra = ra_list[j]
            ax0.plot(x,ra,'royalblue')

        f.savefig('_plots/rom_solution.png',dpi=dpi)
        pl.close(f)


    def compare_rom_solution(self,mu,interval=10,\
                             sol_plot_fname='rom_solution.png',\
                             err_plot_fname='rom_error.png'):

        ra_list = self._rom_ra_list
        L = len(ra_list)
        x = self._x
        dpi = self._dpi

        f0,ax0 = pl.subplots(nrows=1,ncols=1,figsize=(10,3))
        f1,ax1 = pl.subplots(nrows=1,ncols=1,figsize=(10,3))
        u = self.run_hfm(mu)
        for j in range(0,L,interval):
            ra = ra_list[j]
            ax0.plot(x,ra,'royalblue')
            ax0.plot(x,u[:,j],'--r')
            
            ax1.plot(x,ra - u[:,j],'royalblue')

        k = self._rom_tri.find_simplex(mu)
        
        ax0.set_title('element = ' + str(k))
        ax1.set_title('element = ' + str(k))
        sol_plot_fname = os.path.join('_plots',sol_plot_fname)
        err_plot_fname = os.path.join('_plots',err_plot_fname )
        f0.savefig(sol_plot_fname ,dpi=dpi)
        f1.savefig(err_plot_fname ,dpi=dpi)
        
        pl.close(f0)
        pl.close(f1)


    def plot_hfm_solution(self):

        u_list = self._hfm_u_list
        N = len(u_list)
        M = self._hfm_M
        x = self._x
        k2mu_list = self._hfm_k2mu_list
        K = 150
        dpi = self._dpi

        f,ax0 = pl.subplots(nrows=1,ncols=1,figsize=(10,3))
        for k in range(N):
            uk = u_list[k]
            mu = k2mu_list[k]
            for j in range(0,M,K):
                ax0.plot(x,uk[:,j],'royalblue')
            ax0.set_title('$(\mu_1,\mu_2)$ = (' \
                            + str(mu[0]) + ',' + str(mu[1]) + ')')
            fname = os.path.join('_plots','sol_u_' + str(k) + '.png')
            f.savefig(fname,dpi=dpi)
            ax0.clear()
        pl.close(f)


    
    def save(self):
        r"""
        save ROM. 

        - class itself saved in _output/rom_file.pkl
        - more data are stored in _output/ROM_info.txt
        
        """
        
        import pickle
        self.save_as_pickle()
        
        # to save in text-file
        with open('_output/ROM_info.txt',mode='w') as text_file:
            text_file.write('N  = {:10d}\n'.format(self._N))
            text_file.write('h  = {:26.15f}\n'.format(self._h))
            text_file.write('M  = {:10d}\n'.format(self._hfm_M))
            text_file.write('dt = {:26.15f}\n'.format(self._hfm_dt))
            text_file.write('P  = {:10d}\n'.format(self._rom_P))

        # to save as separate files

        # spatial grid
        fname = '_output/x.npy'
        np.save(fname,self._x)

        # time_index_list
        fname = '_output/time_index.npy'
        np.save(fname,self._time_index_list)
        
        # pickle triangulation
        fname = '_output/tri.pkl'
        with open(fname, 'wb') as output:
            pickle.dump(self._rom_tri, output, pickle.HIGHEST_PROTOCOL)


    def load(self):
        r"""
        load data from stored ROM. The class itself should be loaded by
        unpickling e.g. _output/ROM_object.pkl
            
        """
        
        import pickle

        with open('_output/ROM_info.txt',mode='r') as input_file:
            for row in input_file:
                val = row.split('=')[1]
                if 'N' == row.split('=')[0][0]:
                    N = int(val)
                if 'M' == row.split('=')[0][0]:
                    M = int(val)
                elif 'h' == row.split('=')[0][0]:
                    h = float(val)
                elif 'dt' == row.split('=')[0][0:2]:
                    dt = float(val)
                elif 'P' == row.split('=')[0][0]:
                    P = int(val)

        self._N = N
        self._h = h
        self._hfm_M = M
        self._hfm_dt = dt
        self._rom_P = P

        # spatial grid
        fname = '_output/x.npy'
        self._x = np.load(fname)

        # time_index_list
        fname = '_output/time_index.npy'
        self._time_index_list = np.load(fname)
        
        # pickle triangulation
        fname = '_output/tri.pkl'
        with open(fname, 'r') as input_pkl:
            self._rom_tri = pickle.load(input_pkl)


    def save_as_pickle(self,fname='_output/ROM_object.pkl'):

        import pickle

        with open(fname, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        
