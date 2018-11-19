
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy import abs
from numpy import argmin
from numpy import array
from numpy import dot
from numpy import hstack
from numpy import isnan
from numpy import max
from numpy import min
from numpy import newaxis
from numpy import sqrt
from numpy import sum
from numpy import vstack
from numpy import zeros
from numpy.linalg import pinv
from numpy.random import rand

from scipy.linalg import svd
from scipy.optimize import fmin_slsqp
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from compas_tna.diagrams import FormDiagram

from compas.numerical import connectivity_matrix
from compas.numerical import devo_numpy
from compas.numerical import equilibrium_matrix
from compas.numerical import normrow
from compas.numerical import nonpivots
from compas.plotters import MeshPlotter
from compas.utilities import geometric_key
from compas.viewers import VtkViewer

from multiprocessing import Pool
from random import shuffle

import sympy


__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


__all__ = [
    'optimise_single',
    'optimise_multi',
    'plot_form',
    'view_form',
    'randomise_form',
]


def optimise_single(form, solver='devo', polish='slsqp', qmin=1e-6, qmax=5, population=300, generations=500,
                    printout=10, tol=0.001, plot=False, frange=[], indset=None, tension=False, planar=False):

    """ Finds the optimised load-path for a FormDiagram.

    Parameters
    ----------
    form : obj
        The FormDiagram.
    solver : str
        Differential Evolution 'devo' or Genetic Algorithm 'ga' evolutionary solver to use.
    polish : str
        'slsqp' polish or None.
    qmin : float
        Minimum qid value.
    qmax : float
        Maximum qid value.
    population : int
        Number of agents for the evolution solver.
    generations : int
        Number of generations for the evolution solver.
    printout : int
        Frequency of print output to the terminal.
    tol : float
        Tolerance on horizontal force balance.
    plot : bool
        Plot progress of the evolution.
    frange : list
        Minimum and maximum function value to plot.
    indset : list
        Independent set to use.
    tension : bool
        Allow tension edge force densities (experimental).
    planar : bool
        Only consider the x-y plane.

    Returns
    -------
    float
        Optimum load-path value.
    list
        Optimum qids

    """

    if printout:
        print('\n' + '-' * 50)
        print('Load-path optimisation started')
        print('-' * 50)

    # Mapping

    k_i  = form.key_index()
    i_k  = form.index_key()
    i_uv = form.index_uv()
    uv_i = form.uv_index()

    # Vertices and edges

    n     = form.number_of_vertices()
    m     = form.number_of_edges()
    fixed = [k_i[key] for key in form.fixed()]
    rol   = [k_i[key] for key in form.vertices_where({'is_roller': True})]
    edges = [(k_i[u], k_i[v]) for u, v in form.edges()]
    sym   = [uv_i[uv] for uv in form.edges_where({'is_symmetry': True})]
    free  = list(set(range(n)) - set(fixed) - set(rol))

    # Co-ordinates and loads

    xyz = zeros((n, 3))
    x   = zeros((n, 1))
    y   = zeros((n, 1))
    z   = zeros((n, 1))
    px  = zeros((n, 1))
    py  = zeros((n, 1))
    pz  = zeros((n, 1))

    for key, vertex in form.vertex.items():
        i = k_i[key]
        xyz[i, :] = form.vertex_coordinates(key)
        x[i]  = vertex.get('x')
        y[i]  = vertex.get('y')
        px[i] = vertex.get('px', 0)
        py[i] = vertex.get('py', 0)
        pz[i] = vertex.get('pz', 0)

    xy = xyz[:, :2]
    px = px[free]
    py = py[free]
    pz = pz[free]

    # C and E matrices

    C   = connectivity_matrix(edges, 'csr')
    Ci  = C[:, free]
    Cf  = C[:, fixed]
    Cit = Ci.transpose()
    E   = equilibrium_matrix(C, xy, free, 'csr').toarray()
    uvw = C.dot(xyz)
    U   = uvw[:, 0]
    V   = uvw[:, 1]

    # Independent and dependent branches

    if indset:
        ind = []
        for u, v in form.edges():
            if geometric_key(form.edge_midpoint(u, v)[:2] + [0]) in indset:
                ind.append(uv_i[(u, v)])
    else:
        ind = nonpivots(sympy.Matrix(E).rref()[0].tolist())

    k   = len(ind)
    dep = list(set(range(m)) - set(ind))

    for u, v in form.edges():
        form.set_edge_attribute((u, v), 'is_ind', True if uv_i[(u, v)] in ind else False)

    if printout:
        _, s, _ = svd(E)
        print('Form diagram has {0} (RREF): {1} (SVD) independent branches '.format(len(ind), m - len(s)))

    # Set-up

    lh     = normrow(C.dot(xy))**2
    Edinv  = -csr_matrix(pinv(E[:, dep]))
    Ei     = E[:, ind]
    p      = vstack([px, py])
    q      = array([attr['q'] for u, v, attr in form.edges(True)])[:, newaxis]
    bounds = [[qmin, qmax]] * k
    args   = (q, ind, dep, Edinv, Ei, C, Ci, Cit, U, V, p, px, py, pz, tol, z, free, planar, lh, sym, tension)

    # Horizontal checks

    checked = True

    if tol == 0:
        for i in range(10**3):
            q[ind, 0] = rand(k) * qmax
            q[dep] = -Edinv.dot(p - Ei.dot(q[ind]))
            Rx = Cit.dot(U * q.ravel()) - px.ravel()
            Ry = Cit.dot(V * q.ravel()) - py.ravel()
            R  = max(sqrt(Rx**2 + Ry**2))
            if R > tol:
                checked = False
                break

    if checked:

        # Solve

        if solver == 'devo':
            fopt, qopt = _diff_evo(_fint, bounds, population, generations, printout, plot, frange, args)

        if polish == 'slsqp':
            fopt_, qopt_ = _slsqp(_fint_, qopt, bounds, printout, _fieq, args)
            q1 = _zlq_from_qid(qopt_, args)[2]
            if fopt_ < fopt:
                if (min(q1) > -0.001 and not tension) or tension:
                    fopt, qopt = fopt_, qopt_

        z, _, q, q_ = _zlq_from_qid(qopt, args)

        # Unique key

        gkeys = []
        for i in ind:
            u, v = i_uv[i]
            gkeys.append(geometric_key(form.edge_midpoint(u, v)[:2] + [0]))
        form.attributes['indset'] = gkeys

        # Update FormDiagram

        for i in range(n):
            key = i_k[i]
            form.set_vertex_attribute(key=key, name='z', value=float(z[i]))

        for c, qi in enumerate(list(q_.ravel())):
            u, v = i_uv[c]
            form.set_edge_attribute((u, v), 'q', float(qi))

        # Relax

        q    = array([attr['q'] for u, v, attr in form.edges(True)])
        Q    = diags(q)
        CitQ = Cit.dot(Q)
        Di   = CitQ.dot(Ci)
        Df   = CitQ.dot(Cf)
        bx   = px - Df.dot(x[fixed])
        by   = py - Df.dot(y[fixed])
        # bz   = pz - Df.dot(z[fixed])
        x[free, 0] = spsolve(Di, bx)
        y[free, 0] = spsolve(Di, by)
        # z[free, 0] = spsolve(Di, bz)

        for i in range(n):
            form.set_vertex_attributes(key=i_k[i], names='xyz', values=[float(j) for j in [x[i], y[i], z[i]]])

        fopt = 0
        for u, v in form.edges():
            if form.get_edge_attribute((u, v), 'is_symmetry') is False:
                qi = form.get_edge_attribute((u, v), 'q')
                li = form.edge_length(u, v)
                fopt += abs(qi) * li**2

        form.attributes['loadpath'] = fopt

        if printout:
            print('\n' + '-' * 50)
            print('qid range : {0:.3f} : {1:.3f}'.format(min(qopt), max(qopt)))
            print('q range   : {0:.3f} : {1:.3f}'.format(min(q), max(q)))
            print('fopt      : {0:.3f}'.format(fopt))
            print('-' * 50 + '\n')

        return fopt, qopt

    else:

        if printout:
            print('Horizontal equillibrium checks failed')

        return None, None


def _zlq_from_qid(qid, args):

    q, ind, dep, Edinv, Ei, C, Ci, Cit, U, V, p, px, py, pz, tol, z, free, planar, lh, sym, *_ = args
    q[ind, 0] = qid
    q[dep] = -Edinv.dot(p - Ei.dot(q[ind]))
    q_ = 1 * q
    q[sym] *= 0

    if not planar:
        z[free, 0] = spsolve(Cit.dot(diags(q.flatten())).dot(Ci), pz)
    l2 = lh + C.dot(z)**2

    return z, l2, q, q_


def _fint(qid, *args):

    q, ind, dep, Edinv, Ei, C, Ci, Cit, U, V, p, px, py, pz, tol, z, free, planar, lh, sym, tension = args

    z, l2, q, q_ = _zlq_from_qid(qid, args)
    f = dot(abs(q.transpose()), l2)

    if isnan(f):
        return 10**10

    else:

        if not tension:
            f += sum((q[q < 0] - 5)**4)

        Rx = Cit.dot(U * q_.ravel()) - px.ravel()
        Ry = Cit.dot(V * q_.ravel()) - py.ravel()
        Rh = Rx**2 + Ry**2
        Rm = max(sqrt(Rh))
        if Rm > tol:
            f += sum(Rh - tol + 5)**4

        return f


def _fint_(qid, *args):

    z, l2, q, q_ = _zlq_from_qid(qid, args)
    f = dot(abs(q.transpose()), l2)

    if isnan(f):
        return 10**10

    return f


def _fieq(qid, *args):

    q, ind, dep, Edinv, Ei, C, Ci, Cit, U, V, p, px, py, pz, tol, z, free, planar, lh, sym, *_ = args
    tension = args[-1]

    q[ind, 0] = qid
    q[dep] = -Edinv.dot(p - Ei.dot(q[ind]))
    q_ = 1 * q
    q[sym] *= 0

    Rx = Cit.dot(U * q_.ravel()) - px.ravel()
    Ry = Cit.dot(V * q_.ravel()) - py.ravel()
    Rh = Rx**2 + Ry**2
    Rm = max(sqrt(Rh))

    if not tension:
        return hstack([q.ravel() + 10**(-5), tol - Rm])
    return [tol - Rm]


def _slsqp(fn, qid0, bounds, printout, fieq, args):

    pout = 2 if printout else 0
    opt  = fmin_slsqp(fn, qid0, args=args, disp=pout, bounds=bounds, full_output=1, iter=300, f_ieqcons=fieq)

    return opt[1], opt[0]


def _diff_evo(fn, bounds, population, generations, printout, plot, frange, args):

    return devo_numpy(fn=fn, bounds=bounds, population=population, generations=generations, printout=printout,
                      plot=plot, frange=frange, args=args)


def randomise_form(form):

    """ Randomises the FormDiagram by shuffling the edges.

    Parameters
    ----------
    form : obj
        Original FormDiagram.

    Returns
    -------
    obj
        Shuffled FormDiagram.

    """

    # Edges

    edges = [form.edge_coordinates(u, v) for u, v in form.edges()]
    edges = [[sp[:2] + [0], ep[:2] + [0]] for sp, ep in edges]
    shuffle(edges)

    form_ = FormDiagram.from_lines(edges, delete_boundary_face=False)
    form_.update_default_edge_attributes({'is_symmetry': False})
    sym = [geometric_key(form.edge_midpoint(u, v)[:2] + [0])for u, v in form.edges_where({'is_symmetry': True})]
    for u, v in form_.edges():
        if geometric_key(form_.edge_midpoint(u, v)) in sym:
            form_.set_edge_attribute((u, v), 'is_symmetry', True)

    # Vertices

    gkey_key = form_.gkey_key()
    for key, vertex in form.vertex.items():
        gkey = geometric_key(form.vertex_coordinates(key)[:2] + [0])
        form_.vertex[gkey_key[gkey]] = vertex

    form_.attributes['indset'] = []

    return form_


def _worker(data):

    try:

        i, form, save_figs, qmin, qmax, population, generations, simple, tension, tol = data
        fopt, qopt = optimise_single(form, qmin=qmin, qmax=qmax, population=population, generations=generations,
                                     printout=0, tension=tension, tol=tol)

        print('Trial: {0} - Optimum: {1:.1f}'.format(i, fopt))

        if save_figs:
            plotter = plot_form(form, radius=0.1, fix_width=False, simple=simple)
            plotter.save('{0}trial_{1}-fopt_{2:.6f}.png'.format(save_figs, i, fopt))
            del plotter

        return fopt, form

    except:

        print('Trial: {0} - FAILED'.format(i))

        return 10**10, None


def optimise_multi(form, trials=10, save_figs='', qmin=0.001, qmax=5, population=300, generations=500, simple=False,
                   tension=False, tol=0.001):

    """ Finds the optimised load-path for multiple randomised FormDiagrams.

    Parameters
    ----------
    form : obj
        FormDiagram to analyse.
    trials : int
        Number of trials to perform.
    save_figs : str
        Directory to save plots.
    qmin : float
        Minimum qid value.
    qmax : float
        Maximum qid value.
    population : int
        Number of agents for the evolution solver.
    generations : int
        Number of generations for the evolution solver.
    simple : bool
        Simple red and blue colour plotting.
    tension : bool
        Allow tension edge force densities (experimental).
    tol : float
        Tolerance on horizontal force balance.

    Returns
    -------
    list
        Optimum load-path for each trial.
    list
        Each resulting trial FormDiagram.
    int
        Index of the optimum.

    """

    data = [(i, randomise_form(form), save_figs, qmin, qmax, population, generations, simple, tension, tol)
            for i in range(trials)]

    fopts, forms = zip(*Pool().map(_worker, data))
    best = argmin(fopts)

    print('Best: {0} - fopt {1:.1f}'.format(best, fopts[best]))

    return fopts, forms, best


def plot_form(form, radius=0.1, fix_width=False, max_width=10, simple=False):

    """ Extended load-path plotting of a FormDiagram

    Parameters
    ----------
    form : obj
        FormDiagram to plot.
    radius : float
        Radius of vertex markers.
    fix_width : bool
        Fix edge widths as constant.
    max_width : float
        Maximum edge width.
    simple : bool
        Simple red and blue colour plotting.

    Returns
    -------
    obj
        Plotter object.

    """

    q = [attr['q'] for u, v, attr in form.edges(True)]
    qmax  = max(abs(array(q)))
    lines = []

    for u, v in form.edges():
        qi = form.get_edge_attribute((u, v), 'q')

        if simple:
            if qi > 0:
                colour = ['ff', '00', '00']
            elif qi < 0:
                colour = ['00', '00', 'ff']
            else:
                colour = ['aa', 'aa', 'aa']

        else:
            colour = ['00', '00', '00']
            if qi > 0:
                colour[0] = 'ff'
            if form.get_edge_attribute((u, v), 'is_symmetry'):
                colour[1] = 'cc'
            if form.get_edge_attribute((u, v), 'is_ind'):
                colour[2] = 'ff'

        width = max_width if fix_width else (qi / qmax) * max_width

        lines.append({
            'start': form.vertex_coordinates(u),
            'end':   form.vertex_coordinates(v),
            'color': ''.join(colour),
            'width': width,
        })

    plotter = MeshPlotter(form, figsize=(10, 10))
    if radius:
        plotter.draw_vertices(facecolor={i: '#aaaaaa' for i in form.vertices_where({'is_fixed': True})}, radius=radius)
    plotter.draw_lines(lines)

    return plotter


def view_form(form):

    """ View thrust network with compas VtkViewer.

    Parameters
    ----------
    form : obj
        FormDiagram to view thrust network.

    Returns
    -------
    None

    """

    viewer = VtkViewer(datastructure=form)
    viewer.setup()
    viewer.start()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    # Load FormDiagram

    file = '/home/al/compas_loadpath/data/gridshell.json'
    form = FormDiagram.from_json(file)

    # Single run

    # form = randomise_form(form)
    # fopt, qopt = optimise_single(form, qmax=5, population=200, generations=200, printout=10, tol=0.01)

    # Multiple runs

    fopts, forms, best = optimise_multi(form, trials=50, save_figs='/home/al/temp/lp/', qmin=-5, qmax=5,
                                        population=200, generations=200, tol=0.01)
    form = forms[best]

    # plot_form(form, radius=0.05).show()
    view_form(form)
