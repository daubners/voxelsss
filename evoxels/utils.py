import numpy as np
import sympy as sp
import sympy.vector as spv
import evoxels as vox

### Generalized test case
def rhs_convergence_test(
    ODE_class,       # an ODE class with callable rhs(field, t)->torch.Tensor (shape [x,y,z])
    problem_kwargs,  # problem parameters to instantiate ODE
    test_function,   # exact init_fun(x,y,z)->np.ndarray
    convention="cell_center",
    dtype="float32",
    powers = np.array([3,4,5,6,7]),
    backend = "torch"
):
    """Evaluate spatial order of an ODE right-hand side.

    ``test_function`` can be a single sympy expression or a list of
    expressions representing multiple variables. The returned error and
    slope arrays have one entry for each provided function.

    Args:
        ODE_class: an ODE class with callable rhs(field, t).
        problem_kwargs: problem-specific parameters to instantiate ODE.
        test_function: single sympy expression or a list of expressions.
        convention: grid convention.
        dtype: floate precision (``float32`` or ``float64``).
        powers: refine grid in powers of two (i.e. ``Nx = 2**p``).
        backend: use ``torch`` or ``jax`` for testing.
    """
    if isinstance(test_function, (list, tuple)):
        test_functions = list(test_function)
    else:
        test_functions = [test_function]
    n_funcs = len(test_functions)

    dx     = np.zeros(len(powers))
    errors = np.zeros((n_funcs, len(powers)))
    CS = spv.CoordSys3D('CS')

    for i, p in enumerate(powers):
        if convention == 'cell_center':
            vf = vox.VoxelFields((2**p, 2**p, 2**p), (1, 1, 1), convention=convention)
        elif convention == 'staggered_x':
            vf = vox.VoxelFields((2**p + 1, 2**p, 2**p), (1, 1, 1), convention=convention)
        vf.precision = dtype
        grid = vf.meshgrid()
        if backend == 'torch':
            vg = vox.voxelgrid.VoxelGridTorch(vf.grid_info(), precision=vf.precision, device='cpu')
        elif backend == 'jax':
            vg = vox.voxelgrid.VoxelGridJax(vf.grid_info(), precision=vf.precision)
    
        # Initialise fields
        u_list = []
        for func in test_functions:
            init_fun = sp.lambdify((CS.x, CS.y, CS.z), func, "numpy")
            init_data = init_fun(*grid)
            u_list.append(vg.init_scalar_field(init_data))

        u = vg.concatenate(u_list, 0)
        u = vg.trim_boundary_nodes(u)
        ODE = ODE_class(vg, **problem_kwargs)
        rhs_numeric = ODE.rhs(u, 0)
        if n_funcs > 1:
            rhs_analytic = ODE.rhs_analytic(test_functions, 0)
        else:
            rhs_analytic = [ODE.rhs_analytic(test_functions[0], 0)]

        # Compute solutions
        for j, func in enumerate(test_functions):
            comp = vg.export_scalar_field_to_numpy(rhs_numeric[j:j+1])
            exact_fun = sp.lambdify((CS.x, CS.y, CS.z), rhs_analytic[j], "numpy")
            exact = exact_fun(*grid)
            if convention == "staggered_x":
                exact = exact[1:-1, :, :]

            # Error norm
            diff = comp - exact
            errors[j, i] = np.linalg.norm(diff) / np.linalg.norm(exact)
        dx[i] = vf.spacing[0]

    # Fit slope after loop
    slopes = np.array(
        [np.polyfit(np.log(dx), np.log(err), 1)[0] for err in errors]
    )
    if slopes.size == 1:
        slopes = slopes[0]
    order = ODE.order

    return dx, errors if errors.shape[0] > 1 else errors[0], slopes, order
    