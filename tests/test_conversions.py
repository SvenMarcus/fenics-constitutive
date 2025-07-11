from __future__ import annotations

import basix
import dolfinx as df
import numpy as np
import pytest
import ufl
from mpi4py import MPI

from fenics_constitutive import (
    StressStrainConstraint,
    strain_from_grad_u,
    ufl_mandel_strain,
)


def test_strain_from_grad_u():
    grad_u = np.array([[1.0]])
    constraint = StressStrainConstraint.UNIAXIAL_STRAIN
    strain = strain_from_grad_u(grad_u, constraint)
    assert np.allclose(strain, np.array([1.0]))
    constraint = StressStrainConstraint.UNIAXIAL_STRESS
    strain = strain_from_grad_u(grad_u, constraint)
    assert np.allclose(strain, np.array([1.0]))
    grad_u = np.array([[1.0, 2.0], [3.0, 4.0]])
    constraint = StressStrainConstraint.PLANE_STRAIN
    strain = strain_from_grad_u(grad_u, constraint)
    assert np.allclose(strain, np.array([1.0, 4.0, 0.0, 0.5 * (4.0 + 1.0) * 2**0.5]))
    constraint = StressStrainConstraint.PLANE_STRESS
    strain = strain_from_grad_u(grad_u, constraint)
    assert np.allclose(strain, np.array([1.0, 4.0, 0.0, 0.5 * (4.0 + 1.0) * 2**0.5]))
    grad_u = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    constraint = StressStrainConstraint.FULL
    strain = strain_from_grad_u(grad_u, constraint)
    assert np.allclose(
        strain,
        np.array(
            [
                1.0,
                5.0,
                9.0,
                0.5 * (2.0 + 4.0) * 2**0.5,
                0.5 * (3.0 + 7.0) * 2**0.5,
                0.5 * (6.0 + 8.0) * 2**0.5,
            ]
        ),
    )


@pytest.mark.parametrize(
    ("constraint"),
    [
        (StressStrainConstraint.UNIAXIAL_STRAIN),
        (StressStrainConstraint.UNIAXIAL_STRESS),
        (StressStrainConstraint.PLANE_STRAIN),
        (StressStrainConstraint.PLANE_STRESS),
        (StressStrainConstraint.FULL),
    ],
)
def test_ufl_strain_equals_array_conversion(constraint: StressStrainConstraint):
    match constraint.geometric_dim:
        case 1:
            mesh = df.mesh.create_unit_interval(MPI.COMM_WORLD, 2)

            def lam(x):
                return x[0] * 0.1
        case 2:
            mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)

            def lam(x):
                return x[0] * 0.1, x[1] * 0.2
        case 3:
            mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)

            def lam(x):
                return x[0] * 0.1, x[1] * 0.2, x[2] * 0.3

    P1 = df.fem.functionspace(mesh, ("CG", 1, (constraint.geometric_dim,)))
    u = df.fem.Function(P1)
    grad_u_ufl = ufl.grad(u)
    mandel_strain_ufl = ufl_mandel_strain(u, constraint)
    u.interpolate(lam)

    points, weights = basix.make_quadrature(
        basix.CellType[mesh.ufl_cell().cellname()],
        1,
        basix.quadrature.string_to_type("default"),
    )

    expr_grad_u = df.fem.Expression(grad_u_ufl, points)
    expr_mandel_strain = df.fem.Expression(mandel_strain_ufl, points)

    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    grad_u_array = expr_grad_u.eval(mesh, np.arange(n_cells, dtype=np.int32)).flatten()
    strain_array = expr_mandel_strain.eval(
        mesh, np.arange(n_cells, dtype=np.int32)
    ).flatten()
    strain_array_from_grad_u = strain_from_grad_u(grad_u_array, constraint)
    assert np.allclose(strain_array, strain_array_from_grad_u)
