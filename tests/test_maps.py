from __future__ import annotations

from collections.abc import Callable

import dolfinx as df
import numpy as np
import pytest
import ufl
from mpi4py import MPI

from fenics_constitutive import build_subspace_map
from fenics_constitutive.maps import IdentityMap, SubSpaceMap

ElementBuilder = Callable[[df.mesh.Mesh], ufl.FiniteElementBase]

ELEMENT_BUILDERS = [
    lambda mesh: ufl.FiniteElement(
        "Quadrature", mesh.ufl_cell(), 2, quad_scheme="default"
    ),
    lambda mesh: ufl.VectorElement(
        "Quadrature", mesh.ufl_cell(), 2, quad_scheme="default", dim=3
    ),
    lambda mesh: ufl.TensorElement(
        "Quadrature", mesh.ufl_cell(), 2, quad_scheme="default", shape=(3, 3)
    ),
]


def test_subspace_vector_map_vector_equals_tensor_map():
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 7, 11)

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    Q = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), 2, quad_scheme="default")
    QV = ufl.VectorElement(
        "Quadrature", mesh.ufl_cell(), 2, quad_scheme="default", dim=3
    )
    QT = ufl.TensorElement(
        "Quadrature", mesh.ufl_cell(), 2, quad_scheme="default", shape=(3, 3)
    )

    Q_space = df.fem.FunctionSpace(mesh, Q)
    QV_space = df.fem.FunctionSpace(mesh, QV)
    QT_space = df.fem.FunctionSpace(mesh, QT)

    Q_map, *_ = build_subspace_map(cells, Q_space)
    QV_map, *_ = build_subspace_map(cells, QV_space)
    QT_map, *_ = build_subspace_map(cells, QT_space)

    assert isinstance(Q_map, IdentityMap)
    assert isinstance(QV_map, IdentityMap)
    assert isinstance(QT_map, IdentityMap)

    for _ in range(10):
        cell_sample = np.random.permutation(cells)[: num_cells // 2]

        Q_map, *_ = build_subspace_map(cell_sample, Q_space)
        QV_map, *_ = build_subspace_map(cell_sample, QV_space)
        QT_map, *_ = build_subspace_map(cell_sample, QT_space)

        assert isinstance(Q_map, SubSpaceMap)
        assert isinstance(QV_map, SubSpaceMap)
        assert isinstance(QT_map, SubSpaceMap)
        assert np.all(Q_map.parent == QV_map.parent)
        assert np.all(Q_map.child == QV_map.child)
        assert np.all(Q_map.parent == QT_map.parent)
        assert np.all(Q_map.child == QT_map.child)


@pytest.mark.parametrize("element_builder", ELEMENT_BUILDERS)
def test_subspace_map_evaluation(element_builder: ElementBuilder) -> None:
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 7, 11)

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    Q = element_builder(mesh)
    Q_space = df.fem.FunctionSpace(mesh, Q)

    q = df.fem.Function(Q_space)
    q_test = q.copy()

    value_array = np.random.random(q.x.array.shape)
    q.x.array[:] = value_array

    for _ in range(10):
        cell_sample = np.random.permutation(cells)[: num_cells // 2]

        Q_sub_map, submesh, _ = build_subspace_map(cell_sample, Q_space)
        Q_sub = df.fem.FunctionSpace(submesh, Q)
        q_sub = df.fem.Function(Q_sub)

        Q_sub_map.map_to_child(q, q_sub)
        Q_sub_map.map_to_parent(q_sub, q_test)

        q_view = value_array.reshape(num_cells, -1)
        q_test_view = q_test.x.array.reshape(num_cells, -1)
        assert np.all(q_view[cell_sample] == q_test_view[cell_sample])


@pytest.mark.parametrize("element_builder", ELEMENT_BUILDERS)
def test__identity_map_evaluation(
    element_builder: Callable[[df.mesh.Mesh], ufl.FiniteElementBase],
) -> None:
    mesh = df.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 7, 11)

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    Q = element_builder(mesh)
    Q_space = df.fem.FunctionSpace(mesh, Q)

    q = df.fem.Function(Q_space)
    q_test = q.copy()

    value_array = np.random.random(q.x.array.shape)
    q.x.array[:] = value_array

    identity_map, _, Q_sub = build_subspace_map(cells, Q_space)
    q_sub = df.fem.Function(Q_sub)

    identity_map.map_to_child(q, q_sub)
    identity_map.map_to_parent(q_sub, q_test)

    q_view = value_array.reshape(num_cells, -1)
    q_test_view = q_test.x.array.reshape(num_cells, -1)
    assert np.all(q_view == q_test_view)
