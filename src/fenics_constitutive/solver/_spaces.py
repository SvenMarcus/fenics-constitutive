from __future__ import annotations

from dataclasses import dataclass

import dolfinx as df
import ufl

from fenics_constitutive.interfaces import StressStrainConstraint


@dataclass(frozen=True, slots=True)
class ElementSpaces:
    _stress_vector_element: ufl.VectorElement
    _stress_tensor_element: ufl.TensorElement
    _displacement_gradient_tensor_element: ufl.TensorElement
    stress_vector_space: df.fem.FunctionSpace
    q_degree: int

    @staticmethod
    def create(
        mesh: df.mesh.Mesh, constraint: StressStrainConstraint, q_degree: int
    ) -> ElementSpaces:
        gdim = mesh.ufl_cell().geometric_dimension()
        stress_vector_element = ufl.VectorElement(
            "Quadrature",
            mesh.ufl_cell(),
            q_degree,
            quad_scheme="default",
            dim=constraint.stress_strain_dim,
        )
        stress_tensor_element = ufl.TensorElement(
            "Quadrature",
            mesh.ufl_cell(),
            q_degree,
            quad_scheme="default",
            shape=(constraint.stress_strain_dim, constraint.stress_strain_dim),
        )
        displacement_gradient_tensor_element = ufl.TensorElement(
            "Quadrature",
            mesh.ufl_cell(),
            q_degree,
            quad_scheme="default",
            shape=(gdim, gdim),
        )
        stress_vector_space = df.fem.FunctionSpace(mesh, stress_vector_element)
        return ElementSpaces(
            stress_vector_element,
            stress_tensor_element,
            displacement_gradient_tensor_element,
            stress_vector_space,
            q_degree,
        )

    def displacement_gradient_tensor_space(
        self, mesh: df.mesh.Mesh
    ) -> df.fem.FunctionSpace:
        return df.fem.FunctionSpace(mesh, self._displacement_gradient_tensor_element)

    def stress_tensor_space(self, mesh: df.mesh.Mesh) -> df.fem.FunctionSpace:
        return df.fem.FunctionSpace(mesh, self._stress_tensor_element)
