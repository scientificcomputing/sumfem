import pytest
import ufl

from sumfem.symbolic import ComponentError, Domain
from sumfem.symbolic.domains import MEASURE_ENTITY_TYPE


def test_domain_is_a_named_ufl_mesh():
    domain = Domain("electrode", "triangle", gdim=3, degree=2)
    assert isinstance(domain, ufl.Mesh)
    assert domain.name == "electrode"
    assert domain.cell_name == "triangle"
    assert domain.tdim == 2
    assert domain.gdim == 3
    assert ufl.SpatialCoordinate(domain).ufl_shape == (3,)
    assert domain.ufl_coordinate_element().embedded_superdegree == 2


def test_domains_with_same_name_are_distinct_objects():
    assert Domain("cell", "triangle") != Domain("cell", "triangle")


@pytest.mark.parametrize(
    "cell, expected",
    [
        ("interval", {"cell": 1, "facet": 0, "vertex": 0}),
        ("triangle", {"cell": 2, "facet": 1, "ridge": 0, "vertex": 0}),
        ("tetrahedron", {"cell": 3, "facet": 2, "ridge": 1, "peak": 0, "vertex": 0}),
    ],
)
def test_entity_dimensions_follow_codimension(cell, expected):
    domain = Domain("cell", cell)
    assert {t: domain.entity_dimension(t) for t in expected} == expected


@pytest.mark.parametrize(
    "cell, entity_type", [("interval", "ridge"), ("triangle", "peak")]
)
def test_entity_types_that_do_not_exist_are_rejected(cell, entity_type):
    domain = Domain("cell", cell)
    with pytest.raises(ComponentError, match="no entities"):
        domain.entity_dimension(entity_type)
    with pytest.raises(ComponentError, match="no entities"):
        domain.marker_ids("A", entity_type)


def test_ridge_measure_does_not_exist_on_intervals():
    with pytest.raises(ComponentError, match="no entities"):
        Domain("cell", "interval").dr()


@pytest.mark.parametrize("integral_type", sorted(MEASURE_ENTITY_TYPE))
def test_marker_ids_are_assigned_per_entity_dimension(integral_type):
    domain = Domain("cell", "tetrahedron")
    measure = getattr(domain, integral_type)
    assert measure("A").subdomain_id() == (1,)
    assert measure("B").subdomain_id() == (2,)
    # Asking again gives the same id
    assert measure("A").subdomain_id() == (1,)
    dim = domain.entity_dimension(MEASURE_ENTITY_TYPE[integral_type])
    assert domain.markers[dim] == {"A": 1, "B": 2}
    names = {v: k for k, v in ufl.measure.integral_type_to_measure_name.items()}
    assert measure("A").integral_type() == names[integral_type]


def test_coinciding_entity_types_share_markers():
    triangle = Domain("cell", "triangle")
    # On a triangle mesh a ridge is a vertex
    assert triangle.dr("Corner").subdomain_id() == (1,)
    assert triangle.dP("Point").subdomain_id() == (2,)
    assert triangle.dP("Corner").subdomain_id() == (1,)
    assert triangle.markers[0] == {"Corner": 1, "Point": 2}
    # On a tetrahedron mesh a peak is a vertex, while a ridge is an edge
    tetrahedron = Domain("cell", "tetrahedron")
    assert tetrahedron.marker_ids("Tip", "peak") == tetrahedron.dP("Tip").subdomain_id()
    assert tetrahedron.dr("Edge").subdomain_id() == (1,)
    assert tetrahedron.markers[1] == {"Edge": 1}


def test_marker_ids_returns_one_id_per_name():
    domain = Domain("cell", "triangle")
    assert domain.marker_ids("Electrode", "cell") == (1,)
    # New names get the next ids, known names keep theirs, order is kept
    assert domain.marker_ids(["Channel", "Electrode", "Inlet"], "cell") == (2, 1, 3)
    assert domain.marker_ids((), "cell") == ()
    # A single string is one marker, not one marker per character
    assert domain.markers[2] == {"Electrode": 1, "Channel": 2, "Inlet": 3}
    with pytest.raises(ComponentError):
        domain.marker_ids(["Electrode", ""], "cell")


def test_facet_markers_are_shared_by_exterior_and_interior_facets():
    domain = Domain("cell", "triangle")
    assert domain.ds("Membrane").subdomain_id() == (1,)
    assert domain.dS("Membrane").subdomain_id() == (1,)
    # Cell markers are numbered independently of facet markers
    assert domain.dx("Electrode").subdomain_id() == (1,)
    assert domain.markers == {0: {}, 1: {"Membrane": 1}, 2: {"Electrode": 1}}


def test_measures():
    domain = Domain("cell", "quadrilateral")
    whole = domain.dx()
    assert whole.subdomain_id() == "everywhere"
    assert whole.ufl_domain() is domain
    # The whole-domain measure is created once and reused
    assert domain.dx() is whole
    assert domain.dx("A") == whole(domain.marker_ids("A", "cell"))
    assert domain.dx(degree=3).subdomain_id() == "everywhere"
    several = domain.dx("A", "B")
    assert several.subdomain_id() == (1, 2)
    assert domain.ds(degree=4).metadata() == {"quadrature_degree": 4}
    assert domain.dS(metadata={"quadrature_rule": "default"}, degree=2).metadata() == {
        "quadrature_rule": "default",
        "quadrature_degree": 2,
    }


def test_markers_returns_a_copy():
    domain = Domain("cell", "triangle")
    domain.dx("A")
    domain.markers[2]["B"] = 7
    assert domain.markers[2] == {"A": 1}


@pytest.mark.parametrize("name", ["", "a b", "a.b", " a"])
def test_invalid_domain_names(name):
    with pytest.raises(ComponentError):
        Domain(name, "triangle")


def test_invalid_markers_and_integral_types():
    domain = Domain("cell", "triangle")
    with pytest.raises(ComponentError):
        domain.dx("")
    with pytest.raises(ComponentError):
        domain.measure("dX", "A")
    with pytest.raises(ComponentError, match="Unknown entity type"):
        domain.marker_ids("A", "edge")
    with pytest.raises(TypeError):
        domain.dx(3)
