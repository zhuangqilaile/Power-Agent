import pytest
import os
import math
import pyomo.environ as pyo
from pydantic import ValidationError
from main import (
    extract_grid_data,
    solve_dc_opf,
    DC_OPF_Problem,
    Generator,
    Bus,
    Line,
)


TEST_DIR = "tests"


# ---------------------------------------------------------------------------
# Helper: manually constructed DC_OPF_Problem objects for solver-only tests
# These bypass the LLM extraction entirely so tests are deterministic.
# ---------------------------------------------------------------------------

def _make_3bus_baseline():
    """The same grid as test_01_baseline.txt, built directly."""
    return DC_OPF_Problem(
        objective_type="economic_dispatch",
        generators=[
            Generator(name="Coal Plant", at_bus="Bus 1", cost_0=100, cost_1=20, cost_2=0.05,
                      max_capacity_mw=400, min_capacity_mw=50),
            Generator(name="Gas Plant", at_bus="Bus 2", cost_0=50, cost_1=40, cost_2=0.1,
                      max_capacity_mw=300, min_capacity_mw=0),
        ],
        lines=[
            Line(name="Line 1-2", from_bus="Bus 1", to_bus="Bus 2", max_flow_mw=500, reactance=0.02),
            Line(name="Line 2-3", from_bus="Bus 2", to_bus="Bus 3", max_flow_mw=500, reactance=0.02),
            Line(name="Line 1-3", from_bus="Bus 1", to_bus="Bus 3", max_flow_mw=200, reactance=0.02),
        ],
        buses=[
            Bus(name="Bus 1", load_mw=0),
            Bus(name="Bus 2", load_mw=0),
            Bus(name="Bus 3", load_mw=350),
        ],
    )


def _make_2bus_constrained():
    """2-bus grid where the line is congested at 150 MW.
    Bus A has 300 MW capacity, Bus B has 200 MW capacity.
    Total load = 250 MW, line limit = 150 MW.
    """
    return DC_OPF_Problem(
        objective_type="OPF",
        generators=[
            Generator(name="Alpha", at_bus="Bus A", cost_0=200, cost_1=50, cost_2=0.15,
                      max_capacity_mw=300, min_capacity_mw=0),
            Generator(name="Beta", at_bus="Bus B", cost_0=100, cost_1=60, cost_2=0.2,
                      max_capacity_mw=200, min_capacity_mw=0),
        ],
        lines=[
            Line(name="Line A-B", from_bus="Bus A", to_bus="Bus B", max_flow_mw=150, reactance=0.1),
        ],
        buses=[
            Bus(name="Bus A", load_mw=0),
            Bus(name="Bus B", load_mw=250),
        ],
    )


def _make_5bus_multi_gen():
    """5-bus grid with 3 generators and 6 lines."""
    return DC_OPF_Problem(
        objective_type="economic_dispatch",
        generators=[
            Generator(name="Gen 1", at_bus="Bus 1", cost_0=0, cost_1=10, cost_2=0.02,
                      max_capacity_mw=150, min_capacity_mw=10),
            Generator(name="Gen 2", at_bus="Bus 2", cost_0=0, cost_1=20, cost_2=0.04,
                      max_capacity_mw=100, min_capacity_mw=5),
            Generator(name="Gen 3", at_bus="Bus 3", cost_0=0, cost_1=30, cost_2=0.06,
                      max_capacity_mw=80, min_capacity_mw=0),
        ],
        lines=[
            Line(name="Line 1-2", from_bus="Bus 1", to_bus="Bus 2", max_flow_mw=200, reactance=0.02),
            Line(name="Line 1-4", from_bus="Bus 1", to_bus="Bus 4", max_flow_mw=150, reactance=0.04),
            Line(name="Line 2-3", from_bus="Bus 2", to_bus="Bus 3", max_flow_mw=200, reactance=0.02),
            Line(name="Line 2-4", from_bus="Bus 2", to_bus="Bus 4", max_flow_mw=100, reactance=0.06),
            Line(name="Line 3-5", from_bus="Bus 3", to_bus="Bus 5", max_flow_mw=150, reactance=0.03),
            Line(name="Line 4-5", from_bus="Bus 4", to_bus="Bus 5", max_flow_mw=120, reactance=0.05),
        ],
        buses=[
            Bus(name="Bus 1", load_mw=0),
            Bus(name="Bus 2", load_mw=60),
            Bus(name="Bus 3", load_mw=70),
            Bus(name="Bus 4", load_mw=50),
            Bus(name="Bus 5", load_mw=40),
        ],
    )


def _make_linear_only():
    """Grid with pure linear costs (cost_2 = 0). Tests that solver handles
    linear objective correctly even though IPOPT is a nonlinear solver."""
    return DC_OPF_Problem(
        objective_type="minimize_cost",
        generators=[
            Generator(name="Cheap Gen", at_bus="Bus 1", cost_0=0, cost_1=10, cost_2=0.0,
                      max_capacity_mw=400, min_capacity_mw=0),
            Generator(name="Expensive Gen", at_bus="Bus 2", cost_0=0, cost_1=50, cost_2=0.0,
                      max_capacity_mw=300, min_capacity_mw=0),
        ],
        lines=[
            Line(name="Line 1-2", from_bus="Bus 1", to_bus="Bus 2", max_flow_mw=500, reactance=0.02),
        ],
        buses=[
            Bus(name="Bus 1", load_mw=0),
            Bus(name="Bus 2", load_mw=200),
        ],
    )


def _make_4bus_ring():
    """4-bus ring network with 2 cheap generators and 1 expensive peaker."""
    return DC_OPF_Problem(
        objective_type="OPF",
        generators=[
            Generator(name="Solar Farm", at_bus="Bus 1", cost_0=5, cost_1=15, cost_2=0.01,
                      max_capacity_mw=200, min_capacity_mw=0),
            Generator(name="Wind Farm", at_bus="Bus 2", cost_0=3, cost_1=12, cost_2=0.008,
                      max_capacity_mw=180, min_capacity_mw=0),
            Generator(name="Gas Peaker", at_bus="Bus 4", cost_0=80, cost_1=70, cost_2=0.3,
                      max_capacity_mw=100, min_capacity_mw=0),
        ],
        lines=[
            Line(name="Line 1-2", from_bus="Bus 1", to_bus="Bus 2", max_flow_mw=250, reactance=0.01),
            Line(name="Line 2-3", from_bus="Bus 2", to_bus="Bus 3", max_flow_mw=200, reactance=0.02),
            Line(name="Line 3-4", from_bus="Bus 3", to_bus="Bus 4", max_flow_mw=150, reactance=0.03),
            Line(name="Line 4-1", from_bus="Bus 4", to_bus="Bus 1", max_flow_mw=180, reactance=0.04),
        ],
        buses=[
            Bus(name="Bus 1", load_mw=0),
            Bus(name="Bus 2", load_mw=0),
            Bus(name="Bus 3", load_mw=120),
            Bus(name="Bus 4", load_mw=80),
        ],
    )


# ===================================================================
# PART 1: PYDANTIC SCHEMA VALIDATION TESTS (no LLM, no solver)
# ===================================================================

class TestSchemaValidation:

    def test_valid_problem_passes(self):
        """A well-formed DC_OPF_Problem should pass all validators."""
        prob = _make_3bus_baseline()
        assert prob.objective_type == "economic_dispatch"
        assert len(prob.generators) == 2
        assert len(prob.lines) == 3
        assert len(prob.buses) == 3

    def test_missing_reactance_raises(self):
        """Line with null reactance triggers Missing Data Error."""
        with pytest.raises(ValidationError, match="Missing Data Error"):
            DC_OPF_Problem(
                objective_type="OPF",
                generators=[
                    Generator(name="G1", at_bus="B1", cost_1=10, max_capacity_mw=100),
                ],
                lines=[
                    Line(name="L1", from_bus="B1", to_bus="B2", max_flow_mw=50, reactance=None),
                ],
                buses=[
                    Bus(name="B1", load_mw=0),
                    Bus(name="B2", load_mw=50),
                ],
            )

    def test_missing_max_flow_raises(self):
        """Line with null max_flow_mw triggers Missing Data Error."""
        with pytest.raises(ValidationError, match="Missing Data Error"):
            DC_OPF_Problem(
                objective_type="OPF",
                generators=[
                    Generator(name="G1", at_bus="B1", cost_1=10, max_capacity_mw=100),
                ],
                lines=[
                    Line(name="L1", from_bus="B1", to_bus="B2", max_flow_mw=None, reactance=0.02),
                ],
                buses=[
                    Bus(name="B1", load_mw=0),
                    Bus(name="B2", load_mw=50),
                ],
            )

    def test_missing_cost1_raises(self):
        """Generator with null cost_1 triggers Missing Data Error."""
        with pytest.raises(ValidationError, match="Missing Data Error"):
            DC_OPF_Problem(
                objective_type="OPF",
                generators=[
                    Generator(name="G1", at_bus="B1", cost_1=None, max_capacity_mw=100),
                ],
                lines=[
                    Line(name="L1", from_bus="B1", to_bus="B2", max_flow_mw=50, reactance=0.02),
                ],
                buses=[
                    Bus(name="B1", load_mw=0),
                    Bus(name="B2", load_mw=50),
                ],
            )

    def test_missing_capacity_raises(self):
        """Generator with null max_capacity_mw triggers Missing Data Error."""
        with pytest.raises(ValidationError, match="Missing Data Error"):
            DC_OPF_Problem(
                objective_type="OPF",
                generators=[
                    Generator(name="G1", at_bus="B1", cost_1=10, max_capacity_mw=None),
                ],
                lines=[
                    Line(name="L1", from_bus="B1", to_bus="B2", max_flow_mw=50, reactance=0.02),
                ],
                buses=[
                    Bus(name="B1", load_mw=0),
                    Bus(name="B2", load_mw=50),
                ],
            )

    def test_topology_error_line_from_nonexistent_bus(self):
        """Line referencing a nonexistent from_bus triggers Topology Error."""
        with pytest.raises(ValidationError, match="Topology Error"):
            DC_OPF_Problem(
                objective_type="OPF",
                generators=[
                    Generator(name="G1", at_bus="B1", cost_1=10, max_capacity_mw=200),
                ],
                lines=[
                    Line(name="L1", from_bus="B_ghost", to_bus="B2", max_flow_mw=50, reactance=0.02),
                ],
                buses=[
                    Bus(name="B1", load_mw=0),
                    Bus(name="B2", load_mw=50),
                ],
            )

    def test_topology_error_line_to_nonexistent_bus(self):
        """Line referencing a nonexistent to_bus triggers Topology Error."""
        with pytest.raises(ValidationError, match="Topology Error"):
            DC_OPF_Problem(
                objective_type="OPF",
                generators=[
                    Generator(name="G1", at_bus="B1", cost_1=10, max_capacity_mw=200),
                ],
                lines=[
                    Line(name="L1", from_bus="B1", to_bus="B_ghost", max_flow_mw=50, reactance=0.02),
                ],
                buses=[
                    Bus(name="B1", load_mw=0),
                    Bus(name="B2", load_mw=50),
                ],
            )

    def test_physics_error_load_exceeds_capacity(self):
        """Total load > total generation capacity triggers Physics Error."""
        with pytest.raises(ValidationError, match="Physics Error"):
            DC_OPF_Problem(
                objective_type="OPF",
                generators=[
                    Generator(name="G1", at_bus="B1", cost_1=10, max_capacity_mw=50),
                ],
                lines=[
                    Line(name="L1", from_bus="B1", to_bus="B2", max_flow_mw=200, reactance=0.02),
                ],
                buses=[
                    Bus(name="B1", load_mw=0),
                    Bus(name="B2", load_mw=100),
                ],
            )

    def test_value_error_negative_capacity(self):
        """Negative max_capacity_mw triggers Value Error.
        Note: load_mw=0 so Physics check passes (0 >= -10), allowing Value check to fire."""
        with pytest.raises(ValidationError, match="Value Error"):
            DC_OPF_Problem(
                objective_type="OPF",
                generators=[
                    Generator(name="G1", at_bus="B1", cost_1=10, max_capacity_mw=-10),
                ],
                lines=[
                    Line(name="L1", from_bus="B1", to_bus="B2", max_flow_mw=50, reactance=0.02),
                ],
                buses=[
                    Bus(name="B1", load_mw=0),
                    Bus(name="B2", load_mw=0),
                ],
            )

    def test_value_error_negative_min_capacity(self):
        """Negative min_capacity_mw triggers Value Error."""
        with pytest.raises(ValidationError, match="Value Error"):
            DC_OPF_Problem(
                objective_type="OPF",
                generators=[
                    Generator(name="G1", at_bus="B1", cost_1=10, max_capacity_mw=100,
                              min_capacity_mw=-5),
                ],
                lines=[
                    Line(name="L1", from_bus="B1", to_bus="B2", max_flow_mw=50, reactance=0.02),
                ],
                buses=[
                    Bus(name="B1", load_mw=0),
                    Bus(name="B2", load_mw=30),
                ],
            )

    def test_value_error_negative_reactance(self):
        """Negative reactance triggers Value Error."""
        with pytest.raises(ValidationError, match="Value Error"):
            DC_OPF_Problem(
                objective_type="OPF",
                generators=[
                    Generator(name="G1", at_bus="B1", cost_1=10, max_capacity_mw=100),
                ],
                lines=[
                    Line(name="L1", from_bus="B1", to_bus="B2", max_flow_mw=50, reactance=-0.02),
                ],
                buses=[
                    Bus(name="B1", load_mw=0),
                    Bus(name="B2", load_mw=30),
                ],
            )

    def test_value_error_negative_max_flow(self):
        """Negative max_flow_mw triggers Value Error."""
        with pytest.raises(ValidationError, match="Value Error"):
            DC_OPF_Problem(
                objective_type="OPF",
                generators=[
                    Generator(name="G1", at_bus="B1", cost_1=10, max_capacity_mw=100),
                ],
                lines=[
                    Line(name="L1", from_bus="B1", to_bus="B2", max_flow_mw=-50, reactance=0.02),
                ],
                buses=[
                    Bus(name="B1", load_mw=0),
                    Bus(name="B2", load_mw=30),
                ],
            )

    def test_generator_at_bus_not_in_buses_is_allowed(self):
        """at_bus referencing a bus not in the buses list does NOT trigger
        a topology error -- the validator only checks lines, not generators.
        This documents the known behavior gap."""
        prob = DC_OPF_Problem(
            objective_type="OPF",
            generators=[
                Generator(name="G1", at_bus="Ghost Bus", cost_1=10, max_capacity_mw=200),
            ],
            lines=[
                Line(name="L1", from_bus="B1", to_bus="B2", max_flow_mw=50, reactance=0.02),
            ],
            buses=[
                Bus(name="B1", load_mw=0),
                Bus(name="B2", load_mw=30),
            ],
        )
        assert len(prob.generators) == 1


# ===================================================================
# PART 2: SOLVER TESTS (deterministic, no LLM needed)
# ===================================================================

class TestSolver:

    def test_3bus_baseline_optimal(self):
        """Baseline 3-bus problem should solve to optimal."""
        prob = _make_3bus_baseline()
        results, model = solve_dc_opf(prob)
        assert str(results.solver.termination_condition) == "optimal"

    def test_3bus_baseline_total_generation_equals_load(self):
        """Total generation must equal total load (no losses in DC model)."""
        prob = _make_3bus_baseline()
        results, model = solve_dc_opf(prob)
        total_gen = sum(pyo.value(model.gen_power[g]) for g in model.generator)
        total_load = sum(b.load_mw for b in prob.buses)
        assert abs(total_gen - total_load) < 1e-3

    def test_3bus_baseline_generator_within_limits(self):
        """Each generator must respect its min/max capacity."""
        prob = _make_3bus_baseline()
        results, model = solve_dc_opf(prob)
        for g in prob.generators:
            pg = pyo.value(model.gen_power[g.name])
            assert pg >= g.min_capacity_mw - 1e-3
            assert pg <= g.max_capacity_mw + 1e-3

    def test_3bus_baseline_line_within_limits(self):
        """Each line flow must respect its max_flow_mw limit."""
        prob = _make_3bus_baseline()
        results, model = solve_dc_opf(prob)
        for l in prob.lines:
            flow = pyo.value(model.line_flow[l.name])
            assert abs(flow) <= l.max_flow_mw + 1e-3

    def test_3bus_baseline_ref_bus_angle_zero(self):
        """Reference bus (first bus) must have angle = 0."""
        prob = _make_3bus_baseline()
        results, model = solve_dc_opf(prob)
        ref_bus = model.bus.first()
        assert abs(pyo.value(model.theta[ref_bus])) < 1e-6

    def test_3bus_baseline_nodal_balance(self):
        """Each bus must satisfy nodal balance: gen - load = net flow out."""
        prob = _make_3bus_baseline()
        results, model = solve_dc_opf(prob)
        for b in prob.buses:
            bus_name = b.name
            gen_at_bus = sum(
                pyo.value(model.gen_power[g.name])
                for g in prob.generators if g.at_bus == bus_name
            )
            load_at_bus = b.load_mw
            flow_in = sum(
                pyo.value(model.line_flow[l.name])
                for l in prob.lines if l.to_bus == bus_name
            )
            flow_out = sum(
                pyo.value(model.line_flow[l.name])
                for l in prob.lines if l.from_bus == bus_name
            )
            assert abs(gen_at_bus - load_at_bus - (flow_out - flow_in)) < 1e-2

    def test_2bus_constrained_line_limit_binding(self):
        """In 2-bus corridor, the 150 MW line limit should be binding.
        Cheaper gen Alpha at Bus A wants to send more, but line caps at 150."""
        prob = _make_2bus_constrained()
        results, model = solve_dc_opf(prob)
        flow = pyo.value(model.line_flow["Line A-B"])
        assert abs(flow) >= 140

    def test_2bus_constrained_total_generation(self):
        """Total generation must match 250 MW load."""
        prob = _make_2bus_constrained()
        results, model = solve_dc_opf(prob)
        total_gen = sum(pyo.value(model.gen_power[g]) for g in model.generator)
        assert abs(total_gen - 250) < 1e-2

    def test_5bus_multi_gen_optimal(self):
        """5-bus grid with 3 generators should solve to optimal."""
        prob = _make_5bus_multi_gen()
        results, model = solve_dc_opf(prob)
        assert str(results.solver.termination_condition) == "optimal"

    def test_5bus_multi_gen_total_generation(self):
        """Total generation must equal total load = 220 MW."""
        prob = _make_5bus_multi_gen()
        results, model = solve_dc_opf(prob)
        total_gen = sum(pyo.value(model.gen_power[g]) for g in model.generator)
        assert abs(total_gen - 220) < 1e-2

    def test_5bus_multi_gen_cheapest_gen_dispatches_most(self):
        """Gen 1 (cheapest, $10/MW) should dispatch more than Gen 3 ($30/MW)."""
        prob = _make_5bus_multi_gen()
        results, model = solve_dc_opf(prob)
        pg1 = pyo.value(model.gen_power["Gen 1"])
        pg3 = pyo.value(model.gen_power["Gen 3"])
        assert pg1 > pg3

    def test_linear_only_costs_optimal(self):
        """Pure linear costs should still solve with IPOPT."""
        prob = _make_linear_only()
        results, model = solve_dc_opf(prob)
        assert str(results.solver.termination_condition) == "optimal"

    def test_linear_only_cheapest_dispatches_all(self):
        """With linear costs and unconstrained line, cheapest gen should
        dispatch the full load (200 MW) and expensive gen dispatches 0."""
        prob = _make_linear_only()
        results, model = solve_dc_opf(prob)
        pg_cheap = pyo.value(model.gen_power["Cheap Gen"])
        pg_expensive = pyo.value(model.gen_power["Expensive Gen"])
        assert pg_cheap >= 195
        assert pg_expensive <= 5

    def test_4bus_ring_optimal(self):
        """4-bus ring network should solve to optimal."""
        prob = _make_4bus_ring()
        results, model = solve_dc_opf(prob)
        assert str(results.solver.termination_condition) == "optimal"

    def test_4bus_ring_total_generation(self):
        """Total generation must equal total load = 200 MW."""
        prob = _make_4bus_ring()
        results, model = solve_dc_opf(prob)
        total_gen = sum(pyo.value(model.gen_power[g]) for g in model.generator)
        assert abs(total_gen - 200) < 1e-2

    def test_4bus_ring_peaker_dispatches_least(self):
        """Gas Peaker ($70/MW) should dispatch less than Solar ($15) and Wind ($12)."""
        prob = _make_4bus_ring()
        results, model = solve_dc_opf(prob)
        pg_solar = pyo.value(model.gen_power["Solar Farm"])
        pg_wind = pyo.value(model.gen_power["Wind Farm"])
        pg_peaker = pyo.value(model.gen_power["Gas Peaker"])
        assert pg_solar > pg_peaker
        assert pg_wind > pg_peaker


# ===================================================================
# PART 3: LLM EXTRACTION PIPELINE TESTS
# These call the Gemini API and are marked as "llm" for selective
# running. They depend on a valid GEMINI_API_KEY in .env.
# ===================================================================

_test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".txt")]


@pytest.mark.llm
@pytest.mark.parametrize("filename", _test_files)
def test_extraction_pipeline(filename):
    """Full end-to-end test: extract from text -> validate -> solve."""
    filepath = os.path.join(TEST_DIR, filename)

    with open(filepath, "r") as f:
        prompt_text = f.read()

    if "missing_data" in filename:
        with pytest.raises(ValidationError):
            extract_grid_data(prompt_text)
        return

    extracted_data = extract_grid_data(prompt_text)
    results, model = solve_dc_opf(extracted_data)
    assert str(results.solver.termination_condition) == "optimal"


@pytest.mark.llm
@pytest.mark.parametrize("filename", _test_files)
def test_extraction_schema_correctness(filename):
    """Verify the extracted object has correct structural properties."""
    filepath = os.path.join(TEST_DIR, filename)

    with open(filepath, "r") as f:
        prompt_text = f.read()

    if "missing_data" in filename:
        with pytest.raises(ValidationError):
            extract_grid_data(prompt_text)
        return

    extracted = extract_grid_data(prompt_text)

    for gen in extracted.generators:
        assert gen.cost_1 is not None, f"{gen.name} has null cost_1"
        assert gen.max_capacity_mw is not None, f"{gen.name} has null max_capacity_mw"
        assert gen.at_bus is not None and gen.at_bus != ""

    for line in extracted.lines:
        assert line.reactance is not None, f"{line.name} has null reactance"
        assert line.max_flow_mw is not None, f"{line.name} has null max_flow_mw"
        assert line.from_bus is not None and line.from_bus != ""
        assert line.to_bus is not None and line.to_bus != ""

    for bus in extracted.buses:
        assert bus.name is not None and bus.name != ""
        assert bus.load_mw is not None

    bus_names = {b.name for b in extracted.buses}
    for line in extracted.lines:
        assert line.from_bus in bus_names, f"Line {line.name}: from_bus '{line.from_bus}' not in buses"
        assert line.to_bus in bus_names, f"Line {line.name}: to_bus '{line.to_bus}' not in buses"

    for gen in extracted.generators:
        assert gen.at_bus in bus_names, f"Generator {gen.name}: at_bus '{gen.at_bus}' not in buses"


@pytest.mark.llm
def test_extraction_messy_text_bus_consistency(self):
    """Test that messy/informal text still produces consistent bus names."""
    filepath = os.path.join(TEST_DIR, "test_02_messy.txt")
    with open(filepath, "r") as f:
        prompt_text = f.read()

    extracted = extract_grid_data(prompt_text)
    bus_names = {b.name for b in extracted.buses}

    for line in extracted.lines:
        assert line.from_bus in bus_names
        assert line.to_bus in bus_names

    for gen in extracted.generators:
        assert gen.at_bus in bus_names


@pytest.mark.llm
def test_extraction_preserves_zero_costs(self):
    """Hydro dam with zero costs should have cost_0=0, cost_1=0, cost_2=0."""
    filepath = os.path.join(TEST_DIR, "test_02_messy.txt")
    with open(filepath, "r") as f:
        prompt_text = f.read()

    extracted = extract_grid_data(prompt_text)
    hydro = [g for g in extracted.generators if "hydro" in g.name.lower() or "north" in g.name.lower()]
    assert len(hydro) >= 1, "Expected a hydro/North Station generator"
    h = hydro[0]
    assert h.cost_0 == 0.0
    assert h.cost_1 == 0.0
    assert h.cost_2 == 0.0


# ===================================================================
# PART 4: PHYSICS / CONSISTENCY CHECKS ON SOLVER RESULTS
# ===================================================================

class TestSolverPhysics:

    def test_kirchhoff_all_buses(self):
        """Nodal balance holds at every bus for the 5-bus case."""
        prob = _make_5bus_multi_gen()
        results, model = solve_dc_opf(prob)
        for b in prob.buses:
            bus_name = b.name
            gen_at_bus = sum(
                pyo.value(model.gen_power[g.name])
                for g in prob.generators if g.at_bus == bus_name
            )
            load_at_bus = b.load_mw
            flow_in = sum(
                pyo.value(model.line_flow[l.name])
                for l in prob.lines if l.to_bus == bus_name
            )
            flow_out = sum(
                pyo.value(model.line_flow[l.name])
                for l in prob.lines if l.from_bus == bus_name
            )
            assert abs(gen_at_bus - load_at_bus - (flow_out - flow_in)) < 1e-2

    def test_line_flows_symmetric_sign(self):
        """Flow from A to B should be negative of flow from B to A
        (checking DC power flow angle difference sign convention)."""
        prob = _make_3bus_baseline()
        results, model = solve_dc_opf(prob)
        flow_12 = pyo.value(model.line_flow["Line 1-2"])
        theta1 = pyo.value(model.theta["Bus 1"])
        theta2 = pyo.value(model.theta["Bus 2"])
        B_12 = 1 / 0.02
        expected_flow = B_12 * (theta1 - theta2)
        assert abs(flow_12 - expected_flow) < 1e-2

    def test_no_negative_generation(self):
        """No generator should produce negative power."""
        prob = _make_5bus_multi_gen()
        results, model = solve_dc_opf(prob)
        for g in model.generator:
            assert pyo.value(model.gen_power[g]) >= -1e-3


# ===================================================================
# PART 5: EDGE CASES
# ===================================================================

class TestEdgeCases:

    def test_zero_load_bus(self):
        """A bus with zero load should still satisfy nodal balance."""
        prob = _make_3bus_baseline()
        results, model = solve_dc_opf(prob)
        bus_name = "Bus 1"
        gen_at_bus = sum(
            pyo.value(model.gen_power[g.name])
            for g in prob.generators if g.at_bus == bus_name
        )
        flow_in = sum(
            pyo.value(model.line_flow[l.name])
            for l in prob.lines if l.to_bus == bus_name
        )
        flow_out = sum(
            pyo.value(model.line_flow[l.name])
            for l in prob.lines if l.from_bus == bus_name
        )
        assert abs(gen_at_bus - (flow_out - flow_in)) < 1e-2

    def test_single_generator_at_bus(self):
        """When only one generator is at a bus, it handles the local load + export."""
        prob = _make_2bus_constrained()
        results, model = solve_dc_opf(prob)
        pg_alpha = pyo.value(model.gen_power["Alpha"])
        flow_AB = pyo.value(model.line_flow["Line A-B"])
        assert abs(pg_alpha - flow_AB) < 1e-2

    def test_objective_cost_calculation(self):
        """Verify the objective value matches manual cost calculation."""
        prob = _make_3bus_baseline()
        results, model = solve_dc_opf(prob)
        pg_coal = pyo.value(model.gen_power["Coal Plant"])
        pg_gas = pyo.value(model.gen_power["Gas Plant"])
        manual_cost = (0.05 * pg_coal**2 + 20 * pg_coal + 100) + \
                      (0.1 * pg_gas**2 + 40 * pg_gas + 50)
        obj_value = pyo.value(model.f1)
        assert abs(obj_value - manual_cost) < 1e-1