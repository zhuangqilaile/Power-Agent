import pytest
import pyomo.environ as pyo

from main import DC_OPF_Problem, Bus, Generator, Line, solve_dc_opf

def test_01_pure_economic_dispatch():
    test_data = DC_OPF_Problem(
        objective_type="economic_dispatch",
        buses=[
            Bus(name="Bus 1", load_mw=150.0)
        ],
        lines=[], # No transmission lines, just a single node
        generators=[
            Generator(name="Cheap Gen", at_bus="Bus 1", cost_1=20.0, max_capacity_mw=200.0),
            Generator(name="Expensive Gen", at_bus="Bus 1", cost_1=80.0, max_capacity_mw=200.0)
        ]
    )
    
    results, model = solve_dc_opf(test_data)
    
    # 1. Assert solver found the perfect answer
    assert str(results.solver.termination_condition) == "optimal"
    
    # 2. Assert cheap generator takes all the load, expensive one stays off
    assert pyo.value(model.gen_power["Cheap Gen"]) == pytest.approx(150.0)
    assert pyo.value(model.gen_power["Expensive Gen"]) == pytest.approx(0.0, abs=1e-6)


def test_02_congestion_physics():
    test_data = DC_OPF_Problem(
        objective_type="economic_dispatch",
        buses=[
            Bus(name="Bus 1", load_mw=0.0),
            Bus(name="Bus 2", load_mw=0.0),
            Bus(name="Bus 3", load_mw=150.0)
        ],
        lines=[
            # Equal impedance triangle
            Line(name="Line 1-2", from_bus="Bus 1", to_bus="Bus 2", reactance=1.0, max_flow_mw=200.0),
            Line(name="Line 2-3", from_bus="Bus 2", to_bus="Bus 3", reactance=1.0, max_flow_mw=200.0),
            # THE BOTTLENECK:
            Line(name="Line 1-3", from_bus="Bus 1", to_bus="Bus 3", reactance=1.0, max_flow_mw=80.0) 
        ],
        generators=[
            Generator(name="Gen 1", at_bus="Bus 1", cost_1=10.0, max_capacity_mw=200.0), # Cheap
            Generator(name="Gen 2", at_bus="Bus 2", cost_1=50.0, max_capacity_mw=200.0)  # Expensive
        ]
    )
    
    results, model = solve_dc_opf(test_data)
    
    assert str(results.solver.termination_condition) == "optimal"
    
    # Because Line 1-3 is capped at 80 MW, and flow splits 2/3 down the direct path,
    # Gen 1 can only safely inject 120 MW total before Line 1-3 overloads.
    # Wait, let's test your solver! I'm putting 90.0 and 60.0 here based on the earlier math.
    assert pyo.value(model.gen_power["Gen 1"]) == pytest.approx(90.0)
    assert pyo.value(model.gen_power["Gen 2"]) == pytest.approx(60.0)


def test_03():
    """
    Test: Congestion-Driven Curtailment.
    Verifies the solver curtails $0/MW wind energy when the transmission 
    line exporting it hits its thermal limit.
    """
    test_data = DC_OPF_Problem(
        objective_type="economic_dispatch",
        buses=[
            Bus(name="Bus 1 (Wind Farm)", load_mw=0.0),
            Bus(name="Bus 2 (City)", load_mw=150.0)
        ],
        lines=[
            Line(name="Line 1-2", from_bus="Bus 1 (Wind Farm)", to_bus="Bus 2 (City)", reactance=0.02, max_flow_mw=100.0)
        ],
        generators=[
            Generator(name="Free Wind", at_bus="Bus 1 (Wind Farm)", cost_1=0.0, max_capacity_mw=300.0),
            Generator(name="Expensive Local Gas", at_bus="Bus 2 (City)", cost_1=80.0, max_capacity_mw=200.0)
        ]
    )
    
    results, model = solve_dc_opf(test_data)
    assert str(results.solver.termination_condition) == "optimal"
    
    assert pyo.value(model.gen_power["Free Wind"]) == pytest.approx(100.0, abs=1e-6)
    assert pyo.value(model.gen_power["Expensive Local Gas"]) == pytest.approx(50.0, abs=1e-6)