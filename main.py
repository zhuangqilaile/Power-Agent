from pydantic import BaseModel, model_validator, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pyomo.environ as pyo

class Generator(BaseModel):
    name: str
    at_bus: str
    cost_0: float = 0.0
    cost_1: float | None = None
    cost_2: float = 0.0
    max_capacity_mw: float | None = Field(default=None, description="Maximum generation capacity.")
    min_capacity_mw: float = 0.0

class Bus(BaseModel):
    name: str
    load_mw: float

class Line(BaseModel):
    name: str
    from_bus: str | None = None
    to_bus: str | None = None
    max_flow_mw: float | None = Field(default=None, description="The maximum power flow, limit, or capacity of the line.")
    reactance: float | None = Field(default=None, description="The reactance or X value of the line.")

class DC_OPF_Problem(BaseModel):
    objective_type: str
    generators: list[Generator]
    lines: list[Line]
    buses: list[Bus]

    @model_validator(mode='after')
    def check_grid_physics_and_topology(self) -> 'DC_OPF_Problem':
        # 1. Topology Check: Do the lines connect to real buses?
        for line in self.lines:
            if line.reactance is None or line.max_flow_mw is None:
                raise ValueError(f"Missing Data Error: Line '{line.name}' is missing reactance or max flow!")
                
        for gen in self.generators:
            if None in (gen.cost_1, gen.max_capacity_mw):
                raise ValueError(f"Missing Data Error: Generator '{gen.name}' is missing cost or capacity data!")
            
        bus_names = {b.name for b in self.buses}
        for line in self.lines:
            if line.from_bus not in bus_names:
                raise ValueError(f"Topology Error: Line '{line.name}' starts at '{line.from_bus}', but that bus doesn't exist!")
            if line.to_bus not in bus_names:
                raise ValueError(f"Topology Error: Line '{line.name}' ends at '{line.to_bus}', but that bus doesn't exist!")

        # 2. Basic Physics Check: Is there enough generation to meet demand?
        total_load = sum(b.load_mw for b in self.buses)
        total_capacity = sum(g.max_capacity_mw for g in self.generators)
        
        if total_capacity < total_load:
            raise ValueError(f"Physics Error: Grid is doomed! Total Load ({total_load} MW) exceeds Total Max Capacity ({total_capacity} MW).")
        
        # 3. Non-Negative Value Check
        for g in self.generators:
            if g.max_capacity_mw < 0 or g.min_capacity_mw < 0:
                raise ValueError(f"Value Error: Generator '{g.name}' has negative capacity!")
        
        for line in self.lines:
            if line.reactance < 0: 
                raise ValueError(f"Value Error: line '{line.name}' has negative reactance!")
            if line.max_flow_mw < 0:
                raise ValueError(f"Value Error: line '{line.name}' has negative flow capacity!")
        
        return self

def extract_grid_data(prompt_text: str):
    load_dotenv()
    client = genai.Client()
    strict_instructions = (
        "You are a strict grid data extraction system. "
        "CRITICAL RULE: If a numeric value (like reactance, cost, or max capacity) "
        "is not explicitly stated in the text, DO NOT guess, DO NOT calculate it, "
        "and DO NOT assume a default value of 0. You MUST set the value to null. \n\n"
        "Grid Text to Extract:\n"
    )
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=strict_instructions + prompt_text,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DC_OPF_Problem,
            temperature=0.0,
        ),
    )
    extracted_data = DC_OPF_Problem.model_validate_json(response.text)
    return extracted_data

def solve_dc_opf(extracted_data):
    model = pyo.ConcreteModel()

    # extract the paramters
    model.generator = pyo.Set(initialize = [g.name for g in extracted_data.generators])
    model.line = pyo.Set(initialize = [l.name for l in extracted_data.lines])
    model.bus = pyo.Set(initialize= [b.name for b in extracted_data.buses])

    # parameters of generator
    model.gen_power = pyo.Var(model.generator)

    costs = {g.name: [g.cost_0, g.cost_1, g.cost_2] for g in extracted_data.generators}
    model.cost = pyo.Param(model.generator, initialize=costs)
    model.theta = pyo.Var(model.bus, domain=pyo.Reals)
    gen_loc = {g.name: g.at_bus for g in extracted_data.generators}
    model.gen_loc = pyo.Param(model.generator, initialize=gen_loc)

    min_capacity = {g.name: g.min_capacity_mw for g in extracted_data.generators}
    model.min_capicity = pyo.Param(model.generator, initialize=min_capacity)
    max_capacity = {g.name: g.max_capacity_mw for g in extracted_data.generators}
    model.max_capicity = pyo.Param(model.generator, initialize=max_capacity)

    # parameters of line
    from_bus = {l.name: l.from_bus for l in extracted_data.lines}
    to_bus = {l.name: l.to_bus for l in extracted_data.lines}
    model.line_from = pyo.Param(model.line, initialize=from_bus)
    model.line_to = pyo.Param(model.line, initialize=to_bus)

    susceptance = {l.name: 1/l.reactance for l in extracted_data.lines}
    model.B = pyo.Param(model.line, initialize=susceptance)

    max_flow = {l.name: l.max_flow_mw for l in extracted_data.lines}
    model.max_flow = pyo.Param(model.line, initialize=max_flow)

    # parameters of bus
    bus_load = {b.name: b.load_mw for b in extracted_data.buses}
    model.bus_load = pyo.Param(model.bus, initialize=bus_load)


    # Constrainsts
    def max_rule(model, g): # generator limit
        return model.gen_power[g] <= model.max_capicity[g]
    def min_rule(model, g):
        return model.gen_power[g] >= model.min_capicity[g]
    model.c1 = pyo.Constraint(model.generator, rule = max_rule)
    model.c2 = pyo.Constraint(model.generator, rule = min_rule)

    model.ref_angle = pyo.Constraint(expr= model.theta[model.bus.first()] == 0) # set the first bus as reference

    def calc_line_flow(model, line):
        start_bus = model.line_from[line]
        end_bus = model.line_to[line]
        return model.B[line] * (model.theta[start_bus] - model.theta[end_bus])
    model.line_flow = pyo.Expression(model.line, rule=calc_line_flow)

    def line_limit_rule(model, line): # line flow limit
        return (-model.max_flow[line], model.line_flow[line], model.max_flow[line])
    model.line_limits = pyo.Constraint(model.line, rule=line_limit_rule)

    def Nodal_rule(model, bus): # nodal balance
        generator_mw = 0
        for g in model.generator:
            if model.gen_loc[g] == bus:
                generator_mw += model.gen_power[g]
        load_mw = model.bus_load[bus]
        flow_in = sum(model.line_flow[l] for l in model.line if model.line_to[l] == bus)
        flow_out = sum(model.line_flow[l] for l in model.line if model.line_from[l] == bus)
        return (generator_mw - load_mw) == (flow_out - flow_in)
    model.nodal_balance = pyo.Constraint(model.bus, rule=Nodal_rule)

    # Objective function
    model.f1 = pyo.Objective(expr = sum(model.cost[g][2] * model.gen_power[g]**2 + model.cost[g][1] * model.gen_power[g] + model.cost[g][0] for g in model.generator), sense=pyo.minimize)

    # Solve
    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model)
    return results, model