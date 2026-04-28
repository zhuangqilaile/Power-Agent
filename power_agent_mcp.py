from mcp.server.fastmcp import FastMCP
import pyomo.environ as pyo
from main import extract_grid_data, solve_dc_opf

mcp = FastMCP("PowerAgent")

@mcp.tool()
def optimize_grid(grid_description: str) -> str:
    """
    Extracts grid parameters from natural language and runs a DC-OPF optimization.
    Use this tool whenever the user asks to evaluate, simulate, or optimize a power grid.
    
    Args:
        grid_description: The plain text description of the buses, lines, and generators.
    """
    try:
        grid_data = extract_grid_data(grid_description)
        
        results, model = solve_dc_opf(grid_data)
        
        termination = str(results.solver.termination_condition)
        
        # Handle Infeasible grids gracefully so Claude knows what happened
        if termination not in ["optimal", "locallyOptimal"]:
            return f"The Pyomo solver failed. The grid is physically impossible. Termination status: {termination}"
        
        # Format the successful results into a string for Claude to read
        output = ["DC-OPF Optimization Successful!\n", "Generation Schedule:"]
        
        for g in model.generator:
            gen_name = str(g)
            output_mw = pyo.value(model.gen_power[g])
            output.append(f"- {gen_name}: {output_mw:.2f} MW")

            
        return "\n".join(output)

    except Exception as e:
        return f"Power Agent Pipeline Error: {str(e)}"

if __name__ == "__main__":
    mcp.run()