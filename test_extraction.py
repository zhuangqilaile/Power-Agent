import pytest
import os
from dotenv import load_dotenv
from pydantic import ValidationError
from main import extract_grid_data

TEST_DIR = "tests"
test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".txt")]

@pytest.mark.parametrize("filename", test_files)
def test_grid_pipeline(filename):
    filepath = os.path.join(TEST_DIR, filename)
    
    with open(filepath, 'r') as file:
        prompt_text = file.read()

    # Test Case: Missing Data
    # We EXPECT this to fail our Pydantic validation!
    if "missing_data" in filename:
        with pytest.raises(ValidationError):
            extract_grid_data(prompt_text)
        return # Test passes because it successfully caught the bad data!
    
    if "infeasible" in filename:
        with pytest.raises(ValidationError) as exc_info:
            extract_grid_data(prompt_text)
        assert ("Physics Error" in str(exc_info.value)) or ("negative" in str(exc_info.value))
        return
    


    # For all other tests, extraction should work perfectly
    extracted_data = extract_grid_data(prompt_text)

    if "baseline" in filename:
        assert len(extracted_data.buses) == 3
        assert len(extracted_data.lines) == 3
        assert len(extracted_data.generators) == 2
        assert extracted_data.lines[0].max_flow_mw == 500.0
        assert extracted_data.buses[2].load_mw == 350.0
    
    elif filename == "test_02_messy.txt":
        assert len(extracted_data.buses) == 3
        assert len(extracted_data.lines) == 3
        assert len(extracted_data.generators) == 2
        assert extracted_data.lines[0].max_flow_mw == 250.0
        assert extracted_data.buses[1].load_mw == 400

    elif filename == "test_03_redundant.txt":
        assert len(extracted_data.buses) == 3
        assert len(extracted_data.lines) == 3
        assert len(extracted_data.generators) == 2
        alpha_gen = next((g for g in extracted_data.generators if "Alpha" in g.name), None)
        assert alpha_gen.cost_1 == 15.0, "LLM extracted the wrong dispatch cost!"
        assert alpha_gen.max_capacity_mw == 300.0
        
        mountain_line = next((l for l in extracted_data.lines if "1-2" in l.name), None)
        assert mountain_line is not None, "AI missed Line 1-2!"
        assert mountain_line.reactance == 0.05
        assert mountain_line.max_flow_mw == 400.0

    elif "test_04_" in filename:
        assert len(extracted_data.buses) == 5, "AI missed a bus in the 5-bus system!"
        assert len(extracted_data.generators) == 3, "AI missed a generator!"
        assert len(extracted_data.lines) == 6, "AI missed a transmission line!"
        
        bus_4 = next((b for b in extracted_data.buses if "4" in b.name), None)
        assert bus_4 is not None, "AI failed to extract Bus 4!"
        assert bus_4.load_mw == 50.0, "AI got the wrong load for Bus 4!"
        
        gen_2 = next((g for g in extracted_data.generators if "2" in g.name), None)
        assert gen_2 is not None, "AI failed to extract Gen 2!"
        assert gen_2.max_capacity_mw == 100.0
        assert gen_2.cost_1 == 20.0, "AI missed the linear cost for Gen 2!"
        assert gen_2.cost_2 == 0.04, "AI missed the quadratic cost for Gen 2!"
        
        line_45 = next((l for l in extracted_data.lines if "4-5" in l.name), None)
        assert line_45 is not None, "AI failed to extract Line 4-5!"
        assert line_45.reactance == 0.05
        assert line_45.max_flow_mw == 120.0
    
    elif "test_05_" in filename:
        assert len(extracted_data.buses) == 2, "AI found the wrong number of buses!"
        assert len(extracted_data.generators) == 2, "AI found the wrong number of generators!"
        assert len(extracted_data.lines) == 1, "AI missed the transmission line!"
        
        gen_alpha = next((g for g in extracted_data.generators if "Alpha" in g.name), None)
        assert gen_alpha is not None, "AI failed to extract Generator Alpha!"
        assert gen_alpha.cost_0 == 200.0, "AI missed Alpha's no-load cost!"
        assert gen_alpha.cost_1 == 50.0, "AI missed Alpha's linear cost!"
        assert gen_alpha.cost_2 == 0.15, "AI missed Alpha's quadratic cost!"
        assert gen_alpha.max_capacity_mw == 300.0
        
        gen_beta = next((g for g in extracted_data.generators if "Beta" in g.name), None)
        assert gen_beta is not None, "AI failed to extract Generator Beta!"
        assert gen_beta.cost_0 == 100.0
        assert gen_beta.cost_1 == 60.0
        assert gen_beta.cost_2 == 0.2
        
        bus_b = next((b for b in extracted_data.buses if "Bus B" in b.name), None)
        assert bus_b is not None, "AI failed to extract Bus B!"
        assert bus_b.load_mw == 250.0, "AI missed the load at Bus B!"
        
        line = extracted_data.lines[0] # Safe to use [0] since there is exactly 1 line
        assert line.reactance == 0.1
        assert line.max_flow_mw == 150.0

    elif "test_06_" in filename:
        assert len(extracted_data.buses) == 4, "AI found the wrong number of buses!"
        assert len(extracted_data.generators) == 3, "AI found the wrong number of generators!"
        assert len(extracted_data.lines) == 4, "AI missed a transmission line in the ring!"
        
        solar = next((g for g in extracted_data.generators if "Solar" in g.name), None)
        assert solar is not None, "AI failed to extract the Solar Farm!"
        assert "1" in solar.at_bus, "AI put the Solar Farm at the wrong bus!"
        assert solar.cost_1 == 15.0
        
        gas = next((g for g in extracted_data.generators if "Gas" in g.name or "Peaker" in g.name), None)
        assert gas is not None, "AI failed to extract the Gas Peaker!"
        assert gas.cost_0 == 80.0
        assert gas.cost_2 == 0.3
        
        line_41 = next((l for l in extracted_data.lines if "4-1" in l.name or "1-4" in l.name), None)
        assert line_41 is not None, "AI missed the line closing the ring (Line 4-1)!"
        assert line_41.reactance == 0.04
        assert line_41.max_flow_mw == 180.0
        
        bus_3 = next((b for b in extracted_data.buses if "3" in b.name), None)
        assert bus_3 is not None
        assert bus_3.load_mw == 120.0
    
    elif "local_infeasible" in filename:
        assert len(extracted_data.buses) == 2, "AI found the wrong number of buses!"
        assert len(extracted_data.generators) == 1, "AI found the wrong number of generators!"
        assert len(extracted_data.lines) == 1, "AI missed a transmission line in the ring!"

        big_coal = next((g for g in extracted_data.generators if "Coal" in g.name), None)
        assert big_coal is not None, "AI missed Big Coal!"
 
        assert big_coal.max_capacity_mw == 500.0
        assert big_coal.cost_0 == 500.0
        
        tiny_gas = next((g for g in extracted_data.generators if "Gas" in g.name), None)
        assert tiny_gas is not None, "AI missed Tiny Gas!"
        assert tiny_gas.max_capacity_mw == 50.0
        
        line = extracted_data.lines[0] 
        assert line.max_flow_mw == 100.0

