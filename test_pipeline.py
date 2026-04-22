import pytest
import os
from dotenv import load_dotenv
from pydantic import ValidationError
from main import extract_grid_data, solve_dc_opf

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

    # For all other tests, extraction should work perfectly
    extracted_data = extract_grid_data(prompt_text)

    results, _ = solve_dc_opf(extracted_data)
    assert str(results.solver.termination_condition) == "optimal"