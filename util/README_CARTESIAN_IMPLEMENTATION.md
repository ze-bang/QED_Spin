# Comprehensive Cartesian Basis Implementation

## Summary
The file `animate_DSSF_cartesian.py` needs to be extended to ~2400+ lines to match all features of `animate_DSSF.py`.

## Current Status
✓ Basic transformation framework created (709 lines)
✓ Individual Cartesian component plots
✗ Missing: DO channels, sublattice analysis, transverse operators

## Implementation Plan

### Step 1: Add Sublattice Parsing
```python
def parse_species_name_with_sublattice(species_name):
    """Handle species like 'SmSp_q_Qx0_Qy0_Qz0_sub0_sub1'"""
    sub_match = re.search(r'_sub(\d+)_sub(\d+)', species_name)
    if sub_match:
        sublattices = (int(sub_match.group(1)), int(sub_match.group(2)))
        base = re.sub(r'_sub\d+_sub\d+$', '', species_name)
        operator, q_info = parse_species_name(base)
        return operator, q_info, sublattices
    else:
        operator, q_info = parse_species_name(species_name)
        return operator, q_info, None
```

### Step 2: Transform Sublattice Data to Cartesian
```python
def create_cartesian_sublattice_data(h_values, h_dirs, base_pattern, cart_component):
    """
    Create AA, BB, AB correlation data for a Cartesian component
    
    Parameters:
    - base_pattern: e.g., "SmSp_q_Qx0_Qy0_Qz0"
    - cart_component: 'SxSx', 'SySy', 'SzSz', 'SxSy', 'SySz', or 'SzSx'
    
    Returns:
    - Three dicts (freq_data, spectral_data) for AA, BB, AB correlations
    """
    # Logic:
    # 1. Find all sublattice pairs for required ladder operators
    # 2. For each sublattice pair, transform to Cartesian
    # 3. Group into AA, BB, AB based on SUBLATTICE_A and SUBLATTICE_B
```

### Step 3: Add DO Channel Support
```python
def create_cartesian_do_channel(h_values, h_dirs, base_species, cart_component):
    """
    Create DO channel for Cartesian component
    DO = SF + NSF
    
    Parameters:
    - base_species: e.g., "q_Qx0_Qy0_Qz0"
    - cart_component: 'SxSx', etc.
    """
    # Transform both SF and NSF channels
    # Combine: DO = SF + NSF
```

###Step 4: Add Global Transverse Operator
```python
def create_cartesian_transverse_sublattice_data(h_values, h_dirs, base_pattern, cart_component):
    """
    Apply transverse operator weighting to Cartesian sublattice data
    Weight = (z_μ·z_ν - (z_μ·Q̂)(z_ν·Q̂))
    """
    # Get Cartesian sublattice data (AA, BB, AB)
    # Apply transverse weights
```

### Step 5: Main Processing Loop
The main() function needs these sections:

1. **Individual Cartesian Components** (✓ done)
   - For each Q-vector
   - For each component (SxSx, SySy, SzSz, SxSy, SySz, SzSx)
   - Create: stacked, heatmap, animation

2. **DO Channels** (TODO)
   - For each Q-vector with SF/NSF
   - For each Cartesian component
   - Create: DO channel plots

3. **Combined Component Plots** (partial)
   - Show all 6 Cartesian components together
   - At multiple field values

4. **Sublattice Analysis** (TODO)
   - For each Q-vector
   - For each Cartesian component
   - Create: AA, BB, AB plots

5. **Individual Sublattice Pairs** (TODO)
   - For specific sublattice pairs
   - For each Cartesian component
   - Combined plots

6. **Global Transverse** (TODO)
   - Weighted sublattice correlations
   - For each Cartesian component

7. **Magnetization** (✓ can copy from original)
   - No transformation needed

8. **Summary Report** (TODO)
   - Overview of all Cartesian components
   - Key findings

## File Size Estimate
- Header + imports + config: ~130 lines
- Helper functions: ~400 lines
- Transformation functions: ~500 lines
- Plotting functions: ~800 lines
- Sublattice functions: ~600 lines
- Main processing: ~500 lines
- **Total: ~2930 lines**

## Recommended Approach
Due to the file size, I recommend:

1. **Option A**: Extend current animate_DSSF_cartesian.py incrementally
   - Add one category at a time
   - Test each addition

2. **Option B**: Create modular structure
   - `cartesian_transforms.py` - transformation functions
   - `cartesian_plots.py` - plotting functions
   - `cartesian_sublattice.py` - sublattice analysis
   - `animate_DSSF_cartesian_main.py` - main script

3. **Option C**: Generate from template
   - Use original animate_DSSF.py as template
   - Systematically replace ladder operators with Cartesian transformations

## Key Transformation Mappings

| Original (Ladder) | Cartesian Transform |
|-------------------|---------------------|
| SmSp              | SxSx = Re(SmSm+SmSp)/2, SySy = -Re(SmSm-SmSp)/2 |
| SzSz              | SzSz (unchanged) |
| SmSm              | Contributes to SxSx, SySy, SxSy |
| SmSz, SpSz        | SySz = Im(SmSz-SpSz)/2, SzSx = Re(SmSz+SpSz)/2 |

## Testing Strategy
1. Verify transformations with known test cases
2. Check that SxSx + SySy + SzSz = SmSp + SzSz (approximately)
3. Validate sublattice decomposition
4. Compare DO channels with expected symmetries

## Next Steps
Would you like me to:
1. Create the full ~3000 line script (will take multiple operations)
2. Extend current script section by section
3. Create modular structure
4. Provide sed/awk script to automatically transform original?
