#!/usr/bin/env python3
"""
Test script to verify the new features in animate_DSSF_updated.py
Tests SF+NSF→DO combination and TransverseExperimental overlay functionality.
"""

import sys
import os

# Add the util directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the functions we want to test
from animate_DSSF_updated import (
    find_sf_nsf_pairs,
    find_transverse_overlay_pairs,
    parse_species_name,
    get_base_species_name
)

def test_find_sf_nsf_pairs():
    """Test SF/NSF pair detection for both Transverse and TransverseExperimental"""
    print("=" * 70)
    print("TEST 1: SF/NSF Pair Detection")
    print("=" * 70)
    
    # Mock species list
    test_species = [
        "SxSx_q_Qx0_Qy0_Qz0_SF",
        "SxSx_q_Qx0_Qy0_Qz0_NSF",
        "SzSz_q_Qx0_Qy0_Qz0_SF",
        "SzSz_q_Qx0_Qy0_Qz0_NSF",
        "TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_SF",
        "TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_NSF",
        "TransverseExperimental_q_Qx0_Qy0_Qz0_theta45_SF",
        "TransverseExperimental_q_Qx0_Qy0_Qz0_theta45_NSF",
        "SpSm_q_Qx0_Qy0_Qz0",  # No SF/NSF, should be ignored
    ]
    
    pairs = find_sf_nsf_pairs(test_species)
    
    print(f"\nFound {len(pairs)} SF/NSF pairs:\n")
    for do_name, sf_name, nsf_name, operator_type in pairs:
        print(f"  {operator_type}:")
        print(f"    DO:  {do_name}")
        print(f"    SF:  {sf_name}")
        print(f"    NSF: {nsf_name}")
        print()
    
    # Verify results
    assert len(pairs) == 4, f"Expected 4 pairs, got {len(pairs)}"
    
    # Check for specific pairs
    pair_dict = {(sf, nsf): (do, op_type) for do, sf, nsf, op_type in pairs}
    
    assert ("SxSx_q_Qx0_Qy0_Qz0_SF", "SxSx_q_Qx0_Qy0_Qz0_NSF") in pair_dict
    assert ("SzSz_q_Qx0_Qy0_Qz0_SF", "SzSz_q_Qx0_Qy0_Qz0_NSF") in pair_dict
    assert ("TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_SF", 
            "TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_NSF") in pair_dict
    
    # Check operator types
    sxsx_pair = pair_dict[("SxSx_q_Qx0_Qy0_Qz0_SF", "SxSx_q_Qx0_Qy0_Qz0_NSF")]
    assert sxsx_pair[1] == "Transverse", f"Expected Transverse, got {sxsx_pair[1]}"
    
    exp_pair = pair_dict[("TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_SF",
                          "TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_NSF")]
    assert exp_pair[1] == "TransverseExperimental", f"Expected TransverseExperimental, got {exp_pair[1]}"
    
    print("✅ SF/NSF pair detection test PASSED\n")


def test_find_transverse_overlay_pairs():
    """Test overlay pair detection for TransverseExperimental validation"""
    print("=" * 70)
    print("TEST 2: TransverseExperimental Overlay Pair Detection")
    print("=" * 70)
    
    # Mock species list
    test_species = [
        "TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_SF",
        "TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_NSF",
        "SxSx_q_Qx0_Qy0_Qz0_SF",
        "SxSx_q_Qx0_Qy0_Qz0_NSF",
        "SzSz_q_Qx0_Qy0_Qz0_SF",
        "SzSz_q_Qx0_Qy0_Qz0_NSF",
        "TransverseExperimental_q_Qx0_Qy0_Qz0_theta45_SF",  # Missing SxSx/SzSz for theta45
        "TransverseExperimental_q_Qx1_Qy0_Qz0_theta60_SF",  # Different Q-point
        "SxSx_q_Qx1_Qy0_Qz0_SF",
        "SzSz_q_Qx1_Qy0_Qz0_SF",
    ]
    
    overlay_pairs = find_transverse_overlay_pairs(test_species)
    
    print(f"\nFound {len(overlay_pairs)} overlay pairs:\n")
    for exp_species, sxsx, szsz, theta in overlay_pairs:
        print(f"  θ = {theta}°:")
        print(f"    Experimental: {exp_species}")
        print(f"    Overlay: cos²({theta}°)·{sxsx} + sin²({theta}°)·{szsz}")
        print()
    
    # Verify results
    assert len(overlay_pairs) == 3, f"Expected 3 overlay pairs, got {len(overlay_pairs)}"
    
    # Check for specific pairs
    exp_species_list = [exp for exp, _, _, _ in overlay_pairs]
    assert "TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_SF" in exp_species_list
    assert "TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_NSF" in exp_species_list
    assert "TransverseExperimental_q_Qx1_Qy0_Qz0_theta60_SF" in exp_species_list
    
    # Verify theta extraction
    for exp_species, sxsx, szsz, theta in overlay_pairs:
        if "theta30" in exp_species:
            assert theta == 30.0, f"Expected theta=30, got {theta}"
        elif "theta60" in exp_species:
            assert theta == 60.0, f"Expected theta=60, got {theta}"
    
    print("✅ Overlay pair detection test PASSED\n")


def test_parse_species_name():
    """Test species name parsing for all operator types"""
    print("=" * 70)
    print("TEST 3: Species Name Parsing")
    print("=" * 70)
    
    test_cases = [
        ("SxSx_q_Qx0_Qy0_Qz0_SF", {
            'operator': 'SxSx',
            'q_pattern': 'q_Qx0_Qy0_Qz0',
            'channel': 'SF',
            'theta': None,
            'sublattices': None
        }),
        ("TransverseExperimental_q_Qx0_Qy0_Qz0_theta45_NSF", {
            'operator': 'TransverseExperimental',
            'q_pattern': 'q_Qx0_Qy0_Qz0',
            'channel': 'NSF',
            'theta': 45.0,
            'sublattices': None
        }),
        ("SzSz_q_Qx0_Qy0_Qz0_sub0_sub1", {
            'operator': 'SzSz',
            'q_pattern': 'q_Qx0_Qy0_Qz0',
            'channel': None,
            'theta': None,
            'sublattices': (0, 1)
        }),
    ]
    
    print("\nTesting species name parsing:\n")
    for species, expected in test_cases:
        result = parse_species_name(species)
        print(f"  Species: {species}")
        print(f"    Operator: {result['operator']} (expected: {expected['operator']})")
        print(f"    Q-pattern: {result['q_pattern']} (expected: {expected['q_pattern']})")
        print(f"    Channel: {result['channel']} (expected: {expected['channel']})")
        print(f"    Theta: {result['theta']} (expected: {expected['theta']})")
        print(f"    Sublattices: {result['sublattices']} (expected: {expected['sublattices']})")
        
        assert result == expected, f"Parse mismatch for {species}"
        print("    ✅ Correct\n")
    
    print("✅ Species name parsing test PASSED\n")


def test_get_base_species_name():
    """Test base name extraction from species with channels"""
    print("=" * 70)
    print("TEST 4: Base Species Name Extraction")
    print("=" * 70)
    
    test_cases = [
        ("SxSx_q_Qx0_Qy0_Qz0_SF", "SxSx_q_Qx0_Qy0_Qz0"),
        ("SxSx_q_Qx0_Qy0_Qz0_NSF", "SxSx_q_Qx0_Qy0_Qz0"),
        ("SxSx_q_Qx0_Qy0_Qz0_DO", "SxSx_q_Qx0_Qy0_Qz0"),
        ("TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_SF", 
         "TransverseExperimental_q_Qx0_Qy0_Qz0_theta30"),
        ("SpSm_q_Qx0_Qy0_Qz0", "SpSm_q_Qx0_Qy0_Qz0"),  # No channel
    ]
    
    print("\nTesting base name extraction:\n")
    for species, expected_base in test_cases:
        result = get_base_species_name(species)
        status = "✅" if result == expected_base else "❌"
        print(f"  {status} {species} → {result}")
        assert result == expected_base, f"Expected {expected_base}, got {result}"
    
    print("\n✅ Base species name extraction test PASSED\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("TESTING animate_DSSF_updated.py NEW FEATURES")
    print("=" * 70 + "\n")
    
    try:
        test_get_base_species_name()
        test_parse_species_name()
        test_find_sf_nsf_pairs()
        test_find_transverse_overlay_pairs()
        
        print("=" * 70)
        print("ALL TESTS PASSED! ✅")
        print("=" * 70)
        print("\nThe script is ready to use with:")
        print("  • SF+NSF→DO combination for Transverse and TransverseExperimental")
        print("  • cos²θ·SxSx + sin²θ·SzSz overlay validation")
        print()
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
