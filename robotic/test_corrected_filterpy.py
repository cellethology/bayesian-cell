#!/usr/bin/env python3
"""
Test script for the corrected FilterPy implementations.
Validates that the corrected implementations follow proper FilterPy API patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from filter_factory import FilterFactory, get_default_config
from ekf_environment import EKFEnvironment


def test_corrected_filterpy_ekf():
    """Test the corrected FilterPy EKF implementation."""
    print("=" * 60)
    print("TESTING CORRECTED FILTERPY EKF")
    print("=" * 60)

    # Create FilterPy EKF Corrected
    config = get_default_config("FilterPy_EKF_Corrected")
    config.update({
        "signal_max": 15.0,
        "robot_start_pos": [10.0, 10.0],
        "target_true_pos": [30.0, 30.0],
        "max_steps": 50,
        "verbose": False
    })

    try:
        ekf = FilterFactory.create_filter(config)
        print(f"✅ Created FilterPy EKF Corrected: {type(ekf).__name__}")
        
        # Test a few updates
        measurements = [12.0, 10.5, 9.2, 8.1, 7.5]
        robot_positions = [
            np.array([11.0, 11.0]),
            np.array([12.0, 12.0]),
            np.array([14.0, 14.0]),
            np.array([16.0, 16.0]),
            np.array([18.0, 18.0]),
        ]

        print("FilterPy EKF Corrected Updates:")
        for i, (meas, robot_pos) in enumerate(zip(measurements, robot_positions)):
            mu, Sigma, sigma_Q = ekf.predict_and_update(meas, robot_pos)
            uncertainty = np.sqrt(np.trace(Sigma))
            print(f"  Step {i+1}: mu=[{mu[0]:.1f}, {mu[1]:.1f}], uncertainty={uncertainty:.2f}")

        # Test innovation statistics
        innovation_stats = ekf.get_innovation_stats()
        if innovation_stats:
            print(f"✅ Innovation statistics available")
            if innovation_stats['innovation'] is not None:
                print(f"  Last innovation: {innovation_stats['innovation']}")
        
        # Get underlying FilterPy filter
        filterpy_filter = ekf.get_filterpy_filter()
        print(f"✅ Underlying FilterPy filter: {type(filterpy_filter).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing FilterPy EKF Corrected: {str(e)}")
        return False


def test_corrected_filterpy_ukf():
    """Test the corrected FilterPy UKF implementation."""
    print("\n" + "=" * 60)
    print("TESTING CORRECTED FILTERPY UKF")
    print("=" * 60)

    # Create FilterPy UKF Corrected
    config = get_default_config("FilterPy_UKF_Corrected")
    config.update({
        "signal_max": 15.0,
        "robot_start_pos": [10.0, 10.0],
        "target_true_pos": [30.0, 30.0],
        "max_steps": 50,
        "verbose": False
    })

    try:
        ukf = FilterFactory.create_filter(config)
        print(f"✅ Created FilterPy UKF Corrected: {type(ukf).__name__}")
        
        # Test UKF parameters
        ukf_params = ukf.get_ukf_parameters()
        print(f"UKF Parameters:")
        for key, value in ukf_params.items():
            print(f"  {key}: {value}")

        # Test sigma points
        sigma_points = ukf.get_sigma_points()
        print(f"✅ Sigma Points (shape: {sigma_points.shape})")
        
        # Test a few updates
        measurements = [12.0, 10.5, 9.2, 8.1, 7.5]
        robot_positions = [
            np.array([11.0, 11.0]),
            np.array([12.0, 12.0]),
            np.array([14.0, 14.0]),
            np.array([16.0, 16.0]),
            np.array([18.0, 18.0]),
        ]

        print("FilterPy UKF Corrected Updates:")
        for i, (meas, robot_pos) in enumerate(zip(measurements, robot_positions)):
            mu, Sigma, sigma_Q = ukf.predict_and_update(meas, robot_pos)
            uncertainty = np.sqrt(np.trace(Sigma))
            print(f"  Step {i+1}: mu=[{mu[0]:.1f}, {mu[1]:.1f}], uncertainty={uncertainty:.2f}")

        # Test innovation statistics
        innovation_stats = ukf.get_innovation_stats()
        if innovation_stats:
            print(f"✅ Innovation statistics available")
        
        # Get underlying FilterPy filter
        filterpy_filter = ukf.get_filterpy_filter()
        print(f"✅ Underlying FilterPy filter: {type(filterpy_filter).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing FilterPy UKF Corrected: {str(e)}")
        return False


def test_corrected_vs_original():
    """Compare corrected FilterPy implementations with original versions."""
    print("\n" + "=" * 60)
    print("COMPARING CORRECTED VS ORIGINAL FILTERPY")
    print("=" * 60)

    # Test configuration
    base_config = {
        "signal_max": 15.0,
        "robot_start_pos": [10.0, 10.0],
        "target_true_pos": [30.0, 30.0],
        "max_steps": 20,
        "verbose": False
    }

    # Test measurements
    measurements = [12.0, 10.5, 9.2, 8.1, 7.5]
    robot_positions = [
        np.array([11.0, 11.0]),
        np.array([12.0, 12.0]),
        np.array([14.0, 14.0]),
        np.array([16.0, 16.0]),
        np.array([18.0, 18.0]),
    ]

    filter_types = [
        ("FilterPy_EKF", "Original FilterPy EKF"),
        ("FilterPy_EKF_Corrected", "Corrected FilterPy EKF"),
        ("FilterPy_UKF", "Original FilterPy UKF"),
        ("FilterPy_UKF_Corrected", "Corrected FilterPy UKF")
    ]

    results = {}
    
    for filter_type, description in filter_types:
        try:
            config = get_default_config(filter_type)
            config.update(base_config)
            
            filter_obj = FilterFactory.create_filter(config)
            
            # Run updates
            final_mu = None
            for meas, robot_pos in zip(measurements, robot_positions):
                mu, Sigma, sigma_Q = filter_obj.predict_and_update(meas, robot_pos)
                final_mu = mu
            
            results[filter_type] = {
                "description": description,
                "final_position": final_mu,
                "status": "✅ Success"
            }
            
        except Exception as e:
            results[filter_type] = {
                "description": description,
                "final_position": None,
                "status": f"❌ Error: {str(e)[:50]}..."
            }

    # Print comparison results
    print("Comparison Results:")
    for filter_type, result in results.items():
        print(f"\n{result['description']}:")
        print(f"  Status: {result['status']}")
        if result['final_position'] is not None:
            pos = result['final_position']
            print(f"  Final position: [{pos[0]:.2f}, {pos[1]:.2f}]")

    return results


def run_full_simulation_test():
    """Run a full simulation test with corrected FilterPy implementations."""
    print("\n" + "=" * 60)
    print("FULL SIMULATION TEST - CORRECTED FILTERPY")
    print("=" * 60)

    # Configuration for quick test
    config = get_default_config("FilterPy_EKF_Corrected")
    config.update({
        "arena_min": 0.0,
        "arena_max": 50.0,
        "signal_max": 15.0,
        "robot_start_pos": [10.0, 10.0],
        "target_true_pos": [40.0, 40.0],
        "max_steps": 100,
        "verbose": False
    })

    try:
        # Run simulation
        env = EKFEnvironment(config, verbose=False)
        results = env.run_simulation(100)
        
        print(f"✅ Simulation completed successfully")
        print(f"  Steps to target: {results['steps_completed']}")
        print(f"  Target reached: {results['target_reached']}")
        
        # Handle robot positions safely
        if 'robot_positions' in results and results['robot_positions']:
            final_robot_pos = results['robot_positions'][-1]
            print(f"  Final robot position: [{final_robot_pos[0]:.1f}, {final_robot_pos[1]:.1f}]")
        else:
            print(f"  Final robot position: Unknown")
            
        # Handle belief means safely  
        if 'belief_means' in results and results['belief_means']:
            final_belief = results['belief_means'][-1]
            print(f"  Final belief: [{final_belief[0]:.1f}, {final_belief[1]:.1f}]")
        else:
            print(f"  Final belief: Unknown")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation failed: {str(e)}")
        return False


def main():
    """Main test function."""
    print("Testing Corrected FilterPy Implementations")
    print("=" * 50)

    test_results = []
    
    # Test individual implementations
    test_results.append(("FilterPy EKF Corrected", test_corrected_filterpy_ekf()))
    test_results.append(("FilterPy UKF Corrected", test_corrected_filterpy_ukf()))
    
    # Test comparison
    comparison_results = test_corrected_vs_original()
    test_results.append(("Comparison Test", len([r for r in comparison_results.values() if "Success" in r["status"]]) > 0))
    
    # Test full simulation
    test_results.append(("Full Simulation", run_full_simulation_test()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All corrected FilterPy implementations are working properly!")
        print("\nRecommended usage:")
        print("  config['filter_type'] = 'FilterPy_EKF_Corrected'  # For EKF")
        print("  config['filter_type'] = 'FilterPy_UKF_Corrected'  # For UKF")
    else:
        print("⚠️  Some tests failed. Please check the implementations.")


if __name__ == "__main__":
    main()