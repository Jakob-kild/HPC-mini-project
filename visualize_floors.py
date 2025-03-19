#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_building_floor(data_dir, building_id):
    """
    Loads and visualizes the domain and interior mask for a single building.
    """
    domain_file = os.path.join(data_dir, f"{building_id}_domain.npy")
    interior_file = os.path.join(data_dir, f"{building_id}_interior.npy")

    # Load data arrays
    domain = np.load(domain_file)
    interior = np.load(interior_file)

    # Visualize the domain (initial temperatures)
    plt.imshow(domain, cmap="viridis")
    plt.title(f"Domain (Initial Temperatures) - Building {building_id}")
    plt.colorbar(label="Temperature (°C)")
    plt.show()

    # Visualize the interior mask
    plt.imshow(interior, cmap="gray")
    plt.title(f"Interior Mask - Building {building_id}")
    plt.show()


def main():
    """
    Main script to visualize multiple floorplans from the "test_floors" folder.

    Adjust the building_ids list to match the files you have in test_floors.
    """
    data_dir = "test_floors"

    # Update this list if you add or remove building samples
    building_ids = ["10000", "10009", "10014", "10019"]

    for bid in building_ids:
        visualize_building_floor(data_dir, bid)


if __name__ == "__main__":
    main()
