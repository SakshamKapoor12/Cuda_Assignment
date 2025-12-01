import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sys

def plot_nbody(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print("Error: CSV file not found. Run the CUDA program first.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot initial state (step 0)
    initial = df[df['step'] == 0]
    ax.scatter(initial['x'], initial['y'], initial['z'], c='blue', alpha=0.1, label='Start')

    # Plot final state
    max_step = df['step'].max()
    final = df[df['step'] == max_step]
    ax.scatter(final['x'], final['y'], final['z'], c='red', s=20, label='End')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(f'N-Body Simulation: {len(initial)} Particles')
    ax.legend()
    
    output_file = 'simulation_result.png'
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    plot_nbody("nbody_output.csv")