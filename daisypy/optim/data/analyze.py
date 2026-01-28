import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from daisypy.optim import (
    animate_result,
    plot_convergence,
    plot_result,
    plot_result_nd
)

def main(run_dir):
    run_dir = Path(run_dir)
    result_file = run_dir / 'logs' / 'result.csv'

    ani = animate_result(result_file)
    plt.show()

    fig, ax = plot_result(result_file)
    plt.show()

    fig, ax = plot_result_nd(result_file, ('K_aquitard', 'Z_aquitard'))
    plt.show()

    fig, ax = plot_convergence(result_file)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help='Path to run dir')
    args = parser.parse_args()
    main(run_dir=args.run_dir)
