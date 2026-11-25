import numpy as np  # I use numpy for the arrays and the exact solution
import matplotlib.pyplot as plt  # this makes the plots
from scipy.integrate import solve_ivp  # this solves both differential equations together
from pathlib import Path  # this keeps the saved files in one folder


def get_number(prompt, default, low, high):
    while True:  # I keep asking if the input does not work
        try:
            entry = input(f"{prompt} [{default}]: ").strip()  # enter uses the default
            value = float(entry) if entry else default
            if np.isfinite(value) and low <= value <= high:  # nan and infinity do not count
                return value
            print(f"Enter a number from {low} to {high}.")
        except ValueError:
            print("Enter a number instead of text.")


def get_choice(prompt, options, default):
    while True:
        choice = input(f"{prompt} [{default}]: ").strip() or default
        if choice in options:  # only the listed choices are allowed
            return options[choice]
        print("Choose one of these: " + ", ".join(options))


def polymer_odes(t, concentrations, k):
    M, P = concentrations  # M is styrene and P is styrene mass incorporated into polymer, both in g/L
    rate = k * M  # k is an effective constant in 1/min, so rate is in g/L/min
    dM_dt = -rate  # monomer gets used up
    dP_dt = rate  # that same mass gets added to the polymer
    return [dM_dt, dP_dt]  # these are two mass balances, not a chain-count model


def solve_model(M0, k, end_time, method):
    time = np.linspace(0, end_time, 301)  # these are the times where I want results
    solution = solve_ivp(
        polymer_odes, [0, end_time], [M0, 0.0], args=(k,),
        t_eval=time, method=method, rtol=1e-7, atol=1e-9 * M0,
    )  # I start with monomer and no polymer
    if not solution.success:
        raise RuntimeError(solution.message)  # I do not use a failed solver result
    if not np.all(np.isfinite(solution.y)):
        raise RuntimeError("The solver returned invalid numbers.")
    return solution.t, solution.y[0], solution.y[1]


def check_results(time, M, P, M0, k):
    M_exact = M0 * np.exp(-k * time)  # this comes from separating dM/dt = -kM
    P_exact = -M0 * np.expm1(-k * time)  # same as M0*(1-exp(-kt)), but better for small kt
    mass_error = np.max(np.abs(M + P - M0)) / M0  # largest mass balance error relative to M0
    exact_error = max(np.max(np.abs(M - M_exact)), np.max(np.abs(P - P_exact))) / M0
    tolerance = 1e-5  # the check limit is 0.001 percent of the starting concentration
    physical = np.all((M >= -tolerance * M0) & (M <= (1 + tolerance) * M0))
    physical = physical and np.all((P >= -tolerance * M0) & (P <= (1 + tolerance) * M0))
    monotonic = np.all(np.diff(M) <= tolerance * M0) and np.all(np.diff(P) >= -tolerance * M0)
    concentration_units = np.array([1, -1, 0])  # powers of g, L, and min for g/L
    time_units = np.array([0, 0, 1])  # min has a power of one
    k_units = np.array([0, 0, -1])  # k has units of 1/min
    units_ok = np.array_equal(k_units + concentration_units, concentration_units - time_units)
    checks = {
        "Declared units: k*M and dM/dt (also dP/dt)": units_ok,
        "Mass conservation: M + P = M0": mass_error <= tolerance,
        "Agreement with exact solution (both equations)": exact_error <= tolerance,
        "Physical bounds: 0 <= M,P <= M0": physical,
        "Monomer decreases and polymer increases": monotonic,
    }  # a constant result also passes the direction check when k is zero
    return checks, mass_error, exact_error


def make_plot(time, M, P, M0, k, target, plot_choice, output_folder):
    columns = 2 if plot_choice == "both" else 1
    fig, axes = plt.subplots(1, columns, figsize=(6 * columns, 4.5), squeeze=False)
    if plot_choice in ["concentrations", "both"]:
        ax = axes[0, 0]
        ax.plot(time, M, label="Styrene monomer", linewidth=2)
        ax.plot(time, P, label="Polystyrene (monomer-derived mass)", linewidth=2)
        ax.set_ylabel("Mass concentration (g/L)")
        ax.set_title("Monomer consumption and polymer formation")
    if plot_choice in ["conversion", "both"]:
        ax = axes[0, -1]
        ax.plot(time, 100 * (1 - M / M0), label="Monomer conversion", linewidth=2)
        ax.axhline(target, color="gray", linestyle="--", label=f"Target: {target:g}%")
        ax.set_ylabel("Conversion (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Conversion over time")
    for ax in axes.flat:
        ax.set_xlabel("Time (min)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Styrene to polystyrene | illustrative k = {k:g} min^-1")
    fig.tight_layout()
    fig.savefig(output_folder / "polymer_plot.png", dpi=180)  # I save the plot before opening it
    plt.show()
    plt.close(fig)


def main():
    print("\nSTYRENE TO POLYSTYRENE: BATCH CONVERSION MODEL")
    print("Constant temperature, volume, and effective reaction rate constant.")
    print("Defaults are made-up teaching inputs, not measured styrene kinetics.")
    print("Use g/L and minutes. Press enter to use each default.\n")
    M0 = get_number("Starting styrene concentration, g/L", 100.0, 0.001, 1000)
    k = get_number("Effective rate constant, 1/min", 0.05, 0, 1)
    end_time = get_number("Simulation time, min", 60.0, 0.001, 1000)
    target = get_number("Target monomer conversion, %", 90.0, 1, 99.9)
    method = get_choice("Solver: 1 = RK45, 2 = RK23", {"1": "RK45", "2": "RK23"}, "1")
    plot_choice = get_choice(
        "Plot: 1 = concentrations, 2 = conversion, 3 = both",
        {"1": "concentrations", "2": "conversion", "3": "both"}, "3",
    )  # these choices do not require changing the code
    time, M, P = solve_model(M0, k, end_time, method)
    conversion = 100 * (1 - M / M0)  # this is based on monomer consumed
    checks, mass_error, exact_error = check_results(time, M, P, M0, k)
    lines = [
        "STYRENE TO POLYSTYRENE - SIMULATION REPORT",
        "Teaching model; illustrative inputs are not experimentally validated kinetics.",
        f"M0 = {M0:g} g/L; P0 = 0 g/L; k = {k:g} 1/min; time = {end_time:g} min",
        f"Solver = {method}; rtol = 1e-7; atol = {1e-9 * M0:g} g/L",
        "",
        f"{'Time (min)':>12} {'Monomer (g/L)':>16} {'Polymer (g/L)':>16} {'Conversion (%)':>17}",
    ]
    for i in np.linspace(0, len(time) - 1, 11, dtype=int):  # I show 11 rows to keep the table readable
        lines.append(f"{time[i]:12.3f} {M[i]:16.6f} {P[i]:16.6f} {conversion[i]:17.4f}")
    lines.extend(["", "QUALITY CHECKS (evaluated at the 301 output times)"])
    for name, passed in checks.items():
        lines.append(f"{'PASS' if passed else 'FAIL'}: {name}")
    lines.extend([
        f"Maximum mass balance error / M0: {mass_error:.3e}",
        f"Maximum exact-solution error / M0: {exact_error:.3e}",
        "Numerical check limit: 1e-5 of M0; small negative roundoff is tolerated.",
        "Dimensional check: (1/min)*(g/L) = g/(L*min) for both balances.",
        "The dimensional check uses declared units; it cannot detect incorrectly entered units.",
        "",
    ])
    if k == 0:
        lines.append("No reaction occurs when k = 0; the positive conversion target is never reached.")
    else:
        target_time = -np.log1p(-target / 100) / k  # this exact estimate comes from X = 1-exp(-kt)
        lines.append(f"Exact-model time to {target:g}% conversion: {target_time:.3f} min")
        lines.append("Target falls within this run." if target_time <= end_time else "Target falls after this run.")
    if not all(checks.values()):
        lines.append("CHECK FAILED: review the equations, inputs, and solver settings before using these results.")
    output_folder = Path.cwd() / "polymer_results"  # rerunning replaces files in this results folder
    output_folder.mkdir(exist_ok=True)
    data = np.column_stack((time, M, P, conversion))
    np.savetxt(
        output_folder / "polymer_results.csv", data, delimiter=",", fmt="%.10g",
        header="time_min,styrene_g_per_L,polymer_monomer_derived_g_per_L,conversion_percent", comments="",
    )  # the csv has all 301 rows, so it can also be opened in Excel
    report = "\n".join(lines)
    (output_folder / "quality_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nResults saved in: {output_folder}")
    make_plot(time, M, P, M0, k, target, plot_choice, output_folder)


if __name__ == "__main__":
    try:
        main()  # this starts the input menu when the file is run
    except (RuntimeError, OSError) as error:
        print(f"Could not complete the simulation: {error}")
    except (KeyboardInterrupt, EOFError):
        print("\nSimulation cancelled.")
