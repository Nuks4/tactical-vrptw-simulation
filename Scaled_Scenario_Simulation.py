import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from pyvrp import Model
from pyvrp.stop import MaxRuntime

# --- Configuration & Constants ---
counts = {"low": 25, "medium": 50, "high": 75}
iterations = 10
SPEED_MPM = 333.33
W_LABOR = 25.0
C_DIST_METER = 0.5 / 1000
pool_size = 5000
FIXED_TRUCKS = 5
TRUCK_CAPACITY = 30

# --- Equal-Density Scaled Service Area ---
R_INNER = 3000 * np.sqrt(2.5)
R_OUTER = 3000 * np.sqrt(5)

# --- Scaling Factors for PyVRP Integer Precision ---
# Base Currency: Hundredths of a cent ($0.0001)
S_D = 1    # 1 meter
S_T = 10   # 0.1 minute

# Cost Constants:
# Distance: $0.0005 per meter = 5 units ($0.0005 / $0.0001)
# Duration: ($25/60)/10 per 0.1 min = $0.041667 -> 417 units
UNIT_DIST_COST = 5
UNIT_DUR_COST = 417

# --- Operational & Time Slot Setup ---
slot_full, slot_1, slot_2 = [0, 360], [0, 180], [180, 360]

depot_locations = {
    "center": [0.0, 0.0],
    "in-between": [R_INNER, 0.0],  # Edge of the inner zone (~4.74 km)
    "edge": [R_OUTER, 0.0]        # Edge of the outer zone (~6.71 km)
}

time_configs = {
    "A": ("single", "single"),
    "B": ("double", "double"),
    "C": ("double", "single"),
    "D": ("single", "double")
}

# --- PyVRP Solver Function ---


def solve_pyvrp(scenario_data, time_mtx, dist_mtx, num_trucks, truck_capacity):
    model = Model()
    depot_obj = model.add_depot(
        x=int(scenario_data[0, 0] * S_D), y=int(scenario_data[0, 1] * S_D))
    locations = [depot_obj]

    for i in range(1, len(scenario_data)):
        row = scenario_data[i]
        locations.append(model.add_client(
            x=int(row[0] * S_D), y=int(row[1] * S_D),
            delivery=[1],
            service_duration=int(8 * S_T),  # 8 minutes service time
            tw_early=int(row[2] * S_T),
            tw_late=int(row[3] * S_T)))

    model.add_vehicle_type(
        num_available=num_trucks,
        capacity=[truck_capacity],
        fixed_cost=0,
        unit_distance_cost=UNIT_DIST_COST,
        unit_duration_cost=UNIT_DUR_COST
    )

    dists = (dist_mtx * S_D).astype(int)
    times = (time_mtx * S_T).astype(int)
    for i in range(len(locations)):
        for j in range(len(locations)):
            model.add_edge(locations[i], locations[j],
                           distance=dists[i, j], duration=times[i, j])

    # Time-based stopping criterion (5 seconds per scenario)
    res = model.solve(stop=MaxRuntime(5), display=False, seed=42)

    if res.best.is_feasible():
        route_timings = []
        for route in res.best.routes():
            route_timings.append({
                "start": route.start_time() / S_T,
                "end": route.end_time() / S_T
            })

        return {
            "total_cost": res.cost() / 10000.0,  # Convert hundredths of a cent to Euro
            "distance": res.best.distance() / S_D,
            "duration": res.best.duration() / S_T,
            "route_timings": route_timings,
            "feasible": True
        }
    return {"feasible": False}

# --- Data Generation Functions ---


def generate_coordinate_pool(n, r_min, r_max, seed=42):
    np.random.seed(seed)
    r = np.sqrt(np.random.uniform(r_min**2, r_max**2, n))
    theta = np.random.uniform(0, 2 * np.pi, n)
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


def assign_timeslots(n, mode, slot_full, slot_1, slot_2, rng_state):
    timeslots = np.zeros((n, 2))
    if mode == "single":
        timeslots[:] = slot_full
    else:
        indices = np.arange(n)
        rng_state.shuffle(indices)
        mid = n // 2
        timeslots[indices[:mid]] = slot_1
        timeslots[indices[mid:]] = slot_2
    return timeslots


# --- Create Customer Pools with Equal-Density Radii ---
inner_pool = generate_coordinate_pool(pool_size, 0, R_INNER, seed=123)
outer_pool = generate_coordinate_pool(pool_size, R_INNER, R_OUTER, seed=456)

# --- Simulation Result Storage ---
all_results = []
all_customers_master = []

# --- Simulation Execution ---
print(
    f"Starting 5x area/customer simulation with fixed fleet of {FIXED_TRUCKS} trucks...")

for i in range(iterations):
    loop_rng = np.random.RandomState(seed=i)

    for in_label, in_n in counts.items():
        for out_label, out_n in counts.items():
            in_idx = loop_rng.choice(pool_size, size=in_n, replace=False)
            out_idx = loop_rng.choice(pool_size, size=out_n, replace=False)
            curr_in_coords = inner_pool[in_idx]
            curr_out_coords = outer_pool[out_idx]

            for t_label, (in_mode, out_mode) in time_configs.items():
                in_times = assign_timeslots(
                    in_n, in_mode, slot_full, slot_1, slot_2, loop_rng)
                out_times = assign_timeslots(
                    out_n, out_mode, slot_full, slot_1, slot_2, loop_rng)

                customer_data = np.vstack((
                    np.hstack((curr_in_coords, in_times)),
                    np.hstack((curr_out_coords, out_times))
                ))

                for d_label, d_coords in depot_locations.items():
                    for row in customer_data:
                        all_customers_master.append({
                            "Iteration": i, "In_Density": in_label, "Out_Density": out_label,
                            "Time_Config": t_label, "Depot_Loc": d_label,
                            "X": row[0], "Y": row[1], "TW_Start": row[2], "TW_End": row[3]
                        })
                    depot_row = np.array([[d_coords[0], d_coords[1], 0, 360]])
                    full_scenario_data = np.vstack((depot_row, customer_data))

                    dist_mtx = distance_matrix(
                        full_scenario_data[:, :2], full_scenario_data[:, :2])
                    time_mtx = dist_mtx / SPEED_MPM

                    res = solve_pyvrp(full_scenario_data, time_mtx, dist_mtx,
                                      num_trucks=FIXED_TRUCKS, truck_capacity=TRUCK_CAPACITY)

                    result_entry = {
                        "Iteration": i, "In_Density": in_label, "Out_Density": out_label,
                        "Time_Config": t_label, "Depot_Loc": d_label, "Trucks": FIXED_TRUCKS,
                        "Feasible": res["feasible"]}

                    if res["feasible"]:
                        result_entry.update({
                            "Total_Cost": res["total_cost"],
                            "Distance_m": res["distance"],
                            "Duration_min": res["duration"]
                        })
                        for idx, timing in enumerate(res["route_timings"]):
                            result_entry[f"Truck_{idx+1}_Start"] = timing["start"]
                            result_entry[f"Truck_{idx+1}_End"] = timing["end"]

                    all_results.append(result_entry)

    print(f"Iteration {i+1}/{iterations} complete.")

# --- Save Results ---
df = pd.DataFrame(all_results)
df.to_excel("simulation_results_5x.xlsx", index=False)
master_cust_df = pd.DataFrame(all_customers_master)
master_cust_df.to_csv("scenarios_customer_list_5x.txt", sep="\t", index=False)
print("Simulation complete! Outputs saved to simulation_results_5x.xlsx and scenarios_customer_list_5x.txt.")
