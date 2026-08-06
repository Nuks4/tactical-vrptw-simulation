import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from pyvrp import Model
from pyvrp.stop import MaxIterations

# --- Configuration & Constants ---
counts = {"low": 5, "medium": 10, "high": 15}
iterations = 10
SPEED_MPM = 333.33
W_LABOR = 25.0
C_DIST_METER = 0.5 / 1000
pool_size = 5000

# --- Scaling Factors ---
S_D = 10
S_T = 100
S_OBJ = 1000000

# --- Operational & Time Slot Setup ---
slot_full, slot_1, slot_2 = [0, 360], [0, 180], [180, 360]

depot_locations = {
    "center": [0, 0],
    "in-between": [2121.32, 0],
    "edge": [3000, 0]
}

time_configs = {
    "A": ("single", "single"),
    "B": ("double", "double"),
    "C": ("double", "single"),
    "D": ("single", "double")
}

# --- Pyvrp Solver Function ---


def solve_pyvrp(scenario_data, time_mtx, dist_mtx, w_labor, c_dist_m, num_trucks):
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

    cost_per_min = (w_labor / 60)

    unit_dist_cost = int((S_OBJ / S_D) * c_dist_m)
    unit_dur_cost = int((S_OBJ / S_T) * cost_per_min)

    model.add_vehicle_type(
        num_available=num_trucks,
        capacity=[len(scenario_data) + 1],
        fixed_cost=0,
        unit_distance_cost=unit_dist_cost,
        unit_duration_cost=unit_dur_cost
    )

    dists = (dist_mtx * S_D).astype(int)
    times = (time_mtx * S_T).astype(int)
    for i in range(len(locations)):
        for j in range(len(locations)):
            model.add_edge(locations[i], locations[j],
                           distance=dists[i, j], duration=times[i, j])

    res = model.solve(stop=MaxIterations(5000), display=False, seed=42)

    if res.best.is_feasible():
        route_timings = []
        for route in res.best.routes():
            route_timings.append({
                "start": route.start_time() / S_T,
                "end": route.end_time() / S_T
            })

        return {
            "total_cost": res.cost() / S_OBJ,  # Simple division by S_OBJ!
            "distance": res.best.distance() / S_D,
            "duration": res.best.duration() / S_T,
            "route_timings": route_timings,
            "feasible": True
        }
    return {"feasible": False}

# --- Data Generation ---


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


# --- Create Customer Pools ---
inner_pool = generate_coordinate_pool(pool_size, 0, 2121.32, seed=123)
outer_pool = generate_coordinate_pool(pool_size, 2121.32, 3000, seed=456)

# --- Simulation Result Storage ---
all_results = []
all_customers_master = []

# --- Simulation Execution ---
print(f"Starting simulation: {iterations} iterations...")

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

                    for num_v in [1, 2]:
                        res = solve_pyvrp(full_scenario_data, time_mtx, dist_mtx,
                                          W_LABOR, C_DIST_METER, num_trucks=num_v)

                        result_entry = {
                            "Iteration": i, "In_Density": in_label, "Out_Density": out_label,
                            "Time_Config": t_label, "Depot_Loc": d_label, "Trucks": num_v,
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

# --- Save results ---
df = pd.DataFrame(all_results)
df.to_excel("simulation_results.xlsx", index=False)
master_cust_df = pd.DataFrame(all_customers_master)
master_cust_df.to_csv(
    "scenarios_customer_list.txt", sep="\t", index=False)
