#########################################################################################
# Subset to files that are within 3 hours of a buoy overpass to cut down on processing time
#########################################################################################

# List processed S1 and SWOT files 
s1_files = glob.glob(os.path.join(home, "Data/Sentinel1/Processed/*.nc"))
swot2_files = glob.glob(os.path.join(home, "Data/SWOT_L2_HR_Raster/SWOT_L2_HR_Raster_2.0/*.nc"))

# Df of buoy intersection with satellite images 
buoy_sat_int = pd.read_csv(os.path.join(home, "Data", "buoy_satellite_intersection.csv"))

# Find the index of images with a buoy observation within 3 hours 
index = []
for i in range(0, buoy_sat_int.shape[0]):
    print(i)
    sub = buoy_sat_int.iloc[i]
    buoy_id = sub.buoy_id
    buoy_df = pd.read_csv(os.path.join(home, "Data/Buoy/Processed", buoy_id + ".csv"))
    buoy_df['datetime']= pd.to_datetime(buoy_df['datetime'], utc=True, format='ISO8601')

    satellite = sub.satellite
    image_id = os.path.basename(sub.image_id)[:-3]
    if satellite == "Sentinel1": overpass_datetime = datetime.strptime(image_id.split("_")[5], "%Y%m%dT%H%M%S")
    else: overpass_datetime = datetime.strptime(image_id.split("_")[13], "%Y%m%dT%H%M%S")
    overpass_datetime = overpass_datetime.replace(tzinfo=timezone.utc)

    diffs = (buoy_df['datetime'] - overpass_datetime).tolist()
    # Just want differences less than 0 so that the buoy observation is before overpass time 
    diffs_less0 = [t for t in diffs if t < pd.Timedelta(0)]
    # To find the closest to 0, get the max
    if len(diffs_less0) == 0: 
        continue
    else:
        min_diff = abs(max(diffs_less0))
        if min_diff < timedelta(hours=3): 
            index.append(i)
        else: continue

# Names of images within 3 hours of buoy observation
buoy_sat_int_sub = buoy_sat_int.iloc[index]
all_images = buoy_sat_int_sub['image_id'].tolist()
all_images = [os.path.basename(i) for i in all_images]

# Subset S1 nd SWOT 2.0 files using those image names
s1_files = [file for file in s1_files if any([scene in file for scene in all_images])]
swot2_files = [file for file in swot2_files if any([scene in file for scene in all_images])]








def calc_performance_stats_360(df):
    """
    Calculate bias, mae, rmse for wind direction and wind speed
    df: pd df
    return df of metrics
    """
    def calc_bias(o, p, m):
        o_upper = (o + 180) % 360
        if o_upper < o:
            if (p > o) | (p < o_upper):
                return m
            else:
                return m * -1
        if o_upper > o:
            if (p > o) & (p < o_upper):
                return m
            else:
                return m * -1

    # LG Wind Direction
    sub = df.dropna(subset=["wdir_corrected", "wdir_buoy"])
    if sub.shape[0] < 3:
        row1 = pd.DataFrame(
            [["WDIR-LG", sub.shape[0], np.nan, np.nan, np.nan]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    else:
        obs = sub["wdir_buoy"].tolist()
        pred = sub["wdir_corrected"].tolist()
        # MAE. Get the smallest difference
        d1 = [(p_i - o_i) % 360 for p_i, o_i in zip(pred, obs)]
        d2 = [360 - i for i in d1]
        mae = [min(i, j) for i, j in zip(d1, d2)]
        # Bias. If the predicted is within + 180 from the observed, it is positive. 
        # If the predicted is within - 180 from the observed, it is negative
        bias = [calc_bias(o, p, m) for o, p, m in zip(obs, pred, mae)]
        # Root mean square error
        rmse = math.sqrt(np.square(bias).mean())
        bias = np.mean(bias)
        mae = np.mean(mae)
        row1 = pd.DataFrame(
            [["WDIR-LG", sub.shape[0], bias, mae, rmse]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )

    # ERA5 Wind Direction. Use the same subset as for the LG wind direction so it's a direct comparison
    if sub.shape[0] < 3:
        row2 = pd.DataFrame(
            [["WDIR-ERA5", sub.shape[0], np.nan, np.nan, np.nan]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    else:
        obs = sub["wdir_buoy"].tolist()
        pred = sub["wdir_era5"].tolist()
        # MAE. Get the smallest difference
        d1 = [(p_i - o_i) % 360 for p_i, o_i in zip(pred, obs)]
        d2 = [360 - i for i in d1]
        mae = [min(i, j) for i, j in zip(d1, d2)]
        # Bias. If the predicted is within + 180 from the observed, it is positive. If the predicted is within - 180 from the observed, it is negative
        bias = [calc_bias(o, p, m) for o, p, m in zip(obs, pred, mae)]
        # Root mean square error
        rmse = math.sqrt(np.square(bias).mean())
        bias = np.mean(bias)
        mae = np.mean(mae)
        row2 = pd.DataFrame(
            [["WDIR-ERA5", sub.shape[0], bias, mae, rmse]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )

    # CMOD5.n LG wind speed
    sub = df.dropna(subset=["wspd_sat_cmod5n", "wspd_buoy"])
    if sub.shape[0] < 3:
        row3 = pd.DataFrame(
            [["WSPD-CMOD5-SAR", sub.shape[0], np.nan, np.nan, np.nan]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    else:
        obs = sub["wspd_buoy"].tolist()
        pred = sub["wspd_sat_cmod5n"].tolist()
        bias = np.mean(np.subtract(pred, obs))
        mae = np.mean(np.abs(np.subtract(pred, obs)))
        rmse = math.sqrt(np.square(np.subtract(pred, obs)).mean())
        row3 = pd.DataFrame(
            [["WSPD-CMOD5-SAR", sub.shape[0], bias, mae, rmse]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )

    # CMOD5.n ERA5 wind speed
    if sub.shape[0] < 3:
        row4 = pd.DataFrame(
            [["WSPD-CMOD5-ERA5", sub.shape[0], np.nan, np.nan, np.nan]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    else:
        obs = sub["wspd_buoy"].tolist()
        pred = sub["wspd_era5_cmod5n"].tolist()
        bias = np.mean(np.subtract(pred, obs))
        mae = np.mean(np.abs(np.subtract(pred, obs)))
        rmse = math.sqrt(np.square(np.subtract(pred, obs)).mean())
        row4 = pd.DataFrame(
            [["WSPD-CMOD5-ERA5", sub.shape[0], bias, mae, rmse]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )

    # ERA5 wind speed
    if sub.shape[0] < 3:
        row5 = pd.DataFrame(
            [["WSPD-ERA5", sub.shape[0], np.nan, np.nan, np.nan]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )
    else:
        obs = sub["wspd_buoy"].tolist()
        pred = sub["wspd_era5"].tolist()
        bias = np.mean(np.subtract(pred, obs))
        mae = np.mean(np.abs(np.subtract(pred, obs)))
        rmse = math.sqrt(np.square(np.subtract(pred, obs)).mean())
        row5 = pd.DataFrame(
            [["WSPD-ERA5", sub.shape[0], bias, mae, rmse]],
            columns=["ID", "N", "Bias", "MAE", "RMSE"],
        )

    df = pd.concat([row1, row2, row3, row4, row5], axis=0)
    return df








###############################################################################################
# Supplmentary Figure 1
# Assess if there are significant relationships between buoy performance and lake/buoy attributes
###############################################################################################

# For each buoy + satellite + wind streak combo, calcualte the average wind speed and fetch for the datetimes
# that we are making a wind direction estimate
# List to store the avg wind and fetch for each buoy
buoy_avgs = []
# Only include predictions with ME < 30
me_limit = [30]
# List unique buoys
buoys = set(wind_df["buoy_id"])
# Lit the satellites
satellites = ["S1", "SWOT"]
# Create unique combos to loop through
combos = list(itertools.product(*[buoys, satellites, me_limit]))
# Loop through combos and calculate the average wind speed and fetch and store results in df that are appended to list
for combo in combos:
    subset = wind_df[
        (
            (wind_df["satellite"] == combo[1])
            & (wind_df["buoy_id"] == combo[0])
            & (wind_df["me"] <= combo[2])
        )
    ]
    subset_ws = wind_df_ws[
        (
            (wind_df_ws["satellite"] == combo[1])
            & (wind_df_ws["buoy_id"] == combo[0])
            & (wind_df_ws["me"] <= combo[2])
        )
    ]
    if subset.shape[0] == 0:
        continue
    if subset.shape[0] > 0:
        avg_wspd = subset["buoy_wspd_10m"].mean()
        avg_fetch_lake = subset["fetch_lake"].mean()
        avg_fetch_buoy = subset["fetch_buoy"].mean()
        all_df = pd.DataFrame(
            [
                [
                    combo[1],
                    combo[0],
                    "All",
                    combo[2],
                    avg_wspd,
                    avg_fetch_lake,
                    avg_fetch_buoy,
                ]
            ],
            columns=[
                "satellite",
                "buoy_id",
                "Wind_Streak",
                "ME_Limit",
                "avg_wspd",
                "avg_fetch_lake",
                "avg_fetch_buoy",
            ],
        )
    if subset_ws.shape[0] > 0:
        avg_wspd = subset_ws["buoy_wspd_10m"].mean()
        avg_fetch_lake = subset_ws["fetch_lake"].mean()
        avg_fetch_buoy = subset_ws["fetch_buoy"].mean()
        ws_df = pd.DataFrame(
            [
                [
                    combo[1],
                    combo[0],
                    "Wind Streaks",
                    combo[2],
                    avg_wspd,
                    avg_fetch_lake,
                    avg_fetch_buoy,
                ]
            ],
            columns=[
                "satellite",
                "buoy_id",
                "Wind_Streak",
                "ME_Limit",
                "avg_wspd",
                "avg_fetch_lake",
                "avg_fetch_buoy",
            ],
        )
        all_df = pd.concat([all_df, ws_df], axis=0)
    buoy_avgs.append(all_df)
# Combine list of dfs to one df
buoy_avgs = pd.concat(buoy_avgs)
# Get the buoy distance to shore and merge to the buoy attributes
buoy_dist_shore = (
    wind_df.groupby(["buoy_id"])[["distance"]].mean().reset_index(drop=False)
)
buoy_atts = buoy_avgs.merge(buoy_dist_shore, on="buoy_id", how="left")
# Add the PLD ID associated with each buoy and then associate the PLD_ID reference area
buoy_atts = buoy_atts.merge(
    buoy_info.rename(columns={"id": "buoy_id"})[["buoy_id", "type", "pld_id"]],
    on="buoy_id",
    how="left",
)
buoy_atts = buoy_atts.merge(
    pld[["lake_id", "ref_area"]], left_on="pld_id", right_on="lake_id", how="left"
)
# Rename the two columns for merging with wind speed and direction buoy performance dfs
buoy_atts = buoy_atts.rename(columns={"buoy_id": "Buoy", "satellite": "Satellite"})
# Merge the buoy/lake atts with buoy performance dfs
buoy_perf_30_wdir = buoy_perf_30_wdir.merge(
    buoy_atts, on=["Satellite", "Buoy", "Wind_Streak", "ME_Limit"], how="left"
)
buoy_perf_30_wspd = buoy_perf_30_wspd.merge(
    buoy_atts, on=["Satellite", "Buoy", "Wind_Streak", "ME_Limit"], how="left"
)

# Wind direction: calculate the slope, R, and pvalue for the relationship between buoy MAE and the lake/buoy attributes
vars = ["avg_wspd", "avg_fetch_lake", "avg_fetch_buoy", "ref_area", "distance"]
metrics = ["MAE"]
wss = ["All", "Wind Streaks"]
ids = ["LG-Mod"]
satellites = ["SWOT", "S1"]
rels_df = []
for sat in satellites:
    for var in vars:
        for metric in metrics:
            for ws in wss:
                for id in ids:
                    s = buoy_perf_30_wdir[
                        (
                            (buoy_perf_30_wdir["Satellite"] == sat)
                            & (buoy_perf_30_wdir["Wind_Streak"] == ws)
                            & (buoy_perf_30_wdir["ID"] == id)
                        )
                    ]
                    if s.shape[0] < 3:
                        continue
                    slope, intercept, r_value, p_value, std_err = (
                        scipy.stats.linregress(s[metric].tolist(), s[var].tolist())
                    )
                    rels_df.append(
                        pd.DataFrame(
                            [
                                [
                                    sat,
                                    metric,
                                    var,
                                    ws,
                                    id,
                                    slope,
                                    r_value,
                                    p_value,
                                    s.shape[0],
                                ]
                            ],
                            columns=[
                                "Satellite",
                                "Metric",
                                "Variable",
                                "WindStreak",
                                "ID",
                                "Slope",
                                "R",
                                "Pvalue",
                                "N",
                            ],
                        )
                    )
wdir_rel_df = pd.concat(rels_df)
# For the figure, just look at performance~lake/buoy relationships when wind direction was estimated with and without wind streaks
wdir_rel_df = wdir_rel_df[wdir_rel_df["WindStreak"] == "All"]
# Make it long for plotting
wdir_rel_df_long = pd.melt(
    wdir_rel_df,
    id_vars=["Satellite", "Metric", "Variable", "WindStreak", "ID", "N", "Pvalue"],
    value_vars=["Slope", "R"],
)
# Add a pretty column for the Significance
wdir_rel_df_long["Significance"] = pd.Series(dtype="str")
wdir_rel_df_long.loc[wdir_rel_df_long["Pvalue"] <= 0.05, "Significance"] = "<=0.05"
wdir_rel_df_long.loc[wdir_rel_df_long["Pvalue"] > 0.05, "Significance"] = ">0.05"
# Make the variables pretty for plotting
wdir_rel_df_long["Variable"] = wdir_rel_df_long["Variable"].map(
    {
        "avg_wspd": "Wind speed (m/s)",
        "avg_fetch_lake": "Lake fetch (km)",
        "avg_fetch_buoy": "Buoy fetch (km)",
        "ref_area": "Lake area (km2)",
        "distance": "Buoy distance (km)",
    }
)

# Plot the scatter plot of the R for each relationship grouped by satellite and sig denoted with the plot marker
fig, ((ax2)) = plt.subplots(
    1,
    1,
    figsize=(13.5, 4),
)
markers = {"<=0.05": "s", ">0.05": "X"}
g1 = sns.scatterplot(
    ax=ax2,
    data=wdir_rel_df_long[wdir_rel_df_long["variable"] == "R"],
    x="value",
    y="Variable",
    hue="Satellite",
    palette=["#3b528b", "#5ec962"],
    style="Significance",
    markers=markers,
    s=100,
)
ax2.axhline(
    y="Wind speed (m/s)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1
)
ax2.axhline(y="Lake fetch (km)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1)
ax2.axhline(y="Lake area (km2)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1)
ax2.axhline(
    y="Buoy distance (km)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1
)
ax2.axhline(y="Buoy fetch (km)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1)
ax2.set(xlabel="Correlation Coefficient (R)", ylabel="")
sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1), frameon=False)
# plt.show()
plt.savefig(os.path.join(home, "Data/Figures/wdir_buoy_var_reg.png"), dpi=1000)

###############################################################################################
# Supplmentary Figure 2
# Assess if there are significant relationships between
# buoy wind speed performance and lake/buoy attributes
###############################################################################################

# Wind speed: calculate the slope, R, and pvalue for the relationship between buoy MAE,
# Bias, and RMSE and the lake/buoy attributes
vars = ["avg_wspd", "avg_fetch_lake", "avg_fetch_buoy", "ref_area", "distance"]
metrics = ["MAE", "Bias", "RMSE"]
wss = ["All", "Wind Streaks"]
ids = ["CMOD5.N+LG-Mod"]
satellites = ["S1"]
rels_df = []
for sat in satellites:
    for var in vars:
        for metric in metrics:
            for ws in wss:
                for id in ids:
                    s = buoy_perf_30_wspd[
                        (
                            (buoy_perf_30_wspd["Satellite"] == sat)
                            & (buoy_perf_30_wspd["Wind_Streak"] == ws)
                            & (buoy_perf_30_wspd["ID"] == id)
                        )
                    ]
                    if s.shape[0] < 3:
                        continue
                    slope, intercept, r_value, p_value, std_err = (
                        scipy.stats.linregress(s[metric].tolist(), s[var].tolist())
                    )
                    rels_df.append(
                        pd.DataFrame(
                            [
                                [
                                    sat,
                                    metric,
                                    var,
                                    ws,
                                    id,
                                    slope,
                                    r_value,
                                    p_value,
                                    s.shape[0],
                                ]
                            ],
                            columns=[
                                "Satellite",
                                "Metric",
                                "Variable",
                                "WindStreak",
                                "ID",
                                "Slope",
                                "R",
                                "Pvalue",
                                "N",
                            ],
                        )
                    )

# Combine results into df
wspd_rel_df = pd.concat(rels_df)
# Want to include All for wind streaks
wspd_rel_df = wspd_rel_df[wspd_rel_df["WindStreak"] == "All"]
# Make long for plotting
wspd_rel_df_long = pd.melt(
    wspd_rel_df,
    id_vars=["Satellite", "Metric", "Variable", "WindStreak", "ID", "N", "Pvalue"],
    value_vars=["Slope", "R"],
)
# Create a pretty sig column for plotting
wspd_rel_df_long["Significance"] = pd.Series(dtype="str")
wspd_rel_df_long.loc[wspd_rel_df_long["Pvalue"] <= 0.05, "Significance"] = "<=0.05"
wspd_rel_df_long.loc[wspd_rel_df_long["Pvalue"] > 0.05, "Significance"] = ">0.05"
# Change the variable names for pretty plotting
wspd_rel_df_long["Variable"] = wspd_rel_df_long["Variable"].map(
    {
        "avg_wspd": "Wind speed (m/s)",
        "avg_fetch_lake": "Lake fetch (km)",
        "avg_fetch_buoy": "Buoy fetch (km)",
        "ref_area": "Lake area (km2)",
        "distance": "Buoy distance (km)",
    }
)


fig, ((ax2)) = plt.subplots(
    1,
    1,
    figsize=(13.5, 4),
)
markers = {"<=0.05": "s", ">0.05": "X"}
g1 = sns.scatterplot(
    ax=ax2,
    data=wspd_rel_df_long[wspd_rel_df_long["variable"] == "R"],
    x="value",
    y="Variable",
    hue="Metric",
    palette=["#3b528b", "#5ec962", "#440154"],
    style="Significance",
    markers=markers,
    s=100,
)
ax2.axhline(
    y="Wind speed (m/s)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1
)
ax2.axhline(y="Lake fetch (km)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1)
ax2.axhline(y="Lake area (km2)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1)
ax2.axhline(
    y="Buoy distance (km)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1
)
ax2.axhline(y="Buoy fetch (km)", linewidth=1, alpha=0.5, color="grey", ls="-", zorder=1)
ax2.set(xlabel="Correlation Coefficient (R)", ylabel="")
sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1), frameon=False)
# plt.show()
plt.savefig(os.path.join(home, "Data/Figures/wspd_buoy_var_reg.png"), dpi=1000)


