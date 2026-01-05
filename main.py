
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def remove_computer_vision_transports(data):
    """
    This function removes moments where the computer vision algorithm temporarily picks up another object instead of the ball
    
    :param data: x, y data from comp vis alg.
    """
    vis = data[data.visible].copy()
    vis["dt"] = vis.index.to_series().diff()
    vis["dist"] = np.hypot(
        vis["x"].diff(),
        vis["y"].diff()
    )
    JUMP_THRESH = 300

    jump_start_idx = vis[
        (vis["dt"] <= 2) &   # 1-frame gap OR 1-frame gap after nulls
        (vis["dist"] > JUMP_THRESH)
    ].index

    if len(jump_start_idx) == 0:
        cleaned = data
    else:
        clean_mask = pd.Series(True, index=data.index)

        for jump_frame in jump_start_idx:

            # Last good position BEFORE jump
            prev = vis.loc[:jump_frame].iloc[-2]
            x0, y0 = prev.x, prev.y

            # Frames AFTER jump
            after = vis.loc[jump_frame:]

            # Find when it comes back near original trajectory
            return_frame = None
            for f, r in after.iterrows():
                if np.hypot(r.x - x0, r.y - y0) < JUMP_THRESH:
                    return_frame = f
                    break

            if return_frame is None:
                # Remove everything after jump
                clean_mask.loc[jump_frame:] = False
            else:
                # Remove ghost segment
                clean_mask.loc[jump_frame:return_frame] = False

        data.loc[~clean_mask, ["x","y","visible"]] = [np.nan, np.nan, False]
        
    # Remove jumps in large gaps
    MAX_JUMP = 300  
    MAX_GAP = 10   

    last_x, last_y, last_f = None, None, None
    bad_rows = []

    for f, row in data.iterrows():
        if not row["visible"]:
            continue

        if last_x is not None:
            dt = f - last_f
            dist = np.hypot(row["x"] - last_x, row["y"] - last_y)

            if dt > 1 and (dt <= MAX_GAP) and dist > MAX_JUMP:
                bad_rows.append(f)
                continue

        last_x, last_y, last_f = row["x"], row["y"], f

    data.loc[bad_rows, ["x", "y", "visible"]] = [np.nan, np.nan, False]
    return data
    
def unsupervised_hit_bounce_detection(ball_data_i):
    raw_data = pd.DataFrame.from_dict(ball_data_i, orient="index").reset_index(names="frames")
    raw_data["frames"] = raw_data["frames"].astype(int)
    # raw_data = pd.read_json(f"data/per_point_v2/{ball_data_i}").T.reset_index(names="frames")
    
    data = raw_data.copy()

    data["x"] = pd.to_numeric(data["x"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    x0 = data["x"].dropna().iloc[0]
    y0 = data["y"].dropna().iloc[0]

    mask_change = (
        data["x"].notna() &
        data["y"].notna() &
        ((data["x"] != x0) | (data["y"] != y0))
    )

    first_move_idx = mask_change.idxmax()
    data = data.loc[first_move_idx - 1:] 
    
    data = remove_computer_vision_transports(data)

    data[["x", "y"]] = data[["x", "y"]].interpolate(limit_direction="both")
    data = data.sort_values("frames")

    # Create a smoother x,y for derivatives
    data["x_s"] = savgol_filter(data["x"], window_length=9, polyorder=2)
    data["y_s"] = savgol_filter(data["y"], window_length=9, polyorder=2)
    data["dx"] = data["x_s"].shift(-1) - data["x_s"]
    data["dy"] = data["y_s"].shift(-1) - data["y_s"]

    data[["dx", "dy"]] = data[["dx", "dy"]].bfill()

    data["speed"] = (data["dx"] ** 2 + data["dy"] ** 2) ** 0.5

    data["ax"] = data["dx"].shift(-1) - data["dx"]
    data["ay"] = data["dy"].shift(-1) - data["dy"]

    data[["ax", "ay"]] = data[["ax", "ay"]].bfill()

    data["acceleration"] = (data["ax"] ** 2 + data["ay"] ** 2) ** 0.5
    
    cols = ["dx", "dy", "speed", "ax", "ay", "acceleration"]

    for c in cols:
        data[f"{c}_raw"] = data[c]

    data[cols] = data[cols].rolling(5, center=True, min_periods=1).mean()
    
    def dy_over_next_30_frames(row):
        frame = row["frames"]
        y = row["y"]
        next_y = data[(data.frames>frame) & (data.frames <= frame+30)]["y"]
        
        if len(next_y) > 0:
            return next_y.iloc[-1] - y
        else:
            return 0
    data["dy_over_next_30"] = data.apply(dy_over_next_30_frames,axis=1)
    
    def average_dy_over_next_10(row):
        frame = row["frames"]
        window = data[(data.frames >= frame) & (data.frames <= frame + 10)]
        return window.dy.mean()
    data["average_dy_over_next_10"] = data.apply(average_dy_over_next_10,axis=1)
    
    def local_extrema_in_dy_over_next_30(row, df):
        frame = row["frames"]
        dy_over_next_30_frames = row["dy_over_next_30"]
        
        window = df[(df.frames >= frame - 15) & (df.frames <= frame + 15)]
        
        if dy_over_next_30_frames == window.dy_over_next_30.max():
            return "max"
        elif dy_over_next_30_frames == window.dy_over_next_30.min():
            return "min"
    
    upper_hits = data[(data["ay"] <= -.5) & (data.dy_over_next_30 <= -150) & (data.average_dy_over_next_10 < 0) & (data.y > 500)] # good for top hits that make it over the net 
    upper_hits["extrema_in_dy_over_30"] = upper_hits.apply(local_extrema_in_dy_over_next_30,axis=1,args=(upper_hits,))
    upper_hits = upper_hits[upper_hits.extrema_in_dy_over_30 == "min"]
    upper_hits["hit_type"] = "upper"

    lower_hits = data[(data["ay"] >= .45) & (data.dy_over_next_30 >= 150) & (data.average_dy_over_next_10 > 0) & (data.y < 500)] # good for bottom hits that make it over the net
    lower_hits["extrema_in_dy_over_30"] = lower_hits.apply(local_extrema_in_dy_over_next_30,axis=1,args=(lower_hits,))
    lower_hits = lower_hits[lower_hits.extrema_in_dy_over_30 == "max"]
    #Correct lower hits
    corrected_lower_hits = []
    for i, lower_hit in lower_hits.iterrows():
        frame = lower_hit["frames"]
        before = data[(data.frames >= frame - 50) & (data.frames <= frame)]
        idx = before["y"].idxmin()
        min_y_before = data.loc[idx]
        corrected_lower_hits.append(min_y_before)
    if len(corrected_lower_hits) > 0:
        lower_hits = pd.concat(corrected_lower_hits,axis=1).T
        lower_hits["hit_type"] = "lower"
    
    pred_hits = pd.concat([lower_hits,upper_hits]).sort_values("frames")
    
    #Find the serve if it hasn't already happend
    if len(pred_hits) > 0:
        if pred_hits.iloc[0].x == data.iloc[0].x and pred_hits.iloc[0].y == data.iloc[0].y:
            max_ay_before_first_hit = pd.Series()
        else:
            first_hit = pred_hits.iloc[0]
            before_first_hit = data[(data.frames >= first_hit.frames - 50) & (data.frames <= first_hit.frames - 5) & (data.dy >= -2) & (data.dy <= 2)]
            if before_first_hit.shape[0] == 0:
                max_ay_before_first_hit = pd.Series()
            else:
                max_ay_before_first_hit = before_first_hit.loc[before_first_hit.ay.idxmin()]
                max_ay_before_first_hit["hit_type"] = "serve"    
                if np.abs(max_ay_before_first_hit.y - first_hit.y) <= 150:
                    max_ay_before_first_hit = pd.Series()
    else:
        max_ay_before_first_hit = pd.Series()
            
    if len(max_ay_before_first_hit)>0:
        pred_hits = pd.concat([max_ay_before_first_hit.to_frame().T,pred_hits])  
        
    pred_hits["frame_gap"] = pred_hits.frames.diff()
    pred_hits["group_id"] = (pred_hits["frame_gap"] > 300).cumsum()
    
    bounces_between_hits = []
    for gid, pred_hit_group in pred_hits.groupby("group_id"):
        for i  in range(pred_hit_group.shape[0] - 1):
            first_hit = pred_hit_group.iloc[i]
            second_hit = pred_hit_group.iloc[i+1]
            
            first_frame = first_hit.frames
            second_frame = second_hit.frames
            
            if second_frame - 5 - (first_frame + 5) < 3:
                continue
            between = data[(data.frames > first_frame + 5) & (data.frames < second_frame - 5)].copy()
            idx = between["ay"].idxmin()
            min_ay_between = between.loc[idx]
            bounces_between_hits.append(min_ay_between)
        
    raw_data = raw_data.set_index("frames")
    if len(bounces_between_hits) > 0:
        pred_bounces = pd.concat(bounces_between_hits,axis=1).T
        raw_data.loc[pred_bounces.frames,"pred_action"] = "bounce"  
                

    raw_data.loc[pred_hits.frames, "pred_action"] = "hit"

    raw_data["pred_action"] = raw_data["pred_action"].fillna("air")

    raw_data = raw_data.reset_index()
    raw_data["frames"] = raw_data["frames"].astype(str)

    output = {
        row.frames: {
            "x": None if pd.isna(row.x) else int(row.x),
            "y": None if pd.isna(row.y) else int(row.y),
            "visible": bool(row.visible),
            "action": row.action,
            "pred_action": row.pred_action,
        }
        for _, row in raw_data.iterrows()
    }

    return output


def NMS(potential_values, extrema):
    """
    Applies Non-Maximal Suppresion to potential extrema values to find the true extrema
    
    :param potential_values: Potential Values for extrema
    :param extrema: Whether we are finding local max or min
    """
    local_extrema = []
    for i, row in potential_values.iterrows():
        frame = row["frames"]
        y = row["y"]
        candidates = potential_values[(potential_values.frames >= frame - 10) & (potential_values.frames <= frame + 10)]
        max_y_candidates = candidates["y"].max()
        min_y_candidates = candidates["y"].min()
        if extrema == "max":
            if y < max_y_candidates:
                pass
            else:
                local_extrema.append(row)
        else:
            if y > min_y_candidates:
                pass
            else:
                local_extrema.append(row)
    local_extrema = pd.concat(local_extrema,axis=1).T
    return local_extrema

def find_extrema(data):
    # Find potential local minimum and maximum from data using rolling average of dx, dy of ball
    potential_maxes = []
    potential_mins = []

    for i, row in data.iterrows():
        frame = row["frames"]
        dy = row["dy"]
        ay = row["ay"]
        
        before = data[(data.frames >= frame - 3) & (data.frames < frame)]
        after = data[(data.frames > frame) & (data.frames <= frame + 3)]
        around = data[(data.frames >= frame-5) & (data.frames <= frame + 5)]
        
        avg_before_dy = before.dy.mean()
        avg_after_dy = after.dy.mean()
        
        around_ay = around.ay.mean()
        
        if (avg_after_dy < 0 or pd.isna(avg_after_dy)) and (avg_before_dy > 0 or pd.isna(avg_before_dy)) and around_ay < 0: # Local maximum Criteria
            potential_maxes.append(row)
        elif (avg_after_dy > 0 or pd.isna(avg_after_dy)) and (avg_before_dy < 0 or pd.isna(avg_before_dy)) and around_ay > 0: # Local minimum Criteria
            potential_mins.append(row)
            
    potential_maxes = pd.concat(potential_maxes,axis=1).T
    potential_mins = pd.concat(potential_mins,axis=1).T

    # Apply Non-Maximal Suppression to condense potential minimum into actual local minimum
    local_maxes = NMS(potential_maxes,extrema="max")
    local_mins = NMS(potential_mins,extrema="min")
    
    local_mins = local_mins.groupby("y").first().reset_index()
    local_maxes = local_maxes.groupby("y").first().reset_index()
    local_mins["extrema"] = "min"
    local_maxes["extrema"] = "max"

    # Add all the extrema into a df to identify hits and bounces
    extrema = pd.concat([local_mins,local_maxes]).sort_values("frames")
    extrema["dy_from_next_extrema"]  = extrema["y"].diff(-1) * -1
    extrema["dy_from_sim_extrema"] = extrema.groupby("extrema")["y"].diff(-1) * -1
    extrema = extrema.reset_index().drop("index",axis=1)
    
    return extrema


def create_model_input_from_json(ball_data_i):
    raw_data = pd.DataFrame.from_dict(ball_data_i, orient="index").reset_index(names="frames")
    raw_data["frames"] = raw_data["frames"].astype(int)
    # raw_data = pd.read_json(f"data/per_point_v2/{ball_data_i}").T.reset_index(names="frames")
        
    data = raw_data.copy()

    data["x"] = pd.to_numeric(data["x"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")

    data = remove_computer_vision_transports(data)

    data[["x", "y"]] = data[["x", "y"]].interpolate(limit_direction="both")
    data = data.sort_values("frames")

    # Create a smoother x,y for derivatives
    data["x_s"] = savgol_filter(data["x"], window_length=9, polyorder=2)
    data["y_s"] = savgol_filter(data["y"], window_length=9, polyorder=2)
    data.loc[:,"dx"] = data["x_s"].diff(1)
    data.loc[:,"dy"] = data["y_s"].diff(1)
    data[["dx", "dy"]] = data[["dx", "dy"]].bfill()
    data.loc[:,"speed"] = (data["dx"] ** 2 + data["dy"] **2) ** 1/2

    data.loc[:,"ax"] = data["dx"].diff(1)
    data.loc[:,"ay"] = data["dy"].diff(1)
    data[["ax", "ay"]] = data[["ax", "ay"]].bfill()
    data.loc[:,"acceleration"] = (data["ax"] ** 2 + data["ay"] **2) ** 1/2

    extrema = find_extrema(data)

    extrema = extrema[["frames","extrema"]].set_index("frames")
    data = data.set_index("frames")

    data.loc[extrema.index, "extrema"] = extrema["extrema"]
        
    # Creating input to model for prediction
    extrema_map = {"max": 1, "min": -1}
    
    
    data["extrema"] = data["extrema"].apply(lambda e: extrema_map.get(e,0))
    data["visible"] = data["visible"].apply(bool)
    
    X = data.drop(["action"],axis=1)

    cols = ["dx", "dy", "speed", "ax", "ay", "acceleration"]

    for c in cols:
        X[f"{c}_raw"] = X[c]

    X[cols] = X[cols].rolling(5, center=True, min_periods=1).mean()

    lag_features = ['dx', 'dy', 'speed', 'ax', 'ay', 'acceleration']
    for feat in lag_features:
        for lag in range(-3, 4):  # 3 frames on each side
            if lag == 0:
                continue
            X[f"{feat}_lag{lag}"] = X[feat].shift(lag)
    
    return X, raw_data

def supervized_hit_bounce_detection(ball_data_i):
    X_input, raw_data = create_model_input_from_json(ball_data_i)

    model = XGBClassifier()
    model.load_model("hit_bounce_detection.json")
    pred_action = model.predict(X_input)

    raw_data["pred_action"] = pred_action
    action_map = {0: "air", 1: "bounce", 2: "hit"}

    raw_data["pred_action"] = raw_data["pred_action"].map(action_map)
    output = {
            row.frames: {
                "x": None if pd.isna(row.x) else int(row.x),
                "y": None if pd.isna(row.y) else int(row.y),
                "visible": bool(row.visible),
                "action": row.action,
                "pred_action": row.pred_action,
            }
            for _, row in raw_data.iterrows()
        }
    return output