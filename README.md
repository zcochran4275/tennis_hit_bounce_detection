# Tennis Hit and Bounce Detection

This repo contains code that detects hits and bounces from tennis ball data collected from opencv computer vision algrotithms. In the `data/` folder you can find over 300 points from the Final of Roland Garros 2025. The data includes the actual frames that mark the real hits and bounces that occur during the point. You can view an example of such a point below.

![](front_overlay_111.mp4)

## Format of the Ball-Tracking Data

Each JSON file contains entries indexed by the **frame number** (e.g., `56100`).

```json
{
  "56100": {
    "x": 894,
    "y": 395,
    "visible": true,
    "action": "air"
  }
}
```

### Field Descriptions

- **Frame Number (key)**: Frame index in the video (e.g., `56100`)
- **x**: Horizontal pixel position on a 1920-pixel-wide frame
- **y**: Vertical pixel position on a 1080-pixel-high frame
- **visible**: `true` if the ball is detected in this frame
- **action**: Ground-truth label indicating ball state  
  - `"air"`
  - `"hit"`
  - `"bounce"`


## Repo Description

In the `main.py` file you can find 2 algorithms to detect hits and bounces from the ball tracking data. 

1. The first is an unsupervised approach to detecting hits and bounces and doesn't utilize in each data point. This algortims focuses on the change in position, speed, acceleration of the tennis ball throughout a point to detect the hits and bounces and is described in more detail below.

2. The second is a supervised approach to detecting hits and bounces. This approach utilizes the labels to train a ML model (XGBoost) that can predict the frames that hits and bounces occur.

In the `analyze.py` file you can find code that visualizes the predicted vs. actual hits and bounces for a point. As well as a function for creating a recall matrix to show the number of correct and incorrect hit and bounce predictions, a hit or bounce is determined as a correct prediction if it is within 5 frames of the actual hit/bounce.

In the `eval.ipynb` you can find a short script to compute the recall matrix for all points from the final and gives a place to visualize some of the predictions.

## Algorithm Description

### Unsupervised Approach

Given the raw tracking data I applied the following steps

1. Removed moments where the computer vision algorithm temporarily locked on to a different object and the ball appeared to "teleport" for a few frames.
2. Used linear interpolation to fill missing frames. 
3. Calculated dx, dy, speed, ax, ay, acceleration for each frame.
4. Added additional columns that contain a 3 point rolling mean for each of the features above so that the algorithm isn't susceptible to random sharp changes.
5. Detect hits by the following criteria:

    5a. Upper-half hits

    Upper Half Hits: where the acceleration in y is less than -.5 and chanage in y over the next 30 frames is less than -150. I then take these potential upper hits and apply non-maximal suppression to find one hit in each "cluster" where the change in y over next 30 frames is minimized.

    5b. Lower-half hits

    where the acceleration in y is greater than .5 and chanage in y over the next 30 frames is greater than 150. I then take these potential lower hits and apply non-maximal suppression to find one hit in each "cluster" where the change in y over next 30 frames is maximized. Since lower hit points tended to always be a little further up on the curve I realligned the hit to be the minimum in y before the predicted hit (This aligned it closer to the actual hits). 

6. Since the hit detection above doesn't accurately detect serves because as the server throws the ball up the hit no longer has the max/min of change in y over next 30 frames. Therefore we detect possible serves where the server throws the ball and prepend it to our predicted hits.

7. Then we find the bounce that occurs between each hit. This is noted as the frame that has the min acceleration of y. This detects the moment where the y value quickly changes as the ball bounces and then goes back to heading in the same direction.

8. If the last hit is not a net hit we find the last bounce that occurs after the last predicted hit. 

#### Unsupervised Results

Using per-event matching with a ±5 frame tolerance, the model achieved the following performance:

- **Hits:**  
  $$
  \frac{1275}{1600} \approx 79.7\%
  $$

- **Bounces:**  
  $$
  \frac{1136}{1446} \approx 78.6\%
  $$

### Supervised Approach

Given the raw tracking data we performed the following steps to train the model.

1. Removed moments where the computer vision algorithm temporarily locked on to a different object and the ball appeared to "teleport" for a few frames.
2. Used linear interpolation to fill missing frames. 
3. Calculated dx, dy, speed, ax, ay, acceleration for each frame.
4. Added additional columns that contain a 3 point rolling mean for each of the features above so that the algorithm isn't susceptible to random sharp changes.
5. Computed if each frame was a local max/min from second derivative tests and non-maximal suppression.
6. Added the 3 previous frames features as additional temporal context for each frame.
7. Used the features and actual labels to train an XGBoost model to predict which frames are hits / bounces.

#### Supervised Results 

Using per-event matching with a ±5 frame tolerance, the model achieved the following performance:

- **Hits:**  
  $$
  \frac{1376}{1376 + 224} = \frac{1376}{1600} \approx 86\%
  $$

- **Bounces:**  
  $$
  \frac{1258}{1258 + 188} = \frac{1258}{1446} \approx 87\%
  $$