# Foresee-Error-Models
This repo contains all the data and code to generate the error models for the [paper](). 

## Gaussian Process

Gaussian Processes (GPs) are used for the error models because of the following advantages:
- Nonparametric, so no assumption about the data structure
- Simulateneous fitting of the error mean and variance
- Modelling of the epistemic uncertainty (model uncertainty), so basically overapproximating the error variance.
- Hyperparameters can be optimized with log-likelihood maximization (instead of the model itself like NNs).

Since online inference on Gaussian Processes is too costly, the models is sampled over a grid of 1000 steps for each parameter dimension. The values on this grid are linearly interpolated online.

## Lidar
A Lidar measurement consists of a list of angles and the range measured at each angle. Two models are generated: one for the distribution of the error in the ranges and one for the angles. Both models depend on the measured range and inclination angle. Furthermore, the maximum inclination angle is determined from these angles.

### Range error
> **Definition** (Range error). *The difference between the measured range and the true range.*

To visualize the data and fit a GP, run the code in `lidar/range_error/range_error.ipynb`.

<img src="lidar/range_error/range_error_data.png" width="400">

<img src="lidar/range_error/range_error_gp.png" width="400">

### Angle error
> **Definition** (Angle error). *The difference between the reported angle for a range measurement and the true angle.*

To visualize the data and fit a GP, run the code in `lidar/angle_error/angle_error.ipynb`.

<img src="lidar/angle_error/angle_error_data.png" width="400">

<img src="lidar/angle_error/angle_error_gp.png" width="400">

## Trajectory following

### Longitudinal error

### Lateral error

### Orientation error



## Velocity error

The linear and angular velocity are, respectively, measured by the wheel encoders and IMU. The error distributions are modelled with a constant variance and a velocity dependent mean. The mean is used to remove the bias from (calibrate) the measurements and the variance is sent with to measurements to be used downstream in the extended Kalman filters.

### Wheel encoders

The true velocities are obtained with a tachometer, resulting in the following data:

<img src="velocity_error/wheel_encoders_error/wheel_encoders_fit.png" width="400">

### IMU

The true average velocity is calculated from the total orientation reported by the SLAM module over a period of 30 seconds. These are compared to the average of the velocities measured by the IMU. The variance is estimated with the Mean Square Successive Difference (MSSD).

<img src="velocity_error/imu_error/imu_error_fit.png" width="400">

## Pose estimation error

### EKF filter tuning
Q-matrix