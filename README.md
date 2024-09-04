If you are viewing this in Azure DevOps, view on GitHub to see the images: [https://github.com/christiaantheunisse/Foresee-Error-Models](https://github.com/christiaantheunisse/Foresee-Error-Models)

# Foresee-Error-Models
This repo contains all the data and code to generate the error models discussed in *[Evaluating Set-based Occlusion Planners in Traffic Scenarios with Perception Uncertainties](paper.pdf)*. Besides, the data and code to generate the plots for the experiment section are also included. The abstract of the paper is:

> To ensure safely operation of autonomous vehicles (AVs), trajectory planners should account for occlusions. These are areas invisible to the AV that might contain vehicles. Set- based methods can guarantee safety by calculating the reachable set, which is the set of possible states, for each potentially hidden vehicle. A recently published method proved in simulation experiments to reduce the cautiousness by reasoning about these occluded areas over time, assuming perfect input data [[1]](https://ieeexplore.ieee.org/abstract/document/9827171/). We  present a novel algorithm that uses this reasoning and is applicable on a real AV with its accompanying uncertainties and imperfect sensor data. The uncertainties include sensor errors and noise, computation and communication delays and control errors in the trajectory following. This is achieved by modelling the error distributions and accounting for them in the calculations, where the confidence interval for each error is exposed as a setting. Experiments indicate that our algorithm can reduce the traversal time through an intersection by 2.2 seconds with reasoning. An ablation study of the different error measures shows that the errors in the construction of the field of view (FOV) limit the performance the most. Reducing the errors in the FOV construction is therefore our the most important recommendation, besides making the method interaction-aware.


[1] : Sánchez, J. M. G., Nyberg, T., Pek, C., Tumova, J., & Törngren, M. (2022, June). Foresee the unseen: Sequential reasoning about hidden obstacles for safe driving. In 2022 IEEE Intelligent Vehicles Symposium (IV) (pp. 255-264). IEEE. [link](https://ieeexplore.ieee.org/abstract/document/9827171/)

# Error models

## Gaussian Process

Gaussian Processes (GPs) are used for the error models because of the following advantages:
- Nonparametric, so no assumption about the data structure
- Simulateneous fitting of the error mean and variance
- Modelling of the epistemic uncertainty (model uncertainty), so basically overapproximating the error variance.
- Hyperparameters can be optimized with log-likelihood maximization (instead of the model itself like NNs).

Since online inference on Gaussian Processes is too time consuming, a lookup table is generated that is linearly interpolated.

## Lidar
A Lidar measurement consists of a list of angles and the range measured at each angle. Two models are generated: one for the distribution of the error in the ranges and one for the angles. Both models depend on the measured range and inclination angle. Furthermore, the maximum inclination angle is determined from these angles.

### Range error
> **Definition** (Range error). *The difference between the measured range and the true range.*

To visualize the data and fit a GP, run the code in `lidar/range_error/range_error.ipynb`.

<p float="left">
    <!-- <img src="lidar/range_error/range_error_data.png" style="max-height: 350px; height:auto; width:auto;">
    <img src="lidar/range_error/range_error_gp.png" style="max-height: 350px; height:auto; width:auto;"> -->
    <img src="lidar/range_error/range_error_data.png" width="54.94%">
    <img src="lidar/range_error/range_error_gp.png" width="44.06%">
</p>

### Angle error
> **Definition** (Angle error). *The difference between the reported angle for a range measurement and the true angle.*

To visualize the data and fit a GP, run the code in `lidar/angle_error/angle_error.ipynb`.
<p float="left">
    <!-- <img src="lidar/angle_error/angle_error_data.png" style="max-height: 350px; height:auto; width:auto;">
    <img src="lidar/angle_error/angle_error_gp.png" style="max-height: 350px; height:auto; width:auto;"> -->
    <img src="lidar/angle_error/angle_error_data.png" width="57.78%">
    <img src="lidar/angle_error/angle_error_gp.png" width="41.22%">
</p>

## Trajectory following

First, we need to find the relation between the parameters *velocity* $v$, *acceleration* $a$ and *curvature* $kappa$ and the three trajectory following errors: *longitudinal rate*, *lateral* and *orientation*. Two datasets are composed with data for the different parameters:

- `VelAcc`: Varying velocity and acceleration, but constant curvature ($\kappa = 0$)
- `VelCurv`: Varying velocity and curvature, but constant acceleration ($a = 0$)

In the below plot the data is plotted against the errors and Gaussian Processes are fitted to be able to properly assess the relations.

<img src="trajectory_following/dependencies/dependency_long_dt_errors.png" style="max-height: 350px; height:auto; width:auto;">
<img src="trajectory_following/dependencies/dependency_lat_errors.png" style="max-height: 350px; height:auto; width:auto;">
<img src="trajectory_following/dependencies/dependency_orient_errors.png" style="max-height: 350px; height:auto; width:auto;">

### Longitudinal error

The time derivative of the longitudinal error is modelled, i.e. the longitudinal error rate, because the trajectory follower tracks the velocity. It is therefore better to measure the velocity error. The `VelAcc` dataset is used for the error model.

### Lateral error

The acceleration data is unreliable for the lateral error, because the test trajectories always start with a positive acceleration and lateral error of zero, resulting in a low lateral error for positive accelerations. Consequently, the `VelCurv` dataset is for the GP fit.

### Orientation error

The orientation error model also depends on the `VelCurv` dataset, because the correlation with the curvature is much stronger than with the orientation.


## Velocity error

The linear and angular velocity are, respectively, measured by the wheel encoders and IMU. The error distributions are modelled with a constant variance and a velocity dependent mean. The mean is used to remove the bias from (calibrate) the measurements and the variance is sent with to measurements to be used downstream in the extended Kalman filters.

### Wheel encoders

The true velocities are obtained with a tachometer, resulting in the following data:

<img src="velocity_error/wheel_encoders_error/wheel_encoders_fit.png" style="max-height: 350px; height:auto; width:auto;">

### IMU

The true average velocity is calculated from the total orientation reported by the SLAM module over a period of 30 seconds. These are compared to the average of the velocities measured by the IMU. The variance is estimated with the Mean Square Successive Difference (MSSD).

<img src="velocity_error/imu_error/imu_error_fit.png" style="max-height: 350px; height:auto; width:auto;">

## Pose estimation error

### EKF filter tuning
The noise covariance matrix Q of the EKF filter has to be properly tuned to ensure that the estimated uncertainty of the estimated pose is correct. The SLAM pose estimation was used as the true pose the calculate the error in the EKF pose. The plot below shows the cumulative proportion of the errors that was found for each confidence interval. More specifically, when the estimation of the uncertainty is conservative, at least x% of the errors should belong to the x%-confidence interval or better. This is everything above the black dashed lines in the plot.

In the [paper](#) it is argued that the fact that the SLAM pose is imperfect, makes the EKF **position** estimate appear worse. This is especially true when the uncertainty of the SLAM pose is relatively big compared to the EKF uncertainty, which is the case for small SLAM update intervals. This point supported by the plots for the position error.

<img src="pose_estimation_error/pose_estimation_proportions_plot.png" style="max-width: 800px; height:auto; width:auto;">

# Experiments

More information about the robot and software used in the experiments can be found in the [paper](paper.pdf) and on [this GitHub with the mobile robot software](https://github.com/christiaantheunisse/Foresee-the-Unseen-ROS).


## Performance experiments: normal traffic scenarios

The traversal times for the four compared algorithms are. T-tests indicate that the traversal time improvement is significant for all scenarios for `foresee vs. baseline` and for `foresee++ vs. baseline++` for the scenarios 2, 3 and 6. A two-way ANOVA showed that the error models significantly increased the traversal time for both `baseline vs. baseline++` and `foresee and foresee++`.

<img src="experiments/performance_normal_traffic_scenarios/perf_normal_traversal_times.png" style="max-height: 350px; height:auto; width:auto;">

## Ablation study: normal traffic scenarios

The traversal times for the normal traffic scenarios in the ablation study. The Tukey's HSD test indicates that `foresee abl. FOV` and `Lidar` are significantly less cautious than `foresee abl. delay` and `traj`. This leads to the conclusion that the errors in the field of view (FOV) and Lidar contribute the most to the increased cautiousness. 

<p float="left">
    <!-- <img src="experiments/ablation_normal_traffic_scenarios/abl_normal_traversal_times.png" style="max-height: 350px; height:auto; width:auto;">
    <img src="experiments/ablation_normal_traffic_scenarios/ablated_algorithms_tukey_hsd.png" style="max-height: 350px; height:auto; width:auto;"> -->
    <img src="experiments/ablation_normal_traffic_scenarios/abl_normal_traversal_times.png" width="42.09%">
    <img src="experiments/ablation_normal_traffic_scenarios/ablated_algorithms_tukey_hsd.png" width="56.91%">
</p>

## Horizon lengths

This experiments shows that a longer time horizon results in a bigger traversal time, because the method is not interaction-aware. For a horizon of 30, the mobile robot cannot even cross the intersection because it collides with possible hidden obstacles at the boundary of the maximum field of view.

<img src="experiments/horizon_length/horizons_vs_traversal_times.png" style="max-height: 350px; height:auto; width:auto;">
