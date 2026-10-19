# ARCL Simulation — Eurobot 2027

Simulation Gazebo Harmonic / ROS 2 Jazzy de la table Eurobot 2027 et du robot holonome de l'ARCL.

## Lancer la simulation

```bash
./build.sh
source install/setup.bash
ros2 launch arcl_simulation sim.launch.py
```

## Commander le robot

**Aller à une position** (x, y en mètres) :

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: 1.5, y: 1.0}, orientation: {z: 0.0, w: 1.0}}}"
```

Ou avec l'outil **2D Goal Pose** de RViz.

**Commander la vitesse directement** :

```bash
ros2 topic pub /speed_command robot_msgs/msg/RobotSpeed "{vx: 0.1, vy: 0.0, omega: 0.0}"
```

**Se centrer sur une pierre** (tag ArUco visible par la caméra) :

```bash
ros2 action send_goal /rock_centering robot_msgs/action/RockCentering "{}" --feedback
```

**Attraper / lâcher avec les ventouses** :

```bash
ros2 topic pub --once /robot_omni/ventouse std_msgs/msg/Bool "{data: true}"   # attraper
ros2 topic pub --once /robot_omni/ventouse std_msgs/msg/Bool "{data: false}"  # lâcher
```
## Localisation

```
/joint_states ──> wheel_odometry ──> /wheel/odom ──┐
                                                  ├─> ekf_odom (robot_localization) ──> /odometry/filtered ──┐
pose vraie ─────> otos_simulator ──> /otos/odom ──┘                                                          ├─> beacon_localization ──> /pose_estimation
/scan (balises) ─────────────────────────────────────────────────────────────────────────────────────────────┘
```

- `wheel_odometry` : odométrie des 4 roues omni (vitesse + covariance, gonflée en cas de glissement détecté).
- `otos_simulator` : simule le SparkFun OTOS (erreur d'échelle, bruit, biais gyro, perte de suivi > 2,5 m/s). Paramètres réglables (`linear_scale_error`, `gyro_noise_density`, ...).
- `ekf_odom` : `ekf_node` de robot_localization, fusionne les vitesses roues + OTOS dans le repère `odom` ([config/ekf_odom.yaml](src/arcl_simulation/config/ekf_odom.yaml)).
- `beacon_localization` : EKF dans le repère `map`, prédiction avec `/odometry/filtered`, recalage sur les 3 balises vues au lidar.
