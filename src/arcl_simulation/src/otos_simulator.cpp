#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include <Eigen/Dense>
#include <random>
#include <cmath>

// Simulateur du SparkFun Optical Tracking Odometry Sensor (OTOS : capteur optique PAA5160E1 + IMU LSM6DSO).
// Le vrai capteur intègre lui-même une pose (x, y, cap) à partir du flux optique (translation) et du gyro (rotation).
// On le reproduit à partir de la pose vraie de Gazebo (/robot_omni/pose_vraie) en ajoutant :
//   - une erreur de facteur d'échelle linéaire et angulaire (datasheet : 3-5 % d'erreur typique avant calibration,
//     le vrai capteur se calibre avec setLinearScalar / setAngularScalar),
//   - un bruit blanc sur les déplacements (marche aléatoire en position),
//   - un bruit et un biais dérivant sur le gyro,
//   - une perte de suivi au-delà de 2,5 m/s.
// On considère le capteur monté au centre du robot (ou l'offset déjà compensé avec setOffset).
// Sortie : /otos/odom (nav_msgs/Odometry, pose dans le repère odom du capteur, twist dans base_link).

double wrap(double a)
{
  a = std::fmod(a + M_PI, 2.0 * M_PI);
  if (a < 0.0) {a += 2.0 * M_PI;}
  return a - M_PI;
}

double toSec(const builtin_interfaces::msg::Time & t)
{
  return static_cast<double>(t.sec) + 1e-9 * static_cast<double>(t.nanosec);
}

double yawFromQuaternion(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

class OtosSimulator : public rclcpp::Node{

        public:
                OtosSimulator() : Node("otos_simulator"){

                        linear_scale_error = declare_parameter("linear_scale_error", 0.02);
                        angular_scale_error = declare_parameter("angular_scale_error", 0.005);
                        linear_noise_density = declare_parameter("linear_noise_density", 0.002);
                        gyro_noise_density = declare_parameter("gyro_noise_density", 0.0005);
                        gyro_bias_random_walk = declare_parameter("gyro_bias_random_walk", 2e-5);
                        max_speed = declare_parameter("max_speed", 2.5);
                        reported_linear_error = declare_parameter("reported_linear_error", 0.03);
                        reported_angular_error = declare_parameter("reported_angular_error", 0.01);
                        velocity_noise_floor = declare_parameter("velocity_noise_floor", 0.002);
                        angular_velocity_noise_floor = declare_parameter("angular_velocity_noise_floor", 0.002);
                        int seed = declare_parameter("seed", 42);

                        generateur.seed(seed);

                        pose_vraie_sub_ = create_subscription<nav_msgs::msg::Odometry>(
                            "/robot_omni/pose_vraie", 10,
                            std::bind(&OtosSimulator::onPoseVraie, this, std::placeholders::_1)
                        );

                        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/otos/odom", 10);

                        pose = Eigen::Vector3d::Zero();
                        pose_covariance = Eigen::Matrix3d::Zero();

                        RCLCPP_INFO(this->get_logger(), "OtosSimulator launched successfuly");
                }


        private:

                double linear_scale_error;
                double angular_scale_error;
                double linear_noise_density;     // m / sqrt(m parcouru)
                double gyro_noise_density;       // rad / sqrt(s)
                double gyro_bias_random_walk;    // rad/s / sqrt(s)
                double max_speed;                // m/s
                double reported_linear_error;    // erreur relative annoncée dans la covariance
                double reported_angular_error;
                double velocity_noise_floor;
                double angular_velocity_noise_floor;

                std::mt19937 generateur;
                std::normal_distribution<double> gaussienne{0.0, 1.0};

                bool initialization = true;
                double t_precedent = 0;
                Eigen::Vector3d pose_vraie_precedente;

                double gyro_bias = 0;

                // pose intégrée par le capteur, dans son propre repère (origine = pose au démarrage)
                Eigen::Vector3d pose;
                Eigen::Matrix3d pose_covariance;


                rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_vraie_sub_;
                void onPoseVraie(const nav_msgs::msg::Odometry::SharedPtr msg){
                    double t = toSec(msg->header.stamp);
                    Eigen::Vector3d pose_vraie(msg->pose.pose.position.x, msg->pose.pose.position.y,
                                               yawFromQuaternion(msg->pose.pose.orientation));

                    if (initialization){
                        initialization = false;
                        t_precedent = t;
                        pose_vraie_precedente = pose_vraie;
                        return;
                    }

                    double dt = t - t_precedent;
                    if (dt <= 0) return;

                    // déplacement vrai exprimé dans le repère robot (cap moyen sur le pas)
                    double dtheta = wrap(pose_vraie[2] - pose_vraie_precedente[2]);
                    double theta_m = pose_vraie_precedente[2] + dtheta / 2;
                    double dx_monde = pose_vraie[0] - pose_vraie_precedente[0];
                    double dy_monde = pose_vraie[1] - pose_vraie_precedente[1];
                    double dx =  cos(theta_m)*dx_monde + sin(theta_m)*dy_monde;
                    double dy = -sin(theta_m)*dx_monde + cos(theta_m)*dy_monde;
                    double d = std::hypot(dx, dy);

                    t_precedent = t;
                    pose_vraie_precedente = pose_vraie;

                    // capteur optique
                    double dx_mesure = 0, dy_mesure = 0;
                    if (d / dt > max_speed){
                        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "OTOS : vitesse > %.1f m/s, perte de suivi", max_speed);
                    } else {
                        double sigma_lineaire = linear_noise_density * std::sqrt(d);
                        dx_mesure = (1 + linear_scale_error) * dx + sigma_lineaire * gaussienne(generateur);
                        dy_mesure = (1 + linear_scale_error) * dy + sigma_lineaire * gaussienne(generateur);
                    }

                    // gyro
                    gyro_bias += gyro_bias_random_walk * std::sqrt(dt) * gaussienne(generateur);
                    double dtheta_mesure = (1 + angular_scale_error) * dtheta + gyro_bias * dt
                                         + gyro_noise_density * std::sqrt(dt) * gaussienne(generateur);

                    // intégration comme le fait le capteur
                    double theta_m_capteur = pose[2] + dtheta_mesure / 2;
                    pose[0] += cos(theta_m_capteur)*dx_mesure - sin(theta_m_capteur)*dy_mesure;
                    pose[1] += sin(theta_m_capteur)*dx_mesure + cos(theta_m_capteur)*dy_mesure;
                    pose[2] = wrap(pose[2] + dtheta_mesure);

                    double vx = dx_mesure / dt;
                    double vy = dy_mesure / dt;
                    double omega = dtheta_mesure / dt;

                    // covariance annoncée : erreur relative + plancher de bruit
                    double v = std::hypot(vx, vy);
                    double var_v = std::pow(reported_linear_error * v, 2) + std::pow(velocity_noise_floor, 2);
                    double var_omega = std::pow(reported_angular_error * omega, 2) + std::pow(angular_velocity_noise_floor, 2);

                    pose_covariance(0, 0) += var_v * dt * dt;
                    pose_covariance(1, 1) += var_v * dt * dt;
                    pose_covariance(2, 2) += var_omega * dt * dt;

                    publish_odom(msg->header.stamp, vx, vy, omega, var_v, var_omega);
                }


                rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
                void publish_odom(const builtin_interfaces::msg::Time & stamp, double vx, double vy, double omega,
                                  double var_v, double var_omega){
                    nav_msgs::msg::Odometry msg;
                    msg.header.stamp = stamp;
                    msg.header.frame_id = "odom";
                    msg.child_frame_id = "base_link";

                    msg.pose.pose.position.x = pose[0];
                    msg.pose.pose.position.y = pose[1];
                    msg.pose.pose.orientation.z = std::sin(0.5 * pose[2]);
                    msg.pose.pose.orientation.w = std::cos(0.5 * pose[2]);

                    msg.twist.twist.linear.x = vx;
                    msg.twist.twist.linear.y = vy;
                    msg.twist.twist.angular.z = omega;

                    // covariances ROS : 6x6 (x, y, z, roll, pitch, yaw)
                    msg.pose.covariance[0] = pose_covariance(0, 0);
                    msg.pose.covariance[7] = pose_covariance(1, 1);
                    msg.pose.covariance[35] = pose_covariance(2, 2);
                    msg.twist.covariance[0] = var_v;
                    msg.twist.covariance[7] = var_v;
                    msg.twist.covariance[35] = var_omega;
                    for (int k : {2, 3, 4}){
                        msg.pose.covariance[6*k + k] = 1e3;
                        msg.twist.covariance[6*k + k] = 1e3;
                    }

                    odom_pub_->publish(msg);
                }

};


int main(int argc, char * argv[]){
        rclcpp::init(argc, argv);
        rclcpp::spin(std::make_shared<OtosSimulator>());
        rclcpp::shutdown();
        return 0;
}
