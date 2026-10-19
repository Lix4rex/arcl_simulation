#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

// Odométrie des roues : /joint_states -> /wheel/odom (nav_msgs/Odometry)
// La vitesse (twist, repère base_link) et sa covariance sont fusionnées par l'ekf_node de robot_localization.
// La pose intégrée (repère odom) n'est publiée que pour information.

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

class WheelOdometry : public rclcpp::Node{

        public:
                WheelOdometry() : Node("wheel_odometry"){

                        publish_rate = declare_parameter("publish_rate", 100.0);

                        joint_states_sub_ = create_subscription<sensor_msgs::msg::JointState>(
                            "/joint_states", 10,
                            std::bind(&WheelOdometry::onJointStates, this, std::placeholders::_1)
                        );

                        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/wheel/odom", 10);

                        float a = sqrt(2)/4;
                        float b = 1 / (4*wheel_distance_from_center);
                        pseudo_inverse_matrice_roues <<
                            -a, -a, a, a,
                            a, -a, -a, a,
                            b, b, b, b;

                        pose = Eigen::Vector3d::Zero();
                        pose_covariance = Eigen::Matrix3d::Zero();
                        deplacement_cumule = Eigen::Vector3d::Zero();
                        covariance_cumulee = Eigen::Matrix3d::Zero();

                        RCLCPP_INFO(this->get_logger(), "WheelOdometry launched successfuly");
                }


        private:

                float wheel_radius = 0.03;
                float wheel_distance_from_center = 0.162635;

                float nb_ticks = 4096;

                float e_max = 2e-4;
                float lambda_glissement = 30;
                float q = 2*M_PI*wheel_radius/nb_ticks;
                float sigma_q = q*1/sqrt(12);

                float coefficient_glissement = 0.01;

                double publish_rate;

                std::vector<std::string> wheel_names = {"roue_1_joint", "roue_2_joint", "roue_3_joint", "roue_4_joint"};

                Eigen::Matrix<double, 4, 1> joint_state_positions;
                Eigen::Matrix<double, 3, 4> pseudo_inverse_matrice_roues;
                bool initialization = true;

                // pose intégrée dans le repère odom
                Eigen::Vector3d pose;
                Eigen::Matrix3d pose_covariance;

                // déplacement accumulé (repère robot) depuis la dernière publication
                Eigen::Vector3d deplacement_cumule;
                Eigen::Matrix3d covariance_cumulee;
                double t_derniere_publication = 0;


                rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
                void onJointStates(const sensor_msgs::msg::JointState::SharedPtr joint_state_msg){
                    auto positions = joint_state_msg->position;
                    double t = toSec(joint_state_msg->header.stamp);

                    Eigen::Matrix<double, 4, 1> deltaD;
                    for (unsigned long int i=0; i<4; i++){
                        unsigned long int index = std::find(joint_state_msg->name.begin(), joint_state_msg->name.end(), wheel_names[i]) - joint_state_msg->name.begin();
                        if (index >= positions.size()) return;

                        if (initialization){
                            joint_state_positions[i] = positions[index];
                        } else {
                            deltaD[i] = wheel_radius * (positions[index] - joint_state_positions[i]);
                            joint_state_positions[i] = positions[index];
                        }
                    }

                    if (initialization) {initialization = false; t_derniere_publication = t; return;}

                    Eigen::Matrix<double, 3, 1> deplacement_robot = pseudo_inverse_matrice_roues * deltaD;

                    // contrainte des 4 roues : e != 0 => au moins une roue glisse
                    float e = deltaD[0] - deltaD[1] + deltaD[2] - deltaD[3];

                    Eigen::Matrix<double, 4, 4> Sigma_deltaD = Eigen::Matrix<double, 4, 4>::Zero();
                    for (int i=0; i<4; i++){
                        Sigma_deltaD(i, i) = sigma_q*sigma_q + coefficient_glissement*coefficient_glissement*std::abs(deltaD[i]);
                    }

                    Eigen::Matrix<double, 3, 3> covariance_deplacement = pseudo_inverse_matrice_roues * Sigma_deltaD * pseudo_inverse_matrice_roues.transpose();

                    if (std::abs(e) > e_max){
                        covariance_deplacement = lambda_glissement * covariance_deplacement;
                    }

                    integrer(deplacement_robot, covariance_deplacement);

                    // accumulation exprimée dans le repère robot au début de la fenêtre
                    double c = cos(deplacement_cumule[2]);
                    double s = sin(deplacement_cumule[2]);
                    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
                    R(0, 0) = c; R(0, 1) = -s;
                    R(1, 0) = s; R(1, 1) = c;
                    deplacement_cumule += R * deplacement_robot;
                    covariance_cumulee += R * covariance_deplacement * R.transpose();

                    double dt = t - t_derniere_publication;
                    if (dt < 1.0 / publish_rate) return;

                    publish_odom(joint_state_msg->header.stamp, dt);

                    deplacement_cumule = Eigen::Vector3d::Zero();
                    covariance_cumulee = Eigen::Matrix3d::Zero();
                    t_derniere_publication = t;
                }

                void integrer(const Eigen::Vector3d & deplacement_robot, const Eigen::Matrix3d & covariance_deplacement){
                    double theta_m = pose[2] + deplacement_robot[2] / 2;
                    double c = cos(theta_m);
                    double s = sin(theta_m);

                    double variation_position_x = -s*deplacement_robot[0] -c*deplacement_robot[1];
                    double variation_position_y = c*deplacement_robot[0] -s*deplacement_robot[1];

                    Eigen::Matrix3d jacobienne_pose = Eigen::Matrix3d::Identity();
                    jacobienne_pose(0, 2) = variation_position_x;
                    jacobienne_pose(1, 2) = variation_position_y;

                    Eigen::Matrix3d jacobienne_deplacement = Eigen::Matrix3d::Zero();
                    jacobienne_deplacement(0, 0) = c;
                    jacobienne_deplacement(0, 1) = -s;
                    jacobienne_deplacement(0, 2) = variation_position_x/2;
                    jacobienne_deplacement(1, 0) = s;
                    jacobienne_deplacement(1, 1) = c;
                    jacobienne_deplacement(1, 2) = variation_position_y/2;
                    jacobienne_deplacement(2, 2) = 1;

                    pose[0] += c*deplacement_robot[0] - s*deplacement_robot[1];
                    pose[1] += s*deplacement_robot[0] + c*deplacement_robot[1];
                    pose[2] = wrap(pose[2] + deplacement_robot[2]);

                    pose_covariance = jacobienne_pose*pose_covariance*jacobienne_pose.transpose()
                                    + jacobienne_deplacement*covariance_deplacement*jacobienne_deplacement.transpose();
                    pose_covariance = 0.5 * (pose_covariance + pose_covariance.transpose());
                }


                rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
                void publish_odom(const builtin_interfaces::msg::Time & stamp, double dt){
                    nav_msgs::msg::Odometry msg;
                    msg.header.stamp = stamp;
                    msg.header.frame_id = "odom";
                    msg.child_frame_id = "base_link";

                    msg.pose.pose.position.x = pose[0];
                    msg.pose.pose.position.y = pose[1];
                    msg.pose.pose.orientation.z = std::sin(0.5 * pose[2]);
                    msg.pose.pose.orientation.w = std::cos(0.5 * pose[2]);

                    // covariances ROS : 6x6 (x, y, z, roll, pitch, yaw), on remplit le bloc (x, y, yaw)
                    const int idx[3] = {0, 1, 5};
                    for (int i=0; i<3; i++){
                        for (int j=0; j<3; j++){
                            msg.pose.covariance[6*idx[i] + idx[j]] = pose_covariance(i, j);
                            msg.twist.covariance[6*idx[i] + idx[j]] = covariance_cumulee(i, j) / (dt*dt);
                        }
                    }
                    // axes non estimés : grande variance
                    for (int k : {2, 3, 4}){
                        msg.pose.covariance[6*k + k] = 1e3;
                        msg.twist.covariance[6*k + k] = 1e3;
                    }

                    msg.twist.twist.linear.x = deplacement_cumule[0] / dt;
                    msg.twist.twist.linear.y = deplacement_cumule[1] / dt;
                    msg.twist.twist.angular.z = deplacement_cumule[2] / dt;

                    odom_pub_->publish(msg);
                }

};


int main(int argc, char * argv[]){
        rclcpp::init(argc, argv);
        rclcpp::spin(std::make_shared<WheelOdometry>());
        rclcpp::shutdown();
        return 0;
}
