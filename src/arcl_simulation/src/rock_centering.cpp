#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include "geometry_msgs/msg/pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "robot_msgs/msg/robot_speed.hpp"
#include "ros2_aruco_interfaces/msg/aruco_markers.hpp"
#include "std_srvs/srv/trigger.hpp"

using namespace std::chrono_literals;
using ArucoMarkers = ros2_aruco_interfaces::msg::ArucoMarkers;
using RobotSpeed = robot_msgs::msg::RobotSpeed;
using Trigger = std_srvs::srv::Trigger;

class RockCentering : public rclcpp::Node {
    public:
        RockCentering() : Node("rock_centering") {
            aruco_markers_sub_ = create_subscription<ArucoMarkers>(
                "/aruco_markers", 10,
                std::bind(&RockCentering::aruco_markers_callback, this, std::placeholders::_1));

            // Service à réponse différée : on répond seulement quand le centrage est fini.
            centering_srv_ = create_service<Trigger>(
                "/rock_centering",
                [this](std::shared_ptr<rmw_request_id_t> header, std::shared_ptr<Trigger::Request> request) {
                    centering_callback(header, request);
                });

            robot_speed_pub_ = create_publisher<RobotSpeed>("/speed_command", 10);

            RCLCPP_INFO(get_logger(), "RockCentering launched successfully");
        }

    private:
        // Réglages
        int marker_id_ = 13;
        double x_ref_ = 0.0;
        double z_ref_ = 0.2;

        double kp_ = 0.8;
        double kp_theta_ = 1.0;

        double v_max_ = 0.25;
        double omega_max_ = 1.0;

        double tol_pos_ = 0.01;
        double tol_theta_ = 0.08;

        int stable_ticks_needed_ = 10;     // ticks consécutifs dans la tolérance
        double marker_timeout_ = 0.5;      // s sans tag -> le robot s'arrête

        double centering_timeout_ = 15.0;  // s

        // Dernière détection du tag
        bool has_pose_ = false;
        geometry_msgs::msg::Pose pose_;
        rclcpp::Time last_seen_;

        // État du centrage
        bool centering_active_ = false;
        int stable_ticks_ = 0;
        rclcpp::Time start_time_;
        std::shared_ptr<rmw_request_id_t> pending_request_header_;

        rclcpp::Subscription<ArucoMarkers>::SharedPtr aruco_markers_sub_;
        rclcpp::Service<Trigger>::SharedPtr centering_srv_;
        rclcpp::Publisher<RobotSpeed>::SharedPtr robot_speed_pub_;
        rclcpp::TimerBase::SharedPtr aruco_centering_timer_;

        void aruco_markers_callback(const ArucoMarkers::SharedPtr msg) {
            bool found = false;
            double best_z = 1e9;
            for (size_t i = 0; i < msg->marker_ids.size() && i < msg->poses.size(); ++i) {
                if (msg->marker_ids[i] != marker_id_) continue;

                if (msg->poses[i].position.z < best_z) {
                    best_z = msg->poses[i].position.z;
                    pose_ = msg->poses[i];
                    found = true;
                }
            }
            if (found) {
                has_pose_ = true;
                last_seen_ = now();
            }
        }

        void centering_callback(std::shared_ptr<rmw_request_id_t> header, std::shared_ptr<Trigger::Request> /*request*/) {
            if (centering_active_) {
                Trigger::Response response;
                response.success = false;
                response.message = "Centering already in progress";
                centering_srv_->send_response(*header, response);
                return;
            }
            centering_active_ = true;
            stable_ticks_ = 0;
            start_time_ = now();
            pending_request_header_ = header;
            aruco_centering_timer_ = create_wall_timer(20ms, std::bind(&RockCentering::aruco_centering_loop, this));
            RCLCPP_INFO(get_logger(), "Centering started");
        }

        void aruco_centering_loop() {
            if ((now() - start_time_).seconds() > centering_timeout_) {
                finish_centering(false, "Centering timeout");
                return;
            }

            if (!has_pose_ || (now() - last_seen_).seconds() > marker_timeout_) {
                stable_ticks_ = 0;
                publish_speed(0.0, 0.0, 0.0);
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Marker %d not visible, waiting", marker_id_);
                return;
            }

            const auto & p = pose_.position;
            const auto & q = pose_.orientation;

            // Normale du tag (axe z du tag) dans le repère optique = 3e colonne de la matrice de rotation.
            const double nx = 2.0 * (q.x * q.z + q.w * q.y);
            const double nz = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);

            const double err_fwd = p.z - z_ref_;              // > 0 : trop loin -> avancer
            const double err_lat = -(p.x - x_ref_);           // tag à droite (x > 0) -> aller à droite (vy < 0)
            const double err_theta = std::atan2(nx, -nz);     // 0 quand la caméra est face au tag

            const double vx = std::clamp(kp_ * err_fwd, -v_max_, v_max_);
            const double vy = std::clamp(kp_ * err_lat, -v_max_, v_max_);
            const double omega = std::clamp(kp_theta_ * err_theta, -omega_max_, omega_max_);

            const bool ok = std::fabs(err_fwd) < tol_pos_ && std::fabs(err_lat) < tol_pos_ && std::fabs(err_theta) < tol_theta_;
            stable_ticks_ = ok ? stable_ticks_ + 1 : 0;

            if (stable_ticks_ >= stable_ticks_needed_) {
                finish_centering(true, "Centered on ArUco");
                return;
            }
            publish_speed(vx, vy, omega);
        }

        void publish_speed(double vx, double vy, double omega) {
            RobotSpeed msg;
            msg.vx = vx;
            msg.vy = vy;
            msg.omega = omega;
            robot_speed_pub_->publish(msg);
        }

        void finish_centering(bool success, const std::string & message) {
            if (aruco_centering_timer_) aruco_centering_timer_->cancel();
            publish_speed(0.0, 0.0, 0.0);

            Trigger::Response response;
            response.success = success;
            response.message = message;
            centering_srv_->send_response(*pending_request_header_, response);

            pending_request_header_.reset();
            centering_active_ = false;
            RCLCPP_INFO(get_logger(), "Centering finished: %s", message.c_str());
        }
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RockCentering>());
    rclcpp::shutdown();
    return 0;
}