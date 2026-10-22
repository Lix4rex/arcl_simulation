#include "rclcpp/rclcpp.hpp"
#include "ros2_aruco_interfaces/msg/aruco_markers.hpp"

class CamTracking : public rclcpp::Node{

        public:

            CamTracking() : Node("cam_tracking"){

                aruco_markers_sub_ = create_subscription<ros2_aruco_interfaces::msg::ArucoMarkers>(
                    "/deported_aruco_markers", 10,
                    std::bind(&CamTracking::aruco_markers_callback, this, std::placeholders::_1)
                );


                RCLCPP_INFO(this->get_logger(), "CamTracking node launched successfuly");
            }


        private:

            rclcpp::Subscription<ros2_aruco_interfaces::msg::ArucoMarkers>::SharedPtr aruco_markers_sub_;
            void aruco_markers_callback(const ros2_aruco_interfaces::msg::ArucoMarkers::SharedPtr aruco_markers){
                unsigned long int n = aruco_markers->marker_ids.size();
                RCLCPP_INFO(this->get_logger(), "%ld markers détecté(s)", n);
            }

};


int main(int argc, char * argv[]){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CamTracking>());
    if (rclcpp::ok()) rclcpp::shutdown();
    return 0;
}