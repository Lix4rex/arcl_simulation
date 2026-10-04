#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include <vector>
#include <string>

using namespace std::chrono_literals;

class TableMarkers : public rclcpp::Node{

        public:
            TableMarkers() : Node("table_markers"){

                markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
                "/table_markers", 10
                                        );

                // on republie chaque seconde : RViz les voit même s'il est lancé après ce nœud
                timer_ = this->create_wall_timer(1s, std::bind(&TableMarkers::publish_table, this));

                RCLCPP_INFO(this->get_logger(), "TableMarkers launched successfuly");
            }


        private:

                std::vector<std::vector<float>> murs = {
                        {1.5,    -0.011, 0.035,  3.044, 0.022, 0.07},
                        {1.5,     2.011, 0.035,  3.044, 0.022, 0.07},
                        {-0.011,  1.0,   0.035,  0.022, 2.0,   0.07},
                        {3.011,   1.0,   0.035,  0.022, 2.0,   0.07}
                };

                std::string paquet = "ekf_ros_simulation";

                std::vector<std::vector<float>> balises = {{-0.094, 0.05}, {-0.094, 1.95}, {3.094, 1.0}};
                float r_balise = 0.05;
                float hauteur_balise = 0.5;

                rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
                rclcpp::TimerBase::SharedPtr timer_;

                void publish_table(){
                    visualization_msgs::msg::MarkerArray markers;
                    int id = 0;

                    visualization_msgs::msg::Marker tapis = make_marker(id++, visualization_msgs::msg::Marker::MESH_RESOURCE, 0.0, 0.0, 0.001,   1.0, 1.0, 1.0,   0.0, 0.0, 0.0, 0.0);
                    tapis.mesh_resource = "package://arcl_simulation/worlds/tapis_2027.dae";
                    tapis.mesh_use_embedded_materials = true;
                    markers.markers.push_back(tapis);

                    for (unsigned long int i=0; i<murs.size(); i++){
                            markers.markers.push_back(make_marker(id++, visualization_msgs::msg::Marker::CUBE, murs[i][0], murs[i][1], murs[i][2],   murs[i][3], murs[i][4], murs[i][5],   0.6, 0.6, 0.6, 1.0));
                    }

                    for (unsigned long int i=0; i<balises.size(); i++){
                            markers.markers.push_back(make_marker(id++, visualization_msgs::msg::Marker::CYLINDER, balises[i][0], balises[i][1], hauteur_balise/2,   2*r_balise, 2*r_balise, hauteur_balise,   0.1, 0.1, 0.1, 1.0));
                    }

                    markers_pub_->publish(markers);
                }

                visualization_msgs::msg::Marker make_marker(int id, int type, float x, float y, float z, float sx, float sy, float sz, float r, float g, float b, float a){
                    visualization_msgs::msg::Marker marker;
                    marker.header.frame_id = "map";
                    marker.header.stamp = this->now();
                    marker.ns = "table";
                    marker.id = id;
                    marker.type = type;
                    marker.action = visualization_msgs::msg::Marker::ADD;
                    marker.pose.position.x = x;
                    marker.pose.position.y = y;
                    marker.pose.position.z = z;
                    marker.pose.orientation.w = 1.0;
                    marker.scale.x = sx;
                    marker.scale.y = sy;
                    marker.scale.z = sz;
                    marker.color.r = r;
                    marker.color.g = g;
                    marker.color.b = b;
                    marker.color.a = a;
                    return marker;
                }
};


int main(int argc, char * argv[]){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TableMarkers>());
    rclcpp::shutdown();
    return 0;
}