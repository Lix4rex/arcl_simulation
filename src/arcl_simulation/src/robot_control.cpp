#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "robot_msgs/msg/wheel_commands.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "robot_msgs/msg/obstacle_array.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker.hpp"

#include <cmath>
#include <algorithm>
#include <vector>
#include <Eigen/Dense>

using namespace std::chrono_literals;

double wrap(double a)
{
  a = std::fmod(a + M_PI, 2.0 * M_PI);
  if (a < 0.0) {a += 2.0 * M_PI;}
  return a - M_PI;
}

struct TrajectoryPoints{
        int index;
        double x;
        double y;
};

struct TrajectoryTangentes{
        int index;
        double x;
        double y;
};

class RobotControl : public rclcpp::Node{

        public:
        RobotControl() : Node("robot_control"){

                        cmd_vel_wheel1_pub_ = this->create_publisher<std_msgs::msg::Float64>(
                                "/robot_omni/roue_1/cmd_vel", 10
                        );

                        cmd_vel_wheel2_pub_ = this->create_publisher<std_msgs::msg::Float64>(
                                "/robot_omni/roue_2/cmd_vel", 10
                        );

                        cmd_vel_wheel3_pub_ = this->create_publisher<std_msgs::msg::Float64>(
                                "/robot_omni/roue_3/cmd_vel", 10
                        );

                        cmd_vel_wheel4_pub_ = this->create_publisher<std_msgs::msg::Float64>(
                                "/robot_omni/roue_4/cmd_vel", 10
                        );

                        wheel_commands_sub_ = this->create_subscription<robot_msgs::msg::WheelCommands>(
                                "/wheel_commands", 10,
                                std::bind(&RobotControl::wheel_commands_callback, this, std::placeholders::_1)
                        );

                        robot_position_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
                                "/pose_estimation", 10,
                                std::bind(&RobotControl::robot_position_callback, this, std::placeholders::_1)
                        );

                        goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
                                "/goal_pose", 10,
                                std::bind(&RobotControl::goal_callback, this, std::placeholders::_1)
                        );

                        obstacles_sub_ = this->create_subscription<robot_msgs::msg::ObstacleArray>(
                                "/obstacles_scan", 10,
                                std::bind(&RobotControl::obstacles_callback, this, std::placeholders::_1)
                        );

                        obstacle_point_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
                                "/obstacle_point", 10
                        );

                        path_pub_ = this->create_publisher<nav_msgs::msg::Path>(
                                "/trajectoire", 10
                        );

                        control_timer_ = this->create_wall_timer(20ms, std::bind(&RobotControl::trajectory_control_loop, this));

                        trajectory_generation_timer = this->create_wall_timer(100ms, std::bind(&RobotControl::trajectory_generation_loop, this));

                        RCLCPP_INFO(this->get_logger(), "RobotControl launched successfuly");
                }


        private:
                rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cmd_vel_wheel1_pub_;
                rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cmd_vel_wheel2_pub_;
                rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cmd_vel_wheel3_pub_;
                rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cmd_vel_wheel4_pub_;

                rclcpp::Subscription<robot_msgs::msg::WheelCommands>::SharedPtr wheel_commands_sub_;
                void wheel_commands_callback(const robot_msgs::msg::WheelCommands::SharedPtr wheel_commands){
                        goal_active = false;
                        trajectoire_active = false;
                        send_wheel_speeds(wheel_commands->wheel1, wheel_commands->wheel2, wheel_commands->wheel3, wheel_commands->wheel4);
                }

                float wheel_radius = 0.03;
                float wheel_distance_from_center = 0.162635;
                float robot_radius = 0.2f;
                float secu_marge = 0.1f;

                float v_max_trajectoire = 0.2;
                float v_angulaire_max_trajectire = 2.0;
                float a_max_trajectoire = 0.8;

                static const int nb_points = 15;
                int nb_pas_table = 6000;


                float Kp = 2.0;
                float Kp_theta = 1.0;


                bool position_received = false;
                bool goal_active = false;

                float x = 0, y = 0, theta = 0;
                float x_goal = 0, y_goal = 0, theta_goal = 0;

                rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr robot_position_sub_;
                void robot_position_callback(const geometry_msgs::msg::PoseStamped::SharedPtr position){
                        x = position->pose.position.x;
                        y = position->pose.position.y;
                        float z_orientation = position->pose.orientation.z;
                        float w_orientation = position->pose.orientation.w;
                        theta = 2*atan2(z_orientation, w_orientation);
                        position_received = true;
                }


                TrajectoryPoints p[nb_points];
                TrajectoryTangentes m[nb_points];

                float u;

                float t_acceleration = 0;
                float d_acceleration = 0;
                float t_croisiere = 0;
                float v_pic = 0;
                float duree_trajectoire = 0;

                std::vector<float> table_u;
                std::vector<float> table_s;
                float longueur_trajectoire = 0;

                bool trajectoire_active = false;
                double t_debut_trajectoire = 0;
                float theta_depart = 0;

                float h00(float t){
                        return 2*t*t*t - 3*t*t + 1;
                }
                float h00_derive(float t){
                        return 6*t*t - 6*t;
                }
                float h10(float t){
                        return t*t*t - 2*t*t + t;
                }
                float h10_derive(float t){
                        return 3*t*t - 4*t + 1;
                }
                float h01(float t){
                        return -2*t*t*t + 3*t*t;
                }
                float h01_derive(float t){
                        return -6*t*t + 6*t;
                }
                float h11(float t){
                        return t*t*t - t*t;
                }
                float h11_derive(float t){
                        return 3*t*t - 2*t;
                }

                int index_morceau(float u){
                        return std::clamp((int)floor(u), 0, nb_points - 2);
                }

                Eigen::Matrix<double, 2, 1> courbe(float u){
                        Eigen::Matrix<double, 2, 1> value;
                        int index = index_morceau(u);
                        float t = u-index;

                        value(0, 0) = h00(t)*p[index].x + h10(t)*m[index].x + h01(t)*p[index+1].x + h11(t)*m[index+1].x;
                        value(1, 0) = h00(t)*p[index].y + h10(t)*m[index].y + h01(t)*p[index+1].y + h11(t)*m[index+1].y;

                        return value;
                }

                Eigen::Matrix<double, 2, 1> courbe_derive(float u){
                        Eigen::Matrix<double, 2, 1> value;
                        int index = index_morceau(u);
                        float t = u-index;

                        value(0, 0) = h00_derive(t)*p[index].x + h10_derive(t)*m[index].x + h01_derive(t)*p[index+1].x + h11_derive(t)*m[index+1].x;
                        value(1, 0) = h00_derive(t)*p[index].y + h10_derive(t)*m[index].y + h01_derive(t)*p[index+1].y + h11_derive(t)*m[index+1].y;

                        return value;
                }

                void construire_table_distances(){
                        table_u.clear();
                        table_s.clear();

                        float u_max = nb_points - 1;
                        Eigen::Matrix<double, 2, 1> point_precedent = courbe(0);
                        float s = 0;

                        for (int k=0; k<=nb_pas_table; k++){
                                float u_k = u_max * k / nb_pas_table;
                                Eigen::Matrix<double, 2, 1> point = courbe(u_k);
                                s += (point - point_precedent).norm();
                                point_precedent = point;

                                table_u.push_back(u_k);
                                table_s.push_back(s);
                        }

                        longueur_trajectoire = s;
                }

                float u_depuis_s(float s){
                        if (s <= 0) return 0;
                        if (s >= longueur_trajectoire) return table_u.back();

                        // premier indice k tel que table_s[k] >= s
                        int k = std::lower_bound(table_s.begin(), table_s.end(), s) - table_s.begin();
                        if (k == 0) return table_u[0];

                        float ds = table_s[k] - table_s[k-1];
                        if (ds < 1e-9) return table_u[k];
                        float alpha = (s - table_s[k-1]) / ds;
                        return table_u[k-1] + alpha * (table_u[k] - table_u[k-1]);
                }

                void calculer_profil(){
                        float D = longueur_trajectoire;

                        v_pic = v_max_trajectoire;
                        t_acceleration = v_pic / a_max_trajectoire;
                        d_acceleration = v_pic*v_pic / (2*a_max_trajectoire);

                        if (2*d_acceleration > D){
                                v_pic = sqrt(D * a_max_trajectoire);
                                t_acceleration = v_pic / a_max_trajectoire;
                                d_acceleration = D / 2;
                        }

                        t_croisiere = (D - 2*d_acceleration) / v_pic;
                        duree_trajectoire = 2*t_acceleration + t_croisiere;
                }

                void profil(float t, float & s, float & s_point){
                        float D = longueur_trajectoire;
                        float T = duree_trajectoire;
                        float a = a_max_trajectoire;

                        if (t <= 0){
                                s = 0;
                                s_point = 0;
                        } else if (t < t_acceleration){
                                s = 0.5*a*t*t;
                                s_point = a*t;
                        } else if (t < t_acceleration + t_croisiere){
                                s = d_acceleration + v_pic*(t - t_acceleration);
                                s_point = v_pic;
                        } else if (t < T){
                                s = D - 0.5*a*(T - t)*(T - t);
                                s_point = a*(T - t);
                        } else {
                                s = D;
                                s_point = 0;
                        }
                }


                rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
                rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
                void goal_callback(const geometry_msgs::msg::PoseStamped::SharedPtr goal){
                        x_goal = goal->pose.position.x;
                        y_goal = goal->pose.position.y;
                        theta_goal = 2*atan2(goal->pose.orientation.z, goal->pose.orientation.w);

                        create_trajectory();
                        goal_active = true;

                        RCLCPP_INFO(this->get_logger(), "Nouvel objectif : (%.3f, %.3f, %.1f°), trajectoire de %.3f m en %.2f s", x_goal, y_goal, theta_goal*180/M_PI, longueur_trajectoire, duree_trajectoire);
                }


                rclcpp::TimerBase::SharedPtr trajectory_generation_timer;
                void trajectory_generation_loop(){
                        if (!trajectoire_active || !check_conflict()) return;
                        RCLCPP_INFO(this->get_logger(), "Il y a conflit, CHANGEMENT DE TRAJECTOIRE NECESSAIRE");
                        create_trajectory();
                }
                
                rclcpp::Subscription<robot_msgs::msg::ObstacleArray>::SharedPtr obstacles_sub_;
                std::vector<robot_msgs::msg::Obstacle> obstacles; 
                void obstacles_callback(const robot_msgs::msg::ObstacleArray::SharedPtr cluster_msg){
                        obstacles.clear();
                        obstacles.reserve(cluster_msg->obstacles.size());

                        int i=0;
                        for (const auto & obs : cluster_msg->obstacles) {
                                publish_obstacle_markers(cluster_msg, obs.x, obs.y, 0.75*obs.radius, i);
                                obstacles.emplace_back(obs);
                                i++;
                        }
                }

                bool check_conflict(){
                        for (size_t k = 0; k < table_u.size(); k++){
                                Eigen::Matrix<double, 2, 1> point = courbe(table_u[k]);
                                float px = point(0);
                                float py = point(1);

                                for (const auto &obs : obstacles){
                                        float distance_contact = robot_radius;

                                        float vx = px - obs.x;
                                        float vy = py - obs.y;
                                        float distance_obstacle = sqrt(vx*vx + vy*vy);

                                        if (distance_obstacle <= distance_contact) return true;
                                }
                        }
                        return false;
                }

                void create_trajectory(){
                        u=0;

                        float distance = sqrt((x_goal - x)*(x_goal - x) + (y_goal - y)*(y_goal - y));
                        float dir_x = (x_goal - x) / distance;
                        float dir_y = (y_goal - y) / distance;

                        float delta_x = (x_goal - x)/(nb_points - 1);
                        float delta_y = (y_goal - y)/(nb_points - 1);

                        for (int i=0; i<nb_points; i++){
                                float px = x + i*delta_x;
                                float py = y + i*delta_y;

                                if (i > 0 && i < nb_points - 1){
                                        for (const auto &obs : obstacles){
                                                float distance_evitement = robot_radius + secu_marge;

                                                float vx = px - obs.x;
                                                float vy = py - obs.y;
                                                float distance_obstacle = sqrt(vx*vx + vy*vy);

                                                if (distance_obstacle > distance_evitement) continue;

                                                float perp_x = -dir_y;
                                                float perp_y =  dir_x;
                                                float position_laterale_centre = vx*perp_x + vy*perp_y; // produit scalaire du vecteur (point -> obstacle) / (vecteur normal à la trajectoire)
                                                if (position_laterale_centre < 0) {
                                                        perp_x = -perp_x;
                                                        perp_y = -perp_y;
                                                }

                                                float avance = vx*dir_x + vy*dir_y;
                                                float decalage = sqrt(distance_evitement*distance_evitement - avance*avance);

                                                px = obs.x + avance*dir_x + decalage*perp_x;
                                                py = obs.y + avance*dir_y + decalage*perp_y;
                                        }
                                }
                                p[i] = {i, px, py};
                        }

                        m[0] = {0, p[1].x - p[0].x, p[1].y - p[0].y};
                        for (int i=1; i<nb_points-1; i++){
                                m[i] = {i, 0.5*(p[i+1].x - p[i-1].x), 0.5*(p[i+1].y - p[i-1].y)};
                        }
                        m[nb_points-1] = {nb_points-1, p[nb_points-1].x - p[nb_points-2].x, p[nb_points-1].y - p[nb_points-2].y};

                        construire_table_distances();
                        calculer_profil();

                        theta_depart = theta;
                        t_debut_trajectoire = this->now().seconds();
                        trajectoire_active = longueur_trajectoire > 0.001;

                        publish_calculated_points();
                }

                void publish_calculated_points(){
                        nav_msgs::msg::Path chemin;
                        chemin.header.stamp = this->now();
                        chemin.header.frame_id = "map";

                        for (size_t k = 0; k < table_u.size(); k++) {
                                Eigen::Matrix<double, 2, 1> point = courbe(table_u[k]);

                                geometry_msgs::msg::PoseStamped pose;
                                pose.header = chemin.header;
                                pose.pose.position.x = point(0);
                                pose.pose.position.y = point(1);
                                pose.pose.orientation.w = 1.0;
                                chemin.poses.push_back(pose);
                        }

                        path_pub_->publish(chemin);
                }

                rclcpp::TimerBase::SharedPtr control_timer_;
                void trajectory_control_loop(){
                        if (!position_received || !goal_active) return;

                        // 1. temps écoulé depuis le départ (horloge de la simulation si use_sim_time = true)
                        float t = this->now().seconds() - t_debut_trajectoire;
                        if (t >= duree_trajectoire){
                                trajectoire_active = false;
                                RCLCPP_INFO(this->get_logger(), "Fin de trajectoire, objectif atteint");
                                goal_active = false;
                                send_wheel_speeds(0, 0, 0, 0);
                                return;
                        }

                        // 2. profil : combien de mètres sur la courbe, et à quelle vitesse
                        float s, s_point;
                        profil(t, s, s_point);

                        // 3. table : quel u correspond à ces s mètres
                        u = u_depuis_s(s);

                        // 4. consigne : où le robot devrait être, et dans quelle direction va la courbe
                        Eigen::Matrix<double, 2, 1> p_ref = courbe(u);
                        Eigen::Matrix<double, 2, 1> derivee = courbe_derive(u);
                        Eigen::Matrix<double, 2, 1> tangente = derivee / derivee.norm();
                        Eigen::Matrix<double, 2, 1> v_ref = s_point * tangente;

                        // cap : il tourne au même rythme que la distance parcourue
                        float delta_theta = wrap(theta_goal - theta_depart);
                        float theta_ref = wrap(theta_depart + delta_theta * s / longueur_trajectoire);
                        float omega_ref = delta_theta * s_point / longueur_trajectoire;

                        // 5. commande = anticipation + correction de l'écart
                        float vx_table = v_ref(0, 0) + Kp*(p_ref(0, 0) - x);
                        float vy_table = v_ref(1, 0) + Kp*(p_ref(1, 0) - y);
                        float omega = omega_ref + Kp_theta*wrap(theta_ref - theta);

                        float vitesse = sqrt(vx_table*vx_table + vy_table*vy_table);
                        if (vitesse > v_max_trajectoire){
                                vx_table = vx_table * v_max_trajectoire / vitesse;
                                vy_table = vy_table * v_max_trajectoire / vitesse;
                        }
                        omega = std::clamp(omega, -v_angulaire_max_trajectire, v_angulaire_max_trajectire);

                        send_robot_speed(vx_table, vy_table, omega);
                }

                void send_robot_speed(float vx_table, float vy_table, float omega){

                        float vx_robot =  cos(theta)*vx_table + sin(theta)*vy_table;
                        float vy_robot = -sin(theta)*vx_table + cos(theta)*vy_table;

                        float a = sqrt(2)/2;
                        float L = wheel_distance_from_center;
                        float v1 = -a*vx_robot + a*vy_robot + L*omega;
                        float v2 = -a*vx_robot - a*vy_robot + L*omega;
                        float v3 =  a*vx_robot - a*vy_robot + L*omega;
                        float v4 =  a*vx_robot + a*vy_robot + L*omega;

                        send_wheel_speeds(v1/wheel_radius, v2/wheel_radius, v3/wheel_radius, v4/wheel_radius);

                }

                void send_wheel_speeds(float w1, float w2, float w3, float w4){
                        std_msgs::msg::Float64 msg;

                        msg.data = w1;
                        cmd_vel_wheel1_pub_ ->publish(msg);

                        msg.data = w2;
                        cmd_vel_wheel2_pub_ ->publish(msg);

                        msg.data = w3;
                        cmd_vel_wheel3_pub_ ->publish(msg);

                        msg.data = w4;
                        cmd_vel_wheel4_pub_ ->publish(msg);
                }


                rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr obstacle_point_pub_;
                void publish_obstacle_markers(const robot_msgs::msg::ObstacleArray::SharedPtr obstacle_msg, float x, float y, float radius, int i){
                    visualization_msgs::msg::Marker marker;
                    marker.header.frame_id = obstacle_msg->header.frame_id;
                    marker.header.stamp = obstacle_msg->header.stamp;
                    marker.ns = "obstacles";
                    marker.id = i;
                    marker.type = visualization_msgs::msg::Marker::CYLINDER;
                    marker.action = visualization_msgs::msg::Marker::ADD;
                    marker.pose.position.x = x;
                    marker.pose.position.y = y;
                    marker.pose.position.z = 0.0;
                    marker.pose.orientation.w = 1.0;
                    marker.scale.x = radius;
                    marker.scale.y = radius;
                    marker.scale.z = 1.0;
                    marker.color.r = 1.0;
                    marker.color.g = 0.0;
                    marker.color.b = 0.0;
                    marker.color.a = 1.0;
                    marker.lifetime = rclcpp::Duration::from_seconds(0.2);
                    obstacle_point_pub_->publish(marker);
                }
};


int main(int argc, char * argv[]){
        rclcpp::init(argc, argv);
        rclcpp::spin(std::make_shared<RobotControl>());
        rclcpp::shutdown();
        return 0;
}