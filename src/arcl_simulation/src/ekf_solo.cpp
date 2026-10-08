#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "robot_msgs/msg/obstacle_array.hpp"

#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <string>
#include <algorithm>
#include <cmath>

struct HistoryEntry
{
  double t;   // tk
  Eigen::Matrix<double, 3, 1> state_estimation;
  Eigen::Matrix<double, 3, 3> state_covariance_estimation;
  Eigen::Matrix<double, 3, 1> deplacement_robot;
  Eigen::Matrix<double, 3, 3> covariance_deplacement;
};

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

using namespace std::chrono_literals;

class EKFSolo : public rclcpp::Node{

        public:
                EKFSolo() : Node("ekf_solo"){

                        joint_states_sub_ = create_subscription<sensor_msgs::msg::JointState>(
                            "/joint_states", 10,
                            std::bind(&EKFSolo::onJointStates, this, std::placeholders::_1)
                        );

                        estimate_position_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
                            "/pose_estimation", 10
                        );

                        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
                                "/scan", 10,
                                std::bind(&EKFSolo::scan_callback, this, std::placeholders::_1)
                        );

                        balise_point_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
                                "/balise_marker", 10
                        );

                        obstacles_pub_ = this->create_publisher<robot_msgs::msg::ObstacleArray>(
                                "/obstacles_scan", 10
                        );

                        float a = sqrt(2)/4;
                        float b = 1 / (4*wheel_distance_from_center);
                        pseudo_inverse_matrice_roues <<
                            -a, -a, a, a,
                            a, -a, -a, a,
                            b, b, b, b;

                        state_estimation << 0.5, 1.0, 0.0;

                        // P0 : 2 cm et 3° de placement à la main
                        state_covariance_estimation = Eigen::Matrix3d::Zero();
                        state_covariance_estimation(0, 0) = 0.02*0.02;
                        state_covariance_estimation(1, 1) = 0.02*0.02;
                        state_covariance_estimation(2, 2) = 0.052*0.052;

                        Q_ = Eigen::Matrix3d::Zero();
                        Q_(0, 0) = 1e-7;
                        Q_(1, 1) = 1e-7;
                        Q_(2, 2) = 1e-6;

                        Rm_(0, 0) = sigma_rho_*sigma_rho_;
                        Rm_(0, 1) = 0;
                        Rm_(1, 0) = 0;
                        Rm_(1, 1) = sigma_phi_*sigma_phi_;

                        RCLCPP_INFO(this->get_logger(), "EKFSolo launched successfuly");
                }


        private:

                float wheel_radius = 0.03;
                float wheel_distance_from_center = 0.162635;
                float r_balise = 0.05;

                float nb_ticks = 4096;

                float e_max = 2e-4;
                float lambda_glissement = 30;
                float q = 2*M_PI*wheel_radius/nb_ticks;
                float sigma_q = q*1/sqrt(12);

                float coefficient_glissement = 0.01;

                float gamma_porte = 9.21;

                std::vector<std::string> wheel_names = {"roue_1_joint", "roue_2_joint", "roue_3_joint", "roue_4_joint"};
                std::vector<std::vector<float>> balises = {{-0.094, 0.05}, {-0.094, 1.95}, {3.094, 1.0}};

                std::vector<float> initial_state;
                std::vector<std::vector<float>> initial_state_covariance;

                Eigen::Matrix<double, 3, 1> state_estimation;
                Eigen::Matrix<double, 3, 3> state_covariance_estimation;
                Eigen::Matrix<double, 3, 3> Q_;



                Eigen::Matrix<double, 4, 1> joint_state_positions;
                Eigen::Matrix<double, 3, 4> pseudo_inverse_matrice_roues;
                bool initialization = true;

                float sigma_rho_ = 0.02;
                float sigma_phi_ = 0.013;
                Eigen::Matrix<double, 2, 2> Rm_;

                std::deque<HistoryEntry> states_history_;


                rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
                void onJointStates(const sensor_msgs::msg::JointState::SharedPtr joint_state_msg){
                    auto positions = joint_state_msg->position;

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

                    if (initialization) {initialization = false; return;}

                    Eigen::Matrix<double, 3, 1> deplacement_robot = pseudo_inverse_matrice_roues * deltaD;

                    float e = deltaD[0] - deltaD[1] + deltaD[2] - deltaD[3];

                    Eigen::Matrix<double, 4, 4> Sigma_deltaD = Eigen::Matrix<double, 4, 4>::Zero();
                    for (int i=0; i<4; i++){
                        Sigma_deltaD(i, i) = sigma_q*sigma_q + coefficient_glissement*coefficient_glissement*std::abs(deltaD[i]);
                    }

                    Eigen::Matrix<double, 3, 3> covariance_deplacement = pseudo_inverse_matrice_roues * Sigma_deltaD * pseudo_inverse_matrice_roues.transpose();

                    if (std::abs(e) > e_max){
                        covariance_deplacement = lambda_glissement * covariance_deplacement;
                    }

                    prediction(state_estimation, state_covariance_estimation, deplacement_robot, covariance_deplacement);

                    double t = toSec(joint_state_msg->header.stamp);
                    states_history_.push_back({t, state_estimation, state_covariance_estimation, deplacement_robot, covariance_deplacement});

                    // on ne garde que les 200 dernières ms
                    while (states_history_.size() > 2 && states_history_.front().t < t - 0.2){
                        states_history_.pop_front();
                    }

                    publish_estimate_position(joint_state_msg->header.stamp);
                }

                void prediction(Eigen::Matrix<double, 3, 1> & state_estimation, Eigen::Matrix<double, 3, 3> & state_covariance_estimation,
                                Eigen::Matrix<double, 3, 1> deplacement_robot, Eigen::Matrix<double, 3, 3> covariance_deplacement){

                    float theta_m = state_estimation[2] + deplacement_robot[2] / 2;
                    float c = cos(theta_m);
                    float s = sin(theta_m);

                    // jacobiennes calculées avec l'état AVANT la prédiction
                    float variation_position_x = -s*deplacement_robot[0] -c*deplacement_robot[1];
                    float variation_position_y = c*deplacement_robot[0] -s*deplacement_robot[1];

                    Eigen::Matrix<double, 3, 3> jacobienne_prediction_pose = Eigen::Matrix3d::Identity();
                    jacobienne_prediction_pose(0, 2) = variation_position_x;
                    jacobienne_prediction_pose(1, 2) = variation_position_y;

                    Eigen::Matrix<double, 3, 3> jacobienne_prediction_deplacement = Eigen::Matrix<double, 3, 3>::Zero();
                    jacobienne_prediction_deplacement(0, 0) = c;
                    jacobienne_prediction_deplacement(0, 1) = -s;
                    jacobienne_prediction_deplacement(0, 2) = variation_position_x/2;
                    jacobienne_prediction_deplacement(1, 0) = s;
                    jacobienne_prediction_deplacement(1, 1) = c;
                    jacobienne_prediction_deplacement(1, 2) = variation_position_y/2;
                    jacobienne_prediction_deplacement(2, 2) = 1;

                    state_estimation[0] += c*deplacement_robot[0] - s*deplacement_robot[1];
                    state_estimation[1] += s*deplacement_robot[0] + c*deplacement_robot[1];
                    state_estimation[2] += deplacement_robot[2];
                    state_estimation[2] = wrap(state_estimation[2]);

                    state_covariance_estimation = jacobienne_prediction_pose*state_covariance_estimation*jacobienne_prediction_pose.transpose()
                                                + jacobienne_prediction_deplacement*covariance_deplacement*jacobienne_prediction_deplacement.transpose()
                                                + Q_;

                    state_covariance_estimation = 0.5 * (state_covariance_estimation + state_covariance_estimation.transpose());
                }














                const float tolerance_largeur = 0.02;
                const float distance_max_balise = 0.3;
                const float marge = 0.05f;
                rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
                rclcpp::Publisher<robot_msgs::msg::ObstacleArray>::SharedPtr obstacles_pub_;
                void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg){

                        if (states_history_.empty()) return;

                        double tm = toSec(scan_msg->header.stamp);
                        int k = 0;
                        for (size_t i = 0; i < states_history_.size(); ++i) {
                        if (states_history_[i].t <= tm) {k = i;} else {break;}
                        }

                        Eigen::Matrix<double, 3, 1> state_estimation = states_history_[k].state_estimation;
                        Eigen::Matrix<double, 3, 3> state_covariance_estimation = states_history_[k].state_covariance_estimation;

                        std::vector<std::vector<int>> clusters = dbScan(0.1, 2, scan_msg);

                        std::vector<std::vector<float>> cluster_centers;

                        std::vector<std::vector<float>> obstacles_monde;
                        
                        robot_msgs::msg::ObstacleArray out;
                        out.header.stamp = scan_msg->header.stamp;
                        out.header.frame_id = "map";

                        for (unsigned long int j=0; j<clusters.size(); j++){
                                const std::vector<int> &cluster = clusters[j];

                                std::vector<std::vector<float>> pts;
                                for (unsigned long int i=0; i<cluster.size(); i++){
                                        float angle = scan_msg->angle_min + cluster[i]*scan_msg->angle_increment;
                                        float r = scan_msg->ranges[cluster[i]];
                                        pts.push_back({r * (float)cos(angle), r * (float)sin(angle)});
                                }
                                if (pts.empty()) continue;

                                float p_moy_x = 0, p_moy_y = 0;
                                for (const auto &pt : pts){ p_moy_x += pt[0]; p_moy_y += pt[1]; }
                                p_moy_x /= pts.size();
                                p_moy_y /= pts.size();
                                float norme_p_moy = sqrt(p_moy_x*p_moy_x + p_moy_y*p_moy_y);
                                if (norme_p_moy < 1e-3) continue;

                                float theta_r = state_estimation[2];
                                float x_obs = state_estimation[0] + p_moy_x*cos(theta_r) - p_moy_y*sin(theta_r);
                                float y_obs = state_estimation[1] + p_moy_x*sin(theta_r) + p_moy_y*cos(theta_r);

                                float largeur = 0;
                                for (size_t a=0; a<pts.size(); a++)
                                        for (size_t b=a+1; b<pts.size(); b++)
                                                largeur = std::max(largeur, (float)hypot(pts[a][0]-pts[b][0], pts[a][1]-pts[b][1]));

                                float pas_laser   = norme_p_moy * scan_msg->angle_increment;
                                float largeur_max = 2*r_balise + tolerance_largeur;
                                float largeur_min = 2*r_balise - 2*pas_laser - tolerance_largeur;

                                if (largeur > largeur_max || largeur < largeur_min){
                                        obstacles_monde.push_back({x_obs, y_obs});
                                        robot_msgs::msg::Obstacle obs;
                                        obs.x = x_obs;
                                        obs.y = y_obs;
                                        obs.radius = largeur / 2.0f + marge; 
                                        out.obstacles.push_back(obs);
                                        continue;
                                }

                                std::vector<float> c0 = { p_moy_x + r_balise * p_moy_x / norme_p_moy,
                                                        p_moy_y + r_balise * p_moy_y / norme_p_moy };
                                std::vector<float> cluster_center = gauss_newton(c0, pts);

                                float z_r   = sqrt(cluster_center[0]*cluster_center[0] + cluster_center[1]*cluster_center[1]);
                                float z_phi = atan2(cluster_center[1], cluster_center[0]);

                                float xw = state_estimation[0] + z_r * cos(z_phi + theta_r);
                                float yw = state_estimation[1] + z_r * sin(z_phi + theta_r);

                                float sigma_pos = sqrt(std::max(state_covariance_estimation(0, 0), state_covariance_estimation(1, 1)));
                                float seuil_distance = std::max(distance_max_balise, (float)(3*sigma_pos));

                                float d_min = 1e9;
                                for (const auto &bal : balises)
                                        d_min = std::min(d_min, (float)hypot(bal[0] - xw, bal[1] - yw));

                                if (d_min > seuil_distance){
                                        obstacles_monde.push_back({x_obs, y_obs});
                                        robot_msgs::msg::Obstacle obs;
                                        obs.x = x_obs;
                                        obs.y = y_obs;
                                        obs.radius = largeur / 2.0f + marge; 
                                        out.obstacles.push_back(obs);
                                        continue;
                                }

                                cluster_centers.push_back({z_r, z_phi});
                                publish_markers(scan_msg, cluster_center, j);
                        }

                        obstacles_pub_->publish(out);

                        for (unsigned long int c_index=0; c_index<cluster_centers.size(); c_index++){

                                float meilleur_d_carre = gamma_porte;
                                Eigen::Matrix<double, 2, 1> meilleure_innovation;
                                Eigen::Matrix<double, 2, 3> meilleure_Hj;
                                Eigen::Matrix<double, 2, 2> meilleure_S_inverse;

                                for (unsigned long int b_index=0; b_index<balises.size(); b_index++){
                                        float deltaX = balises[b_index][0] - state_estimation[0];
                                        float deltaY = balises[b_index][1] - state_estimation[1];
                                        float q = deltaX*deltaX + deltaY*deltaY;
                                        if (q < 1e-6) continue;

                                        float theta = state_estimation[2];

                                        Eigen::Matrix<double, 2, 1> hj;
                                        hj(0, 0) = sqrt(q);
                                        hj(1, 0) = atan2(deltaY, deltaX) - theta;

                                        Eigen::Matrix<double, 2, 3> Hj;
                                        Hj(0, 0) = -1/sqrt(q) * deltaX;
                                        Hj(0, 1) = -1/sqrt(q) * deltaY;
                                        Hj(0, 2) = 0;
                                        Hj(1, 0) = 1/q * deltaY;
                                        Hj(1, 1) = -1/q * deltaX;
                                        Hj(1, 2) = -1;

                                        Eigen::Matrix<double, 2, 1> innovation;
                                        innovation(0, 0) = cluster_centers[c_index][0] - hj(0, 0);
                                        innovation(1, 0) = wrap(cluster_centers[c_index][1] - hj(1, 0));

                                        Eigen::Matrix<double, 2, 2> S = Hj*state_covariance_estimation*Hj.transpose() + Rm_;
                                        float deltaS = S(0, 0)*S(1, 1) - S(0, 1)*S(1, 0);
                                        if (deltaS <= 0) continue;

                                        Eigen::Matrix<double, 2, 2> S_inverse;
                                        S_inverse(0, 0) =  S(1, 1)/deltaS;
                                        S_inverse(0, 1) = -S(0, 1)/deltaS;
                                        S_inverse(1, 0) = -S(1, 0)/deltaS;
                                        S_inverse(1, 1) =  S(0, 0)/deltaS;

                                        float d_carre = (innovation.transpose() * S_inverse * innovation)(0, 0);

                                        if (d_carre < meilleur_d_carre){
                                                meilleur_d_carre = d_carre;
                                                meilleure_innovation = innovation;
                                                meilleure_Hj = Hj;
                                                meilleure_S_inverse = S_inverse;
                                        }
                                }

                                if (meilleur_d_carre >= gamma_porte) continue;

                                Eigen::Matrix<double, 3, 2> gain_kalman = state_covariance_estimation * meilleure_Hj.transpose() * meilleure_S_inverse;

                                state_estimation = state_estimation + gain_kalman*meilleure_innovation;
                                state_estimation[2] = wrap(state_estimation[2]);

                                Eigen::Matrix3d I_KH = Eigen::Matrix3d::Identity() - gain_kalman*meilleure_Hj;
                                state_covariance_estimation = I_KH*state_covariance_estimation*I_KH.transpose() + gain_kalman*Rm_*gain_kalman.transpose();
                                state_covariance_estimation = 0.5 * (state_covariance_estimation + state_covariance_estimation.transpose());
                        }

                        states_history_[k].state_estimation = state_estimation;
                        states_history_[k].state_covariance_estimation = state_covariance_estimation;
                        for (size_t i = k + 1; i < states_history_.size(); ++i) {
                                prediction(state_estimation, state_covariance_estimation, states_history_[i].deplacement_robot, states_history_[i].covariance_deplacement);
                                states_history_[i].state_estimation = state_estimation;
                                states_history_[i].state_covariance_estimation = state_covariance_estimation;
                        }

                        this->state_estimation = state_estimation;
                        this->state_covariance_estimation = state_covariance_estimation;

                        publish_estimate_position(scan_msg->header.stamp);
                }

                std::vector<float> gauss_newton(std::vector<float> c, std::vector<std::vector<float>> pts){

                        for (int iter=0; iter < 20; iter++){
                                float a = 0;   // somme jx*jx
                                float b = 0;   // somme jx*jy
                                float d = 0;   // somme jy*jy
                                float g0 = 0;  // somme jx*residu
                                float g1 = 0;  // somme jy*residu

                                for (unsigned long int i=0; i < pts.size(); i++){
                                        std::vector<float> point = pts[i];
                                        float distance_c_point = sqrt((point[0] - c[0])*(point[0] - c[0]) + (point[1] - c[1])*(point[1] - c[1]));

                                        if (distance_c_point < 0.000001) continue;

                                        float residu = distance_c_point - r_balise;
                                        float jx = (c[0] - point[0]) / distance_c_point;
                                        float jy = (c[1] - point[1]) / distance_c_point;

                                        a += jx*jx;
                                        b += jx*jy;
                                        d += jy*jy;
                                        g0 += jx*residu;
                                        g1 += jy*residu;
                                }

                                float det = a*d - b*b;
                                if (fabs(det) < 0.000000001) break;

                                float delta_x = -(d*g0 - b*g1) / det;
                                float delta_y = -(a*g1 - b*g0) / det;

                                c[0] += delta_x;
                                c[1] += delta_y;

                                if (sqrt(delta_x*delta_x + delta_y*delta_y) < 0.0001) break;
                        }

                        return c;
                }

                std::vector<std::vector<int>> dbScan(const float epsilon, const float minPts, const sensor_msgs::msg::LaserScan::SharedPtr scan_msg){
                        const unsigned long int nbPtns = scan_msg->ranges.size();

                        std::vector<int> labels(nbPtns, -1);

                        std::vector<std::vector<int>> clusters;

                        int cid = 0;

                        for (unsigned long int i=0; i < nbPtns; i++){
                                if (labels[i] == -1){
                                        std::vector<int> neighbors = get_neighbors(scan_msg, i, epsilon);

                                        if (neighbors.size() < minPts){
                                                labels[i] = 0;
                                        } else {
                                                cid +=1;
                                                std::vector<int> temp;
                                                temp.push_back(i);
                                                clusters.push_back(temp);
                                                labels[i] = cid;

                                                for(unsigned long int j=0; j < neighbors.size(); j++){
                                                        if (labels[neighbors[j]] == -1){
                                                                labels[neighbors[j]] = cid;
                                                                clusters[cid-1].push_back(neighbors[j]);
                                                                std::vector<int> neighbors2 = get_neighbors(scan_msg, neighbors[j], epsilon);
                                                                if (neighbors2.size() >= minPts) {
                                                                        for (size_t k = 0; k < neighbors2.size(); k++) {
                                                                                if (std::find(neighbors.begin(), neighbors.end(), neighbors2[k]) == neighbors.end()) {
                                                                                        neighbors.push_back(neighbors2[k]);
                                                                                }
                                                                        }
                                                                }
                                                        } else if (labels[neighbors[j]] == 0){
                                                                labels[neighbors[j]] = cid;
                                                                clusters[cid-1].push_back(neighbors[j]);
                                                        }
                                                }
                                        }
                                }
                        }

                        return clusters;
                }

                std::vector<int> get_neighbors(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg, int i, const float epsilon){
                        std::vector<int> neighbors;

                        const unsigned long int nbPtns = scan_msg->ranges.size();
                        for (unsigned long int j=0; j<nbPtns; j++){
                                if (distance(scan_msg, i, j) < epsilon){
                                        neighbors.push_back(j);
                                }
                        }

                        return neighbors;
                }

                float distance(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg, int i, int j){
                        int p1 = std::min(i, j);
                        int p2 = std::max(i, j);

                        float theta1 = scan_msg->angle_min + p1*scan_msg->angle_increment;
                        float theta2 = scan_msg->angle_min + p2*scan_msg->angle_increment;

                        float r1 = scan_msg->ranges[p1];
                        float r2 = scan_msg->ranges[p2];

                        float x1 = r1*cos(theta1);
                        float y1 = r1*sin(theta1);

                        float x2 = r2*cos(theta2);
                        float y2 = r2*sin(theta2);

                        return std::sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
                }


                rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr estimate_position_pub_;
                void publish_estimate_position(const builtin_interfaces::msg::Time & stamp){
                    geometry_msgs::msg::PoseStamped msg;

                    msg.header.stamp = stamp;
                    msg.header.frame_id = "map";
                    msg.pose.position.x = state_estimation[0];
                    msg.pose.position.y = state_estimation[1];
                    msg.pose.orientation.z = std::sin(0.5 * state_estimation[2]);
                    msg.pose.orientation.w = std::cos(0.5 * state_estimation[2]);

                    estimate_position_pub_->publish(msg);
                }

                rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr balise_point_pub_;
                void publish_markers(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg, std::vector<float> cluster_center, int j){
                    visualization_msgs::msg::Marker marker;
                    marker.header.frame_id = scan_msg->header.frame_id;
                    marker.header.stamp = scan_msg->header.stamp;
                    marker.ns = "balises";
                    marker.id = j;
                    marker.type = visualization_msgs::msg::Marker::CYLINDER;
                    marker.action = visualization_msgs::msg::Marker::ADD;
                    marker.pose.position.x = cluster_center[0];
                    marker.pose.position.y = cluster_center[1];
                    marker.pose.position.z = 0.0;
                    marker.pose.orientation.w = 1.0;
                    marker.scale.x = 2 * r_balise;
                    marker.scale.y = 2 * r_balise;
                    marker.scale.z = 1.0;
                    marker.color.r = 1.0;
                    marker.color.g = 0.0;
                    marker.color.b = 0.0;
                    marker.color.a = 1.0;
                    marker.lifetime = rclcpp::Duration::from_seconds(0.2);
                    balise_point_pub_->publish(marker);
                }

};


int main(int argc, char * argv[]){
        rclcpp::init(argc, argv);
        rclcpp::spin(std::make_shared<EKFSolo>());
        rclcpp::shutdown();
        return 0;
}