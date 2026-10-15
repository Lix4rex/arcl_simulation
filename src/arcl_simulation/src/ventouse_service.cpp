#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>

// Les callbacks sont sérialisés par le SingleThreadedExecutor et ne bloquent jamais.
class VentouseService : public rclcpp::Node
{
public:
  using Service = std_srvs::srv::SetBool;
  using Clock = std::chrono::steady_clock;

  VentouseService() : Node("ventouse_service")
  {
    const double timeout = declare_parameter("operation_timeout_sec", 2.0);
    if (!std::isfinite(timeout) || timeout <= 0.0 || timeout > 3600.0) {
      throw std::invalid_argument("operation_timeout_sec doit être dans ]0, 3600]");
    }
    timeout_ = std::chrono::milliseconds(static_cast<int64_t>(std::ceil(timeout * 1000.0)));
    // Une seconde de marge pour transporter le résultat, y compris avec un délai personnalisé.
    watchdog_ = std::max(std::chrono::milliseconds(3000), timeout_ + std::chrono::seconds(1));
    std::random_device random;
    std::ostringstream prefix;
    prefix << std::hex << random() << '-' << random() << '-'
           << Clock::now().time_since_epoch().count();
    prefix_ = prefix.str();

    command_ = create_publisher<std_msgs::msg::String>("/robot_omni/ventouse/commande", 10);
    result_ = create_subscription<std_msgs::msg::String>(
      "/robot_omni/ventouse/resultat", 10,
      [this](const std_msgs::msg::String & msg) {on_result(msg);});
    service_ = create_service<Service>(
      "/robot_omni/ventouse/set",
      [this](std::shared_ptr<rmw_request_id_t> header, Service::Request::SharedPtr request) {
        on_request(std::move(header), request->data);
      });
    timer_ = create_wall_timer(std::chrono::milliseconds(20), [this]() {
      if (pending_ && !pending_->sent && Clock::now() >= pending_->deadline) {
        const auto header = pending_->header;
        pending_.reset();
        respond(header, false, "Backend de ventouse indisponible");
        return;
      }
      if (pending_ && !pending_->sent && Clock::now() >= pending_->next_probe) {
        publish_command("PING");
        pending_->next_probe = Clock::now() + std::chrono::milliseconds(100);
      }
      if (pending_ && !pending_->answered && Clock::now() >= pending_->deadline) {
        pending_->answered = true;
        respond(pending_->header, false, "Résultat de la simulation indisponible");
        RCLCPP_WARN(get_logger(), "Résultat absent : ventouse réservée jusqu'au résultat terminal");
      }
    });
  }

private:
  struct Pending
  {
    std::string id;
    std::shared_ptr<rmw_request_id_t> header;
    Clock::time_point deadline;
    Clock::time_point next_probe;
    bool take;
    bool sent{false};
    bool answered{false};
  };

  void respond(const std::shared_ptr<rmw_request_id_t> & header, bool success,
    const std::string & message)
  {
    Service::Response response;
    response.success = success;
    response.message = message;
    service_->send_response(*header, response);
  }

  void on_request(std::shared_ptr<rmw_request_id_t> header, bool take)
  {
    if (pending_) {
      respond(header, false, "Ventouse occupée");
      return;
    }
    const auto id = prefix_ + '-' + std::to_string(++counter_);
    const auto now = Clock::now();
    pending_ = Pending{id, std::move(header), now + watchdog_,
      now + std::chrono::milliseconds(100), take};
    // Le bridge peut être découvert avant le plugin : vérifier le trajet aller-retour
    // avec une commande sans effet physique avant de réserver une opération backend.
    publish_command("PING");
  }

  void publish_command(const std::string & operation)
  {
    const auto expiration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch() + timeout_).count();
    std_msgs::msg::String msg;
    msg.data = "1 " + pending_->id + " " + operation + " " + std::to_string(expiration);
    command_->publish(msg);
  }

  void on_result(const std_msgs::msg::String & msg)
  {
    std::istringstream input(msg.data);
    std::string version, id, status, code, extra;
    if (!(input >> version >> id >> status >> code) || version != "1" || (input >> extra) ||
      !pending_ || id != pending_->id)
    {
      return;
    }
    if (status == "OK" && code == "READY") {
      if (!pending_->sent) {
        pending_->sent = true;
        pending_->deadline = Clock::now() + watchdog_;
        publish_command(pending_->take ? "TAKE" : "RELEASE");
      }
      return;
    }
    // Un résultat mal formé ne peut ni libérer la réservation ni terminer une demande.
    const bool success = status == "OK";
    std::string message;
    if (success && code == "TAKEN") {message = "Objet attaché, montée commandée";}
    else if (success && code == "RELEASED") {message = "Objet relâché";}
    else if (success && code == "ALREADY_HELD") {message = "Objet déjà tenu";}
    else if (success && code == "ALREADY_FREE") {message = "Ventouse déjà libre";}
    else if (status == "ERROR" && code == "BUSY") {message = "Ventouse occupée";}
    else if (status == "ERROR" && code == "TIMEOUT") {message = "Délai de l'opération dépassé";}
    else if (status == "ERROR" && code == "INVALID_COMMAND") {message = "Commande invalide";}
    else if (status == "ERROR" && code == "NOT_READY") {message = "Ventouse indisponible";}
    else {return;}

    const auto finished = *pending_;
    pending_.reset();
    if (!finished.answered) {
      respond(finished.header, success, message);
    }
  }

  std::chrono::milliseconds timeout_{2000}, watchdog_{3000};
  std::string prefix_;
  uint64_t counter_{0};
  std::optional<Pending> pending_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr command_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr result_;
  rclcpp::Service<Service>::SharedPtr service_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VentouseService>());
  rclcpp::shutdown();
  return 0;
}
