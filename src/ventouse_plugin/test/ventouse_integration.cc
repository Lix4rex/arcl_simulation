#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <gz/msgs/double.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/sim/Server.hh>
#include <gz/sim/ServerConfig.hh>
#include <gz/sim/TestFixture.hh>
#include <gz/sim/components/DetachableJoint.hh>
#include <gz/transport/Node.hh>

using namespace std::chrono_literals;

class VentouseIntegration : public testing::Test
{
protected:
  void Start(bool contact, bool lift = true)
  {
    static int sequence = 0;
    base = "/ventouse_test_" + std::to_string(++sequence);
    gz::sim::ServerConfig config;
    const std::string object = contact ? R"(
      <model name='objet'><pose>0.199 0 0 0 0 0</pose><link name='link'>
        <inertial><mass>1</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial>
        <collision name='c'><geometry><box><size>0.2 0.2 0.2</size></box></geometry></collision>
      </link></model>)" : "";
    ASSERT_TRUE(config.SetSdfString(
      "<sdf version='1.9'><world name='test'>"
      "<gravity>-1 0 0</gravity>"
      "<physics name='p' type='ignored'><max_step_size>0.001</max_step_size></physics>"
      "<plugin filename='gz-sim-physics-system' name='gz::sim::systems::Physics'/>"
      "<model name='robot'><static>true</static><link name='chassis'>"
      "<collision name='bouche'><geometry><box><size>0.2 0.2 0.2</size></box></geometry></collision>"
      "</link><plugin filename='Ventouse' name='ventouse::Ventouse'>"
      "<link>chassis</link><collision>bouche</collision>"
      "<topic_commande>" + base + "/commande</topic_commande>"
      "<topic_resultat>" + base + "/resultat</topic_resultat>"
      "<topic_etat>" + base + "/etat</topic_etat>" +
      (lift ? "<topic_ascenseur>" + base + "/lift</topic_ascenseur>" : "") +
      "<hauteur>0.05</hauteur><delai_lacher>0.7</delai_lacher>"
      "</plugin></model>" + object + "</world></sdf>"));
    fixture = std::make_unique<gz::sim::TestFixture>(config);
    fixture->OnPostUpdate([this](const gz::sim::UpdateInfo &info,
                                const gz::sim::EntityComponentManager &ecm) {
      simTime = std::chrono::duration<double>(info.simTime).count();
      joints = 0;
      ecm.Each<gz::sim::components::DetachableJoint>(
          [this](const gz::sim::Entity &, const gz::sim::components::DetachableJoint *) {
            ++joints;
            return true;
          });
    }).Finalize();
    fixture->Server()->SetUpdatePeriod(0ms);
    ASSERT_TRUE(node.Subscribe(base + "/resultat", &VentouseIntegration::Result, this));
    ASSERT_TRUE(node.Subscribe(base + "/lift", &VentouseIntegration::Lift, this));
    command = node.Advertise<gz::msgs::StringMsg>(base + "/commande");
    ASSERT_TRUE(Wait([this]() { return command.HasConnections(); }));
    Step();  // physique et contacts initiaux
  }

  void TearDown() override
  {
    node.Unsubscribe(base + "/resultat");
    node.Unsubscribe(base + "/lift");
    fixture.reset();
  }

  void Result(const gz::msgs::StringMsg &msg)
  {
    std::lock_guard<std::mutex> lock(mutex);
    results.push_back(msg.data());
  }

  void Lift(const gz::msgs::Double &msg)
  {
    std::lock_guard<std::mutex> lock(mutex);
    lifts.push_back(msg.data());
  }

  void Step(bool paused = false)
  {
    ASSERT_TRUE(fixture->Server()->RunOnce(paused));
  }

  bool Wait(const std::function<bool()> &predicate, bool paused = false)
  {
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    while (!predicate() && std::chrono::steady_clock::now() < deadline)
    {
      Step(paused);
      std::this_thread::sleep_for(1ms);
    }
    return predicate();
  }

  void Send(const std::string &id, const std::string &action, int timeoutMs = 2000)
  {
    const auto expiry = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() + timeoutMs;
    gz::msgs::StringMsg msg;
    msg.set_data("1 " + id + " " + action + " " + std::to_string(expiry));
    ASSERT_TRUE(command.Publish(msg));
  }

  bool HasResult(const std::string &id, const std::string &statusCode)
  {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto &result : results)
      if (result == "1 " + id + " " + statusCode)
        return true;
    return false;
  }

  bool HasLift(double position)
  {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto value : lifts)
      if (value == position)
        return true;
    return false;
  }

  gz::transport::Node node;
  gz::transport::Node::Publisher command;
  std::unique_ptr<gz::sim::TestFixture> fixture;
  std::string base;
  std::mutex mutex;
  std::vector<std::string> results;
  std::vector<double> lifts;
  int joints{0};
  double simTime{0};
};

TEST_F(VentouseIntegration, ProbeWhilePausedDoesNotAttachOrReserveOperation)
{
  Start(true);
  Send("probe", "PING");
  ASSERT_TRUE(Wait([this]() { return HasResult("probe", "OK READY"); }, true));
  EXPECT_EQ(joints, 0);
  EXPECT_FALSE(HasLift(0.05));
  Send("take", "TAKE");
  ASSERT_TRUE(Wait([this]() { return HasResult("take", "OK TAKEN"); }));
  EXPECT_EQ(joints, 1);
}

TEST_F(VentouseIntegration, ContactAttachesAndReleaseWaitsBeforeRemovingJoint)
{
  Start(true);
  Send("take", "TAKE");
  ASSERT_TRUE(Wait([this]() { return HasResult("take", "OK TAKEN"); }));
  EXPECT_EQ(joints, 1);
  ASSERT_TRUE(Wait([this]() { return HasLift(0.05); }));
  Send("again", "TAKE");
  ASSERT_TRUE(Wait([this]() { return HasResult("again", "OK ALREADY_HELD"); }));
  EXPECT_EQ(joints, 1);

  Send("release", "RELEASE");
  ASSERT_TRUE(Wait([this]() { return HasLift(0.0); }));
  const double started = simTime;
  for (int i = 0; i < 300; ++i) Step();
  EXPECT_EQ(joints, 1);
  EXPECT_FALSE(HasResult("release", "OK RELEASED"));
  ASSERT_TRUE(Wait([this]() { return HasResult("release", "OK RELEASED"); }));
  EXPECT_GE(simTime - started, 0.69);
  // RequestRemoveEntity est appliquée par Gazebo en fin de pas, après PostUpdate.
  ASSERT_TRUE(Wait([this]() { return joints == 0; }));
  EXPECT_EQ(joints, 0);
  Send("free", "RELEASE");
  ASSERT_TRUE(Wait([this]() { return HasResult("free", "OK ALREADY_FREE"); }));
}

TEST_F(VentouseIntegration, PauseExpiresTakeAndRejectsConcurrentCommand)
{
  Start(false);
  Send("take", "TAKE", 150);
  // La commande est réservée avant son expiration, même si aucun objet ne touche.
  for (int i = 0; i < 10; ++i) { Step(); std::this_thread::sleep_for(1ms); }
  Send("busy", "RELEASE");
  ASSERT_TRUE(Wait([this]() { return HasResult("busy", "ERROR BUSY"); }, true));
  ASSERT_TRUE(Wait([this]() { return HasResult("take", "ERROR TIMEOUT"); }, true));
  EXPECT_EQ(joints, 0);
  Send("free", "RELEASE");
  ASSERT_TRUE(Wait([this]() { return HasResult("free", "OK ALREADY_FREE"); }, true));
}

TEST_F(VentouseIntegration, PauseDuringReleaseKeepsJointAndRestoresLiftCommand)
{
  Start(true);
  Send("take", "TAKE");
  ASSERT_TRUE(Wait([this]() { return HasResult("take", "OK TAKEN"); }));
  {
    std::lock_guard<std::mutex> lock(mutex);
    lifts.clear();
  }
  Send("release", "RELEASE", 150);
  ASSERT_TRUE(Wait([this]() { return HasLift(0.0); }));
  ASSERT_TRUE(Wait([this]() { return HasResult("release", "ERROR TIMEOUT"); }, true));
  ASSERT_TRUE(Wait([this]() { return HasLift(0.05); }, true));
  EXPECT_EQ(joints, 1);
  Send("held", "TAKE");
  ASSERT_TRUE(Wait([this]() { return HasResult("held", "OK ALREADY_HELD"); }, true));
}

TEST_F(VentouseIntegration, ReleaseWithoutLiftAndExpiredCommands)
{
  Start(true, false);
  Send("expired", "TAKE", -100);
  ASSERT_TRUE(Wait([this]() { return HasResult("expired", "ERROR TIMEOUT"); }));
  EXPECT_EQ(joints, 0);
  Send("invalid", "UNKNOWN");
  ASSERT_TRUE(Wait([this]() { return HasResult("invalid", "ERROR INVALID_COMMAND"); }));
  Send("take", "TAKE");
  ASSERT_TRUE(Wait([this]() { return HasResult("take", "OK TAKEN"); }));
  Send("release", "RELEASE");
  const double started = simTime;
  ASSERT_TRUE(Wait([this]() { return HasResult("release", "OK RELEASED"); }));
  EXPECT_LT(simTime - started, 0.1);
  ASSERT_TRUE(Wait([this]() { return joints == 0; }));
  EXPECT_EQ(joints, 0);
}
