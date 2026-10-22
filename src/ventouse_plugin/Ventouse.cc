// Ventouses génériques pour Gazebo Harmonic (gz-sim 8).
// Sur commande, colle au robot chaque objet (non statique) qui touche une des ventouses,
// puis lève l'ascenseur. Au lâcher : l'ascenseur redescend, puis les objets sont relâchés.
//
//   ros2 service call /robot_omni/ventouse/set std_srvs/srv/SetBool '{data: true}'
//   ros2 service call /robot_omni/ventouse/set std_srvs/srv/SetBool '{data: false}'
//
// Paramètres SDF (dans le <model> du robot) :
//   <link>      link qui porte les ventouses               (défaut : chassis)
//   <collision> collision d'une bouche de ventouse, répétable (au moins une)
//   <topic_commande> topic StringMsg des commandes identifiées
//   <topic_resultat> topic StringMsg des résultats finaux
//   <topic_ascenseur> topic gz.msgs.Double du JointPositionController de l'ascenseur (optionnel)
//   <hauteur>   hauteur de levée en m                      (défaut : 0.05)
//   <delai_lacher> temps de descente avant de relâcher, s  (défaut : 0.7)
//
// Une seule opération est acceptée à la fois, sans file d'attente.
// Commande : 1 <id> <PING|TAKE|RELEASE> <expiration_unix_ms>
// Résultat : 1 <id> <OK|ERROR> <code>
// L'échéance réelle est vérifiée même lorsque la simulation est en pause.
// État de diagnostic sur /model/<robot>/ventouse/etat (configurable).

#include <mutex>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gz/msgs/double.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>

#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/ContactSensorData.hh>
#include <gz/sim/components/DetachableJoint.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/Static.hh>

using namespace gz;
using namespace sim;

class Ventouse : public System, public ISystemConfigure, public ISystemPreUpdate
{
  public: ~Ventouse() override
  {
    if (!this->topicCommande.empty())
      this->node.Unsubscribe(this->topicCommande);
  }

  public: void Configure(const Entity &_entity,
                         const std::shared_ptr<const sdf::Element> &_sdf,
                         EntityComponentManager &_ecm, EventManager &) override
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->modele = _entity;
    Model model(_entity);
    const auto nomLink = _sdf->Get<std::string>("link", "chassis").first;
    const auto base = "/model/" + model.Name(_ecm) + "/ventouse";
    const auto topic = _sdf->Get<std::string>("topic_commande", base + "/commande").first;
    this->topicCommande = topic;
    this->pubResultat = this->node.Advertise<msgs::StringMsg>(
        _sdf->Get<std::string>("topic_resultat", base + "/resultat").first);
    this->pubEtat = this->node.Advertise<msgs::StringMsg>(
        _sdf->Get<std::string>("topic_etat", base + "/etat").first);
    // Même un plugin mal configuré doit pouvoir répondre NOT_READY.
    this->node.Subscribe(topic, &Ventouse::SurCommande, this);

    this->link = model.LinkByName(_ecm, nomLink);
    if (this->link == kNullEntity)
    {
      gzerr << "[Ventouse] link [" << nomLink << "] introuvable\n";
      return;
    }

    for (auto e = _sdf->FindElement("collision"); e; e = e->GetNextElement("collision"))
    {
      const auto nomCol = e->Get<std::string>();
      const Entity col = Link(this->link).CollisionByName(_ecm, nomCol);
      if (col == kNullEntity)
      {
        gzerr << "[Ventouse] collision [" << nomCol << "] introuvable\n";
        continue;
      }
      // Demande au moteur physique de remplir les contacts de cette collision.
      if (!_ecm.Component<components::ContactSensorData>(col))
        _ecm.CreateComponent(col, components::ContactSensorData());
      this->bouches.push_back(col);
    }
    if (this->bouches.empty())
    {
      gzerr << "[Ventouse] aucune <collision> valide\n";
      return;
    }

    this->hauteur = _sdf->Get<double>("hauteur", 0.05).first;
    this->delaiLacher = _sdf->Get<double>("delai_lacher", 0.7).first;
    if (_sdf->HasElement("topic_ascenseur"))
    {
      this->pubAscenseur = this->node.Advertise<msgs::Double>(
          _sdf->Get<std::string>("topic_ascenseur"));
      this->avecAscenseur = true;
    }

    gzmsg << "[Ventouse] " << this->bouches.size()
          << " ventouse(s) prête(s), commande sur " << topic << "\n";
    this->ok = true;
  }

  private: enum class Etat { LIBRE, ASPIRATION, TENU, DESCENTE };

  private: struct Operation
  {
    std::string id;
    bool attraper;
    std::chrono::steady_clock::time_point echeance;
    bool commencee{false};
  };

  private: void PublierResultat(const std::string &_id, bool _success,
                               const std::string &_code)
  {
    msgs::StringMsg msg;
    msg.set_data("1 " + _id + (_success ? " OK " : " ERROR ") + _code);
    this->pubResultat.Publish(msg);
  }

  // Appelée uniquement avec le mutex détenu, après mise à jour de l'état physique.
  private: void TerminerOperation(bool _success, const std::string &_code)
  {
    const auto id = this->operation->id;
    this->operation.reset();
    this->PublierResultat(id, _success, _code);
  }

  private: void SurCommande(const msgs::StringMsg &_msg)
  {
    std::lock_guard<std::mutex> l(this->mutex);
    std::istringstream input(_msg.data());
    std::string version, id, commande, extra;
    int64_t expiration;
    if (!(input >> version >> id))
    {
      gzerr << "[Ventouse] commande sans identifiant\n";
      return;
    }
    if (version != "1" || !(input >> commande >> expiration) || (input >> extra) ||
        (commande != "PING" && commande != "TAKE" && commande != "RELEASE"))
    {
      this->PublierResultat(id, false, "INVALID_COMMAND");
      return;
    }
    if (!this->ok)
    {
      this->PublierResultat(id, false, "NOT_READY");
      return;
    }
    if (commande == "PING")
    {
      this->PublierResultat(id, true, "READY");
      return;
    }
    if (this->operation)
    {
      this->PublierResultat(id, false, "BUSY");
      return;
    }
    const auto maintenant = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (expiration <= maintenant)
    {
      this->PublierResultat(id, false, "TIMEOUT");
      return;
    }
    // Borne le protocole et évite tout débordement lors de la conversion d'horloge.
    if (expiration > maintenant + 3600000)
    {
      this->PublierResultat(id, false, "INVALID_COMMAND");
      return;
    }
    this->operation = Operation{id, commande == "TAKE",
        std::chrono::steady_clock::now() + std::chrono::milliseconds(expiration - maintenant)};
  }

  public: void PreUpdate(const UpdateInfo &_info, EntityComponentManager &_ecm) override
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    if (!this->ok || !this->operation)
      return;
    if (std::chrono::steady_clock::now() >= this->operation->echeance)
    {
      if (this->etat == Etat::ASPIRATION)
        this->ChangerEtat(Etat::LIBRE);
      else if (this->etat == Etat::DESCENTE)
      {
        this->Ascenseur(this->hauteur);
        this->ChangerEtat(Etat::TENU);  // liaisons conservées
      }
      this->TerminerOperation(false, "TIMEOUT");
      return;
    }
    // Les commandes idempotentes ne nécessitent aucun pas de physique.
    if (!this->operation->commencee)
    {
      if (this->operation->attraper && this->etat == Etat::TENU)
      {
        this->TerminerOperation(true, "ALREADY_HELD");
        return;
      }
      if (!this->operation->attraper && this->etat == Etat::LIBRE)
      {
        this->TerminerOperation(true, "ALREADY_FREE");
        return;
      }
    }
    if (_info.paused)
      return;
    const double t = std::chrono::duration<double>(_info.simTime).count();

    if (!this->operation->commencee)
    {
      this->operation->commencee = true;
      if (this->operation->attraper)
      {
        this->ChangerEtat(Etat::ASPIRATION);
        gzmsg << "[Ventouse] aspiration, contacts : " << this->DecrireContacts(_ecm) << "\n";
      }
      else
      {
        this->Ascenseur(0.0);
        this->lacherA = this->avecAscenseur ? t + this->delaiLacher : t;
        this->ChangerEtat(Etat::DESCENTE);
      }
    }

    if (this->etat == Etat::DESCENTE)
    {
      if (t < this->lacherA)
        return;
      for (const Entity j : this->liaisons)
        _ecm.RequestRemoveEntity(j);
      this->liaisons.clear();
      gzmsg << "[Ventouse] lâché\n";
      this->ChangerEtat(Etat::LIBRE);
      this->TerminerOperation(true, "RELEASED");
      return;                 // contacts à jour au pas suivant
    }

    // Aspiration active : on colle dès qu'un objet touche.
    if (this->etat == Etat::ASPIRATION && this->Attraper(_ecm))
    {
      this->Ascenseur(this->hauteur);
      this->ChangerEtat(Etat::TENU);
      this->TerminerOperation(true, "TAKEN");
    }
  }

  // Diagnostic : liste de ce que touchent les ventouses.
  private: std::string DecrireContacts(EntityComponentManager &_ecm)
  {
    std::set<std::string> noms;
    for (const Entity bouche : this->bouches)
    {
      auto contacts = _ecm.Component<components::ContactSensorData>(bouche);
      if (!contacts)
        continue;
      for (const auto &c : contacts->Data().contact())
      {
        const Entity autre = c.collision1().id() == bouche ?
            c.collision2().id() : c.collision1().id();
        auto nom = _ecm.Component<components::Name>(topLevelModel(autre, _ecm));
        noms.insert(nom ? nom->Data() : "?");
      }
    }
    if (noms.empty())
      return "aucun";
    std::string r;
    for (const auto &n : noms)
      r += (r.empty() ? "" : ", ") + n;
    return r;
  }

  private: void ChangerEtat(Etat _e)
  {
    this->etat = _e;
    msgs::StringMsg m;
    m.set_data(_e == Etat::LIBRE ? "libre" : _e == Etat::ASPIRATION ? "aspiration" :
               _e == Etat::TENU ? "tenu" : "descente");
    this->pubEtat.Publish(m);
  }

  private: void Ascenseur(double _z)
  {
    if (!this->avecAscenseur)
      return;
    msgs::Double m;
    m.set_data(_z);
    this->pubAscenseur.Publish(m);
  }

  private: bool Attraper(EntityComponentManager &_ecm)
  {
    std::set<Entity> touches;   // links touchés, chacun attaché une seule fois
    for (const Entity bouche : this->bouches)
    {
      auto contacts = _ecm.Component<components::ContactSensorData>(bouche);
      if (!contacts)
        continue;
      for (const auto &c : contacts->Data().contact())
      {
        const Entity autre = c.collision1().id() == bouche ?
            c.collision2().id() : c.collision1().id();
        auto parent = _ecm.Component<components::ParentEntity>(autre);
        if (!parent)
          continue;
        const Entity linkAutre = parent->Data();
        const Entity modeleAutre = topLevelModel(linkAutre, _ecm);
        if (modeleAutre == this->modele)
          continue;                       // soi-même
        auto stat = _ecm.Component<components::Static>(modeleAutre);
        if (stat && stat->Data())
          continue;                       // mur, balise...
        touches.insert(linkAutre);
      }
    }

    for (const Entity l : touches)
    {
      const Entity j = _ecm.CreateEntity();
      _ecm.CreateComponent(j, components::DetachableJoint({this->link, l, "fixed"}));
      this->liaisons.push_back(j);
      auto nom = _ecm.Component<components::Name>(topLevelModel(l, _ecm));
      gzmsg << "[Ventouse] attrapé " << (nom ? nom->Data() : "?") << "\n";
    }
    return !touches.empty();
  }

  private: transport::Node node;
  private: std::string topicCommande;
  private: transport::Node::Publisher pubAscenseur, pubEtat, pubResultat;
  private: bool avecAscenseur{false};
  private: double hauteur{0.05}, delaiLacher{0.7}, lacherA{-1};
  private: Entity modele{kNullEntity}, link{kNullEntity};
  private: std::vector<Entity> bouches, liaisons;
  private: std::mutex mutex;
  private: std::optional<Operation> operation;
  private: Etat etat{Etat::LIBRE};
  private: bool ok{false};
};

GZ_ADD_PLUGIN(Ventouse, System, ISystemConfigure, ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(Ventouse, "ventouse::Ventouse")
