// Ventouses génériques pour Gazebo Harmonic (gz-sim 8).
// Sur commande, colle au robot chaque objet (non statique) qui touche une des ventouses,
// puis lève l'ascenseur. Au lâcher : l'ascenseur redescend, puis les objets sont relâchés.
//
//   attraper : gz topic -t /robot_omni/ventouse -m gz.msgs.Boolean -p 'data: true'
//   lâcher   : gz topic -t /robot_omni/ventouse -m gz.msgs.Boolean -p 'data: false'
//
// Paramètres SDF (dans le <model> du robot) :
//   <link>      link qui porte les ventouses               (défaut : chassis)
//   <collision> collision d'une bouche de ventouse, répétable (au moins une)
//   <topic>     topic de commande                          (défaut : /model/<robot>/ventouse)
//   <topic_ascenseur> topic gz.msgs.Double du JointPositionController de l'ascenseur (optionnel)
//   <hauteur>   hauteur de levée en m                      (défaut : 0.05)
//   <delai_lacher> temps de descente avant de relâcher, s  (défaut : 0.7)
//
// "attraper" met l'aspiration en marche : elle reste active jusqu'à ce qu'un objet touche
// une ventouse (il est alors collé et levé) ou jusqu'à "lâcher".
// Les commandes sont mises en file et exécutées dans l'ordre : on peut enchaîner
// attraper / lâcher / attraper... sans attendre, rien n'est perdu.
// État publié (gz.msgs.StringMsg) sur <topic>/etat : "libre", "aspiration", "tenu" ou "descente".

#include <deque>
#include <mutex>
#include <chrono>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <gz/msgs/boolean.pb.h>
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
  public: void Configure(const Entity &_entity,
                         const std::shared_ptr<const sdf::Element> &_sdf,
                         EntityComponentManager &_ecm, EventManager &) override
  {
    this->modele = _entity;
    Model model(_entity);
    const auto nomLink = _sdf->Get<std::string>("link", "chassis").first;
    const auto topic = _sdf->Get<std::string>(
        "topic", "/model/" + model.Name(_ecm) + "/ventouse").first;

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

    this->pubEtat = this->node.Advertise<msgs::StringMsg>(topic + "/etat");
    this->node.Subscribe(topic, &Ventouse::SurCommande, this);
    gzmsg << "[Ventouse] " << this->bouches.size()
          << " ventouse(s) prête(s), commande sur " << topic << "\n";
    this->ok = true;
  }

  private: enum class Etat { LIBRE, ASPIRATION, TENU, DESCENTE };

  private: void SurCommande(const msgs::Boolean &_msg)
  {
    std::lock_guard<std::mutex> l(this->mutex);
    this->file.push_back(_msg.data());
  }

  public: void PreUpdate(const UpdateInfo &_info, EntityComponentManager &_ecm) override
  {
    if (!this->ok || _info.paused)
      return;
    const double t = std::chrono::duration<double>(_info.simTime).count();

    // Descente en cours : on relâche quand elle est finie, les commandes suivantes attendent.
    if (this->etat == Etat::DESCENTE)
    {
      if (t < this->lacherA)
        return;
      for (const Entity j : this->liaisons)
        _ecm.RequestRemoveEntity(j);
      this->liaisons.clear();
      gzmsg << "[Ventouse] lâché\n";
      this->ChangerEtat(Etat::LIBRE);
      return;                 // contacts à jour au pas suivant
    }

    // Une commande par pas de simulation, dans l'ordre d'arrivée.
    bool aCmd = false, cmd = false;
    {
      std::lock_guard<std::mutex> l(this->mutex);
      if (!this->file.empty())
      {
        aCmd = true;
        cmd = this->file.front();
        this->file.pop_front();
      }
    }
    if (aCmd)
    {
      if (cmd && this->etat == Etat::LIBRE)
      {
        this->ChangerEtat(Etat::ASPIRATION);
        gzmsg << "[Ventouse] aspiration en marche, contacts actuels : "
              << this->DecrireContacts(_ecm) << "\n";
      }
      else if (!cmd && this->etat == Etat::ASPIRATION)
      {
        gzmsg << "[Ventouse] aspiration coupée (rien attrapé)\n";
        this->ChangerEtat(Etat::LIBRE);
      }
      else if (!cmd && this->etat == Etat::TENU)
      {
        this->Ascenseur(0.0);   // redescend d'abord, relâche ensuite
        this->lacherA = this->avecAscenseur ? t + this->delaiLacher : t;
        this->ChangerEtat(Etat::DESCENTE);
        return;
      }
      // sinon : attraper alors qu'on tient déjà, ou lâcher alors qu'on est libre -> rien à faire
    }

    // Aspiration active : on colle dès qu'un objet touche.
    if (this->etat == Etat::ASPIRATION && this->Attraper(_ecm))
    {
      this->Ascenseur(this->hauteur);
      this->ChangerEtat(Etat::TENU);
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
  private: transport::Node::Publisher pubAscenseur, pubEtat;
  private: bool avecAscenseur{false};
  private: double hauteur{0.05}, delaiLacher{0.7}, lacherA{-1};
  private: Entity modele{kNullEntity}, link{kNullEntity};
  private: std::vector<Entity> bouches, liaisons;
  private: std::mutex mutex;
  private: std::deque<bool> file;
  private: Etat etat{Etat::LIBRE};
  private: bool ok{false};
};

GZ_ADD_PLUGIN(Ventouse, System, ISystemConfigure, ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(Ventouse, "ventouse::Ventouse")
