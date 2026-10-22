# Agent : contrats techniques et flux d’entraînement

## Architecture et provenance

| Module | Responsabilité |
|---|---|
| `world_observer.py` | Assemble les observations ROS en snapshots NDJSON. |
| `gazebo_pose_source.py` | Fournit uniquement les poses des cartons du simulateur, comme substitut explicite de la caméra externe. |
| `world_encoder.py` | Construit les caractéristiques normalisées puis les embeddings des familles d’entités. |
| `policy.py` | Politique masquée, sélection de cible et estimation de valeur. |
| `action_executor.py` / `run_agent.py` | Exécution ROS asynchrone et journalisation de l’utilisation live. |
| `rl/environment.py` / `rl/collect.py` | Match à deux robots, collecte stochastique, résultats et récompenses. |
| `records.py` / `train_policy.py` | Finalisation des fichiers puis mises à jour PPO et validation. |

Les cartons viennent de la caméra externe, ici remplacée explicitement par les
poses des seuls modèles Gazebo `pierre_*`. La pose de notre robot vient de
`/pose_estimation`, les obstacles de `/obstacles_scan` et les tags de
`/aruco_markers`. Aucun repli vers la pose vraie du robot n’est autorisé.
Le backend RL fournit séparément la vérité `world` et les `observations` de
chaque robot. La vérité sert aux points et aux journaux, jamais à reconstruire
une observation manquante. Pour une politique, l’adversaire est un obstacle
ponctuel ; son cap, sa charge et sa ventouse sont exclus.

Les mesures périmées restent inconnues. Les ArUco sont transformés depuis
`camera_optical` avec les extrinsèques de l’URDF ; une association ambiguë ne
fabrique pas d’identité de carton. Le décalage temporel pose/image est contrôlé.
L’assembleur ne déduit pas une charge vide de l’absence de mesure.

L'observateur ouvre un bridge Gazebo vers ROS uniquement pour
`/robot_omni/ventouse/etat` (`gz.msgs.StringMsg` vers `std_msgs/msg/String`).
Les transitions `libre`, `aspiration`, `tenu`, `descente` alimentent respectivement
`suction_state`: `free`, `aspirating`, `holding`, `lowering`. Ce flux publie aux
changements, pas périodiquement : aucun état initial n'est supposé et le dernier
événement est conservé, avec son âge dans `sources.suction`. L'arrêt du bridge
ou un message invalide invalide cet état. Une coupure Gazebo silencieuse ou un
redémarrage du simulateur n'est pas détectable par ce seul flux.
`payload_state`, `held_object` et les contacts ne sont pas déduits de cet état.

## Contrat des snapshots

Toutes les positions et dimensions sont en mètres dans le même repère fixe de
l'arène (`map`), les angles `yaw` sont en radians. `sequence` augmente à chaque
snapshot et `timestamp` est le temps de référence en secondes. Le producteur
perception doit transformer les sources ROS dans ce repère et fournir des pistes
stables : les IDs ArUco partagés ne sont pas des IDs de carton.

Chaque ligne est un objet avec ces champs :

- `sequence`, `timestamp`, `arena.bounds` (`x_min`, `x_max`, `y_min`, `y_max`).
- `robot.pose` (`x`, `y`, `z`, `yaw`) estimée et `pose_age_s`. Une pose périmée
  devient `null` et la politique répond `WAIT`. `suction_state`, `payload_state`,
  `held_object`, `contact_confirmed` et `contact_target_id` peuvent être `null`
  quand aucune source fiable ne les fournit.
- `suction_state` distingue `free`, `aspirating`, `holding` et `lowering`.
  Il décrit l'actionneur ; `payload_state` décrit la charge estimée. L'un n'est
  pas inventé à partir de l'autre.
- `cartons`, `obstacles` : listes d'entités avec un `id` interne
  stable, `pose`, `size`, `confidence`, `age_s`, `visible` et `accessible`
  quand disponibles. Un carton donne aussi `posture` (`vertical` ou `horizontal`), `marker_id` et
  `approach_pose` ; `camera_distance` (ou `distance`) peut décrire la portée
  mesurée d'un tag visible. Le centrage ROS ne dépend pas de cette association
  à un carton. Ce tag reste une observation, pas l'identité du carton.
- `other_robots` reste accepté comme entrée : seule sa position `x,y` est
  transférée aux obstacles de la politique, sans cap, charge ou état de ventouse.
- `aruco_detections` : détections locales transformées en `map`, avec leur âge,
  normale de face `normal` (`x,y,z`), profondeur optique `camera_depth`,
  portée et carton associé si l'association est unique. Les détections ambiguës
  restent présentes avec `associated_carton_id: null` et ne rendent aucun carton
  `visible`.
- `zones` : listes de rectangles avec `id`, `center` ou `pose`, `yaw`, `length`,
  `width`, `capacity`, `occupied`, `release_allowed`, `accessible` et
  `approach_pose`. `yaw` tourne le rectangle dans le repère de l'arène.
  Ces informations sont des observations pour le réseau, pas des conditions
  imposées avant de commander un dépôt.
- `mission` : `total_s` (90 par défaut), `remaining_s`, score et compteurs de cartons quand disponibles.
- `active_operation` : opération en cours si un contrôleur externe en a une,
  avec `type`, `status` (`issuing`, `running` ou `unknown`), cible éventuelle,
  `target_pose`, `elapsed_s` et `timeout_s` quand connus. L'exécuteur
  conserve lui-même l'état de ses processus.

Les tableaux doivent contenir toutes les pistes pertinentes, et non uniquement
la cible la plus proche. Une donnée inconnue reste inconnue : ne pas publier un
robot chargé comme vide ni une zone comme libre en l'absence de mesure.
`accessible` doit refléter une vraie vérification de l'approche et du trajet,
pas seulement la présence d'une pose cible.

### Encodage et normalisation

Chaque famille a son propre petit encodeur. Aucun tableau n'est tronqué ; les
vecteurs individuels sont conservés, puis leurs moyennes et leurs effectifs
alimentent le contexte global. `WorldEncoder.feature_dicts(snapshot)` expose
tous les scalaires nommés avant cette compression apprise, ainsi que la matrice
d'association ArUco/cartons. `feature_names` donne l'ordre des entrées.

| Information | Représentation dans le réseau |
|---|---|
| Positions | `x,y` dans les axes de la table, normalisés par ses bornes entre 0 et 1 ; `z` et dimensions divisés par le plus grand côté de la table. |
| Cartons | Posture explicite ; `(sin(4θ)+1)/2, (cos(4θ)+1)/2` si vertical, facteur 2 si horizontal. Les angles sont relatifs aux axes de la table. |
| Robot et approches | Cap dirigé avec facteur 1 : deux orientations opposées restent distinctes. |
| Ventouse et charge | Catégories séparées, avec un état inconnu. Le carton porté conserve position, posture, orientation et dimensions ; son identité sert à marquer la piste correspondante. |
| Actions | Type exact, statut, pose objectif, cibles marquées dans les listes, temps écoulé/délai ; dernier type, résultat et cible également conservés. |
| ArUco | Position, normale de face, profondeur optique, fraîcheur, confiance et association. Les détections non associées sont encodées aussi ; un indicateur distingue les tags ID 13 utilisables par le centrage des autres tags. |
| Mission | Temps restant/temps total, sans doublon dans le robot ; score et compteurs s'ils sont connus. |

Les sinus/cosinus et composantes de normale sont ramenés de `[-1,1]` à `[0,1]`.
Les âges sont divisés par le temps total. Effectifs, capacités et autres
quantités positives sans maximum configuré utilisent `n/(1+n)` ; le score,
éventuellement signé, utilise `0.5 + s/(2*(1+abs(s)))`. Ces transformations sont
inversibles : aucune saturation ne confond deux mesures hors des bornes
habituelles. Une position hors table ou un délai dépassé peut donc dépasser 1.

Chaque mesure facultative a un indicateur `_known` : inconnu reste distinct
de zéro ou de faux, et une confiance absente ne devient pas 1. L'état des
sources distingue une liste vide observée d'une source indisponible. Les
vitesses, le contact, les IDs numériques et les métadonnées de provenance ne
sont pas appris ; les IDs servent uniquement aux associations et aux cibles.
Les bornes réelles de l'arène sont conservées par ses quatre coins lors d'un
changement de repère, pour obtenir le même encodage depuis `map` ou la vue ego.

## Politique et exécution

La politique possède des têtes pour l’action, la cible carton, la cible zone et
la valeur `V(observation)`. En utilisation courante, elle choisit l’argmax.
En collecte, elle échantillonne une action parmi les actions autorisées, puis
une cible parmi les entités autorisées si nécessaire. La décision contient
`behavior.sampled`, `behavior.log_prob` et `behavior.value` : la log-probabilité
est celle du couple action/cible, pas seulement celle de l’action.
L’inférence et l’évaluation des actions pour PPO partagent les mêmes masques.
Une pose inconnue force `WAIT` sans inventer de probabilité de collecte.

| Action | Commande et conditions |
|---|---|
| `WAIT` | Ne commande rien. Seule possibilité pendant une opération active. |
| `GO_TO_CARTON` | Cible unique avec pose d’approche ; publication `/goal_pose`. |
| `GO_TO_ZONE` | Zone unique avec pose d’approche ; publication `/goal_pose`, même à vide. |
| `CENTER` | Goal vide `/rock_centering` ; ROS choisit le tag ID 13 de plus faible profondeur optique. Pour changer de cible, déplacer le robot. |
| `TAKE` | Service ventouse avec `true`, sans cible ni confirmation locale de contact ; la prise dépend du placement effectif. |
| `RELEASE` | Service avec `false`, sans cible ni condition de charge ou de zone. |

La charge, l'accessibilité et la compatibilité géométrique ne masquent aucune
action. Le réseau peut tenter une prise déjà chargé, un dépôt hors zone ou un
centrage sans tag ; le résultat dépend de ROS et du simulateur. Les anciens
paramètres de cible de `CENTER`, `TAKE` et `RELEASE` sont ignorés par l’exécuteur.
Une navigation exige une cible observée avec pose d'approche ; l'exécuteur
vérifie que cette pose est finie et dans les bornes de la table.

Les commandes ROS restent séparées du shell qui source le workspace. Les
processus sont asynchrones et sérialisés ; l’arrivée de navigation est vérifiée
sur une pose fraîche. Timeout ou résultat ambigu donnent `unknown` et conservent
le verrou. Le topic de navigation actuel ne permet pas d’annuler un goal distant :
fermer le client local ne garantit pas l’arrêt de ce goal. Le runner ne lance
aucune nouvelle décision à la réception d’un snapshot terminal.

## Déroulement d’un épisode RL

| Étape | Traitement |
|---|---|
| 1. Reset | Le backend donne vérité et observations distinctes ; l’environnement crée un ID de partie et garde le reset en mémoire. |
| 2. Observation | Chaque vue est canonisée dans le repère du robot ; les quatre coins de la table, les objets portés et les normales de tags suivent la transformation. Le temps total de l’épisode est ajouté. |
| 3. Décision | Les deux politiques choisissent à partir de leur propre vue. Actions, probabilités et valeurs de collecte sont conservées. |
| 4. Exécution | Les IDs choisis sont résolus dans les observations d’origine ; les poses d’approche envoyées au backend sont donc en `map`, même si la politique les a vues en ego. |
| 5. Retour | `advance` fournit nouvelle vérité, nouvelles observations et réponses d’exécution. Les points d’action sont calculés depuis ces résultats. |
| 6. Fin | Signal explicite de fin ou limite de 90 secondes par défaut ; le gagnant est déterminé depuis les scores vrais ou un `winner_id` explicite. |
| 7. Cibles | Les points terminaux sont ajoutés à la dernière transition. Les retours sont propagés en arrière et les cibles de victoire renseignées sur toutes les lignes. |
| 8. Sauvegarde | Le fichier externe est écrit seulement après cette mise à jour complète. L’entraîneur le relit sans avoir à corriger les premières lignes. |

Le backend reçoit `max_game_seconds`, le budget restant. Dépasser cette durée,
reculer dans le temps ou ne faire avancer ni l’horloge ni la fin du match est une
erreur. Les séquences périmées et les cibles de navigation ambiguës sont rejetées
avant l’envoi. Les observations ne sont jamais remplacées par la vérité.
Une erreur de step clôt l’épisode comme incomplet ; il faut un reset avant une
nouvelle tentative, car le backend a pu avancer malgré l’erreur.

### Interface du backend et récompenses

La fabrique indiquée à `collect` renvoie un objet avec `reset()` et
`advance(actions, *, max_game_seconds)`, et éventuellement `close()`.
`reset()` et `advance()` renvoient la structure suivante ; `responses` est
facultatif au reset :

```python
{
    "world": {"game_time_s": 0.0, "robots": [...], "scores": {"bleu": 0, "rouge": 0}},
    "observations": {"bleu": snapshot_bleu, "rouge": snapshot_rouge},
    "responses": {"bleu": resultat_bleu, "rouge": resultat_rouge},
}
```

Les deux IDs restent stables. Les snapshots de départ sont dans `map`.
`world.match.finished` ou `world.match_finished` peut terminer le match avant
l’échéance. Les égalités explicites utilisent `winner_id: "draw"` ou `"tie"` ;
sinon les scores finaux permettent de les reconnaître. Un résultat absent reste
inconnu, il n’est pas assimilé à une égalité.

Les fonctions volontairement non implémentées restent :

- `assign_action_points(previous_world, actions, responses, next_world, robot_ids)` ;
- `assign_winner_points(winner_id, robot_ids)`.

Elles doivent retourner une valeur finie par robot. Les actions données à la
fonction de points sont celles transmises au backend, avec leurs poses `map`.
La récompense terminale s’ajoute aux points de la dernière action. Pour un usage
Python, les fonctions peuvent aussi être injectées au constructeur de
`TwoRobotArenaEnv`. Sans elles, `rewards_pending` reste explicite et les fichiers
ne deviennent pas des exemples PPO. Le mode `collect --record-only` est prévu
pour cette situation ; aucun barème de production n’est inventé.

## Fichiers externes et finalisation des cibles

Par défaut, le runner écrit `runs/agent/<episode_id>.jsonl` et la collecte RL
`runs/episodes/<episode_id>.jsonl`. Un chemin explicite finissant par `.jsonl`
ajoute les épisodes finalisés à ce fichier. Un répertoire produit un fichier
par partie, publié par renommage atomique ; il convient donc aussi à plusieurs
processus de collecte. Pendant une partie, les lignes restent en mémoire.
Un arrêt brutal du processus avant finalisation peut perdre cet épisode.

Les lignes RL ont les types `reset`, `decision`, `transition`, `episode_end`.
Une transition conserve notamment :

- `agent_observations` : vues exactes des deux politiques avant la décision ;
- `actions` : décisions et probabilités de collecte ; `backend_actions` : poses converties pour le simulateur ;
- `world`, `next_world`, observations brutes et réponses : diagnostic séparé ;
- `reward` : points immédiats, auxquels s’ajoutent les points terminaux sur la dernière transition ;
- `return_targets` : retours actualisés par robot, calculés à la fin ;
- `winner_targets` : 1 pour le gagnant, 0 pour le perdant, 0.5 chacun pour une égalité ;
- `complete`, `winner_known`, `rewards_complete`, `episode_id`, `gamma`.

Les cibles de victoire sont propagées sur toutes les lignes, y compris le reset
et les premières décisions, avant toute écriture. Les retours sont définis par
`G[t] = reward[t] + gamma * G[t+1]`, avec zéro au-delà de la fin du match.
Le défaut est `gamma = 0.99` par décision, configurable à la collecte.
Le `winner_target` est un résultat descriptif ; il n’est pas automatiquement
utilisé comme récompense à la place des fonctions de points.

Une fermeture ou un reset prématuré finalise un journal `complete: false`, sans
cibles inventées. Une fin sans résultat connu conserve `winner_targets: null`.
Les hooks Python doivent appeler `env.close()` dans un `finally` pour conserver
les épisodes interrompus ; le collecteur CLI le fait automatiquement.

Le runner live enregistre les observations, décisions et résultats disponibles.
Il reconnaît `match.finished`, `match_finished` ou `mission.remaining_s <= 0`.
Une simple fin de fichier ne prouve pas la fin de partie. Ces journaux n’ont pas
de récompenses calculées par le backend et restent diagnostiques pour PPO.

## Entraînement par renforcement et partitions

`train_policy` relit les épisodes complets et récompensés. Il ne forme ses
entrées qu’à partir de `agent_observations`, jamais depuis `world` ni les
observations suivantes. Les transitions doivent avoir des retours finaux et
des décisions échantillonnées avec log-probabilité et valeur de collecte.
Les attentes imposées par une pose inutilisable, dépourvues de ces données,
sont conservées dans le journal mais ne constituent pas des entrées apprenables.

L’avantage est `G[t] - V_collecte[t]`. Les mises à jour PPO utilisent le rapport
entre probabilités actuelle et de collecte, avec clipping à 0.2 par défaut,
une perte de valeur et un bonus d’entropie. Les gradients sont bornés à 0.5.
Il s’agit de mises à jour sur des épisodes enregistrés, avec un petit nombre de
passes (4 par défaut). Recollecter régulièrement avec le modèle mis à jour ;
la réutilisation illimitée d’anciens lots n’est pas une boucle PPO en ligne.

La séparation train/validation porte sur les IDs de parties entières : les deux
robots et tous les instants d’une même partie restent dans la même partition.
Elle est reproductible avec une graine fixe. Au moins deux épisodes prêts sont
nécessaires pour la séparation automatique. Des jeux explicites de validation
et de test peuvent être fournis ; tout chevauchement d’IDs est refusé.
La normalisation des avantages est estimée uniquement sur le train.

La validation ne prend aucun gradient et sert à sélectionner le meilleur
checkpoint, éventuellement celui de départ si aucune époque ne l’améliore.
Le test facultatif n’est évalué qu’après cette sélection. Le fichier
`.metrics.json` conserve IDs de partitions, effectifs, épisodes exclus, métriques
train/validation, meilleure époque et résultat du test éventuel.
Ces pertes ne remplacent pas une évaluation du taux de victoire en nouveaux matchs.

## Limites volontairement conservées

La caméra externe réelle, le backend de simulation RL, les sources fiables de
charge/ventouse et la configuration des zones restent à raccorder. Le centrage
vise environ 20 cm entre caméra et tag et ne garantit pas le contact physique.
Les règles de points restent à définir. Les tests de flux utilisent un backend
fictif et des récompenses locales au test, sans lancer ROS/Gazebo. Les commandes
de lancement et d’entraînement sont regroupées dans [README.md](README.md).
