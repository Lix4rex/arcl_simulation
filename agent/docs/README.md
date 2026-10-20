# Utiliser et entraîner l’agent

Toutes les commandes se lancent depuis la racine du dépôt. Les contrats,
le déroulement d’un épisode et les détails PPO sont dans [doc.md](doc.md).

## Installation

```sh
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install -r agent/requirements.txt
```

Pour la simulation ROS/Gazebo : compiler le workspace avec `./build.sh`, puis
sourcer `install/setup.bash`. Le backend RL et les fonctions de récompense
restent à raccorder/définir ; leurs interfaces sont décrites dans `doc.md`.

## Essayer un snapshot sans commander ROS

```sh
python3 -c 'import json; print(json.dumps(json.load(open("agent/docs/example_world.json"))))' \
  | python3 -m agent.architecture.run_agent --dry-run \
      --model checkpoints/policy.pt --records runs/replay
```

Le modèle est créé s’il manque. Les décisions et l’observation correspondante
sont enregistrées dans `runs/replay/<episode_id>.jsonl`. Un fichier d’entrée
sans fin de partie produit un épisode incomplet, exclu de l’entraînement.

Pour rejouer un flux NDJSON déjà disponible :

```sh
python3 -m agent.architecture.run_agent --dry-run \
  --model checkpoints/policy.pt --records runs/replay < snapshots.jsonl
```

## Utiliser l’agent avec ROS/Gazebo

```sh
source install/setup.bash
python3 -m agent.architecture.world_observer --carton-source gazebo_sim \
  | python3 -m agent.architecture.run_agent \
      --model checkpoints/policy.pt --records runs/live
```

Cette commande exécute réellement les actions ROS. `--ros-setup chemin/setup.bash`
permet de choisir un autre overlay. Une pose absente impose l'attente ;
une charge inconnue ne bloque pas les actions.
L'observateur lance son propre bridge Gazebo vers ROS pour
`/robot_omni/ventouse/etat` et renseigne `robot.suction_state`.
Le plugin publie uniquement aux changements d'état : avant le premier message,
la valeur reste `None`, puis la dernière valeur est conservée. Si le bridge
s'arrête ou si le message est invalide, elle redevient `None`. Ce flux ne fournit
ni heartbeat ni détection d'un redémarrage du simulateur ; l'âge du dernier
événement figure dans `sources.suction.age_s`.
La charge, l'objet tenu et les contacts restent `None`.
Les zones et obstacles fixes se renseignent dans `agent/architecture/arena_map.json`.

Le runner s’arrête sur le signal de fin de match ou un temps restant nul.
Les journaux live conservent les décisions et le résultat connu, mais ne sont
pas des données PPO récompensées. Pour l’entraînement RL, utiliser la collecte
ci-dessous. `--sample-actions` active l’échantillonnage dans le runner si nécessaire.

## Collecter des épisodes RL

Remplacer `mon_backend:create_backend` par la fabrique de votre adaptateur
simulateur. Elle doit fournir `reset()` et `advance()` selon `doc.md`.

Pour vérifier le branchement avant de définir les fonctions de points :

```sh
python3 -m agent.rl.collect --backend mon_backend:create_backend \
  --robot-id bleu --opponent-id rouge --episodes 10 --record-only \
  --model checkpoints/policy.pt --records runs/diagnostic
```

Ces fichiers sont explicitement marqués sans récompenses et ne sont pas utilisés
pour apprendre. Une fois `assign_action_points` et `assign_winner_points`
définies dans `agent/rl/environment.py`, collecter les données d’entraînement :

```sh
python3 -m agent.rl.collect --backend mon_backend:create_backend \
  --robot-id bleu --opponent-id rouge --episodes 20 \
  --model checkpoints/policy.pt --records runs/batch-001
```

Les deux politiques échantillonnent actions et cibles. Par défaut, elles utilisent
le même modèle ; `--opponent-model chemin.pt` choisit une autre politique adverse.
Le modèle de collecte doit être conservé pour l’étape suivante.

## Entraîner puis charger le modèle retenu

Au moins deux épisodes complets et récompensés sont nécessaires pour la
séparation automatique train/validation :

```sh
python3 -m agent.architecture.train_policy runs/batch-001 \
  --model checkpoints/policy.pt --output checkpoints/policy-next.pt \
  --validation-fraction 0.2 --epochs 4 --seed 7
```

Le meilleur checkpoint selon la validation est conservé, ainsi que
`checkpoints/policy-next.metrics.json` avec les partitions et métriques.
Les épisodes incomplets, sans récompenses ou sans décisions échantillonnées
sont exclus. Pour apprendre uniquement d’un robot, ajouter `--robot-id bleu`.

Avec des partitions préparées séparément :

```sh
python3 -m agent.architecture.train_policy runs/train \
  --validation-data runs/validation --test-data runs/test \
  --model checkpoints/policy.pt --output checkpoints/policy-next.pt
```

Le jeu de test est facultatif et évalué uniquement après sélection du checkpoint.
Réutiliser `checkpoints/policy-next.pt` avec `run_agent --model` pour l’inférence,
ou avec `collect --model` pour une nouvelle collecte. Pour un nouveau cycle PPO,
collecter un nouveau lot avec ce modèle plutôt que réentraîner continuellement
sur les mêmes anciennes parties. Les anciennes lignes supervisées
`{world, decision}` ne sont pas le format de cet entraînement RL.

## Vérifier le flux sans ROS

```sh
python3 -m unittest discover -s agent/tests -v
```

Les tests utilisent un backend fictif et des récompenses locales au test ; ils
ne modifient pas les fonctions de points de production et ne pilotent aucun robot.
