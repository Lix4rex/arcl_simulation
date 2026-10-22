# Utiliser et entraîner l’agent

Toutes les commandes se lancent depuis la racine du dépôt. Les contrats,
le déroulement d’un épisode et les détails PPO sont dans [doc.md](doc.md).

## Installation

```sh
python3 -m pip install --user --break-system-packages 'numpy>=1.24'
python3 -m pip uninstall --break-system-packages -y torch
python3 -m pip install --user --break-system-packages 'torch==2.14.1+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

Sur Ubuntu 24.04, `--break-system-packages` est requis par pip pour le Python
géré par la distribution ; `--user` installe les paquets dans `~/.local`, sans
`sudo` et sans environnement virtuel. Utiliser le même `python3` pour lancer
l’observer et l’agent. L’index PyTorch sélectionné fournit le wheel CPU, sans
runtime CUDA ; `torch.version.cuda` doit afficher `None`.

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

Cette commande publie les déplacements `PoseStamped` sur `/goal_pose`, et utilise
les trois services de ventouse et celui du poignet. Le topic de déplacement ne
fournit pas d’accusé de fin : l’exécuteur confirme par pose et vitesse fraîches,
sinon il garde le verrou `unknown`. Une pose ou un angle de poignet inconnu
masque les ordres dépendants. Les états de ventouse restent inconnus si leurs
topics ne sont pas reliés à ROS.
Les zones et obstacles fixes se renseignent dans `agent/architecture/arena_map.json`.
La cadence par défaut est 10 Hz. `--robot-margin` (0,20 m) réserve une marge
sur les limites de carte ; `--velocity-scale` configure l’échelle de
normalisation des vitesses lors de la création d’un checkpoint.

Le runner s’arrête sur le signal de fin de match ou un temps restant nul.
Les journaux live conservent les décisions et le résultat connu, mais ne sont
pas des données PPO récompensées. Pour l’entraînement RL, utiliser la collecte
ci-dessous. `--sample-actions` active l’échantillonnage pendant la collecte live.

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
