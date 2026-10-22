# Guide rapide de l’agent

Toutes les commandes partent de la racine du dépôt.

## Build et environnement ROS

Dans un terminal, compiler le workspace :

```sh
source /opt/ros/jazzy/setup.bash
./build.sh
source install/setup.bash
```

Dans chaque nouveau terminal ROS, recharger l’environnement :

```sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

## Capter les actions du pupitre

Terminal 1 — lancer la simulation :

```sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch arcl_simulation sim.launch.py rviz:=false
```

Terminal 2 — lancer le pupitre :

```sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 src/arcl_simulation/launch/move.launch.py
```

Cliquer sur **Enregistrer une démonstration**, effectuer les actions, puis
arrêter l’enregistrement. Les exemples observation–cible sont ajoutés à
`runs/targets.jsonl` sans remplacer les précédents. Les `IDLE` sont enregistrés
à 10 Hz pendant une opération, pas pendant les attentes ni le centrage.

## Capter les actions de la stratégie C++

Terminal 1 — lancer la simulation :

```sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch arcl_simulation sim.launch.py rviz:=false
```

Terminal 2 — démarrer l’enregistreur et attendre `first observation received` :

```sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 -m agent.architecture.record_strategy --carton-source gazebo_sim
```

Terminal 3 — démarrer la stratégie :

```sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run arcl_simulation strategy
```

Les cibles sont ajoutées à `runs/targets.jsonl` à la fin de la stratégie.
Les `IDLE` pendant les opérations sont gardés ; l’étape `IDLE` immédiate et le
centrage sont ignorés. `gazebo_sim` utilise les poses Gazebo des cartons.

## Entraîner l’agent

Dans un terminal, entraîner sur le fichier commun de cibles :

```sh
python3 -m agent.architecture.train_targets runs/targets.jsonl \
  --output checkpoints/policy.pt --epochs 20
```
Ajouter `--model checkpoints/policy.pt ` si on veut repartir d'un modèle déjà existant.

Le découpage entraînement/validation se fait par session ou épisode ; il faut
au moins deux groupes. Garde, déplace ou supprime `runs/targets.jsonl`
manuellement selon que tu souhaites sauvegarder ou repartir de zéro.

À l’entraînement, chaque action présente dans le jeu d’entraînement est
rééquilibrée au nombre d’exemples de l’action la plus fréquente en réutilisant
des exemples si nécessaire. Chaque exemple d’entraînement est ensuite reflété
une fois de l’autre côté du terrain (`x′ = x_min + x_max − x`). Les angles
deviennent `θ′ = π − θ` (sinus inchangé, cosinus inversé) et la vitesse
angulaire change de signe. Les cibles `MOVE` sont transformées aussi ;
validation et test restent inchangés.

## Lancer une partie avec l’agent

Terminal 1 — lancer la simulation :

```sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch arcl_simulation sim.launch.py rviz:=false
```

Terminal 2 — lancer l’observateur et l’agent :

```sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 -m agent.architecture.world_observer --carton-source gazebo_sim \
  | python3 -m agent.architecture.run_agent \
      --model checkpoints/policy.pt --records runs/live
```

`--sample-actions` active l’échantillonnage au lieu du choix déterministe. Les
journaux `runs/live` ne sont pas des cibles self-play : ils ne contiennent pas
les récompenses par action.

## Self-play et préparation des cibles

Il faut un backend à deux robots fournissant `reset()` et `advance()`, et
implémenter les récompenses dans `agent/rl/environment.py`. Aucun backend Gazebo
prêt à l’emploi n’est encore fourni. Une fois `mon_backend:create_backend`
configuré :

Terminal 1 — collecter les parties et leurs actions :

```sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 -m agent.rl.collect --backend mon_backend:create_backend \
  --robot-id bleu --opponent-id rouge --episodes 20 \
  --model checkpoints/policy.pt --records runs/selfplay
```

Sans `--opponent-model`, les deux robots utilisent le même modèle. Après la fin
de la collecte, préparer les cibles à partir des épisodes complets récompensés :

Terminal 2 — convertir les journaux en cibles :

```sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 -m agent.rl.prepare_targets runs/selfplay \
  --output runs/targets.jsonl --target-step-size 0.1
```

Terminal 3 — entraîner sur toutes les cibles :

```sh
python3 -m agent.architecture.train_targets runs/targets.jsonl \
  --model checkpoints/policy.pt \
  --output checkpoints/policy-next.pt --epochs 20
```

La préparation des cibles est une commande séparée après la collecte ;
`--record-only` ne fournit pas les récompenses nécessaires.

## Tests

Dans un terminal à la racine du dépôt :

```sh
python3 -m unittest discover -s agent/tests -v
```
