# Configurer et lancer après un clone

Ce fichier fournit les informations pour un dédmarrage rapide.
En cas de problème, se référer au `README.md` à la racine du repo.
---

Le dépôt fournit les configurations Intel/NVIDIA et les scripts de setup et de
compilation pour ROS 2 Jazzy / Gazebo Harmonic. Aucune modification des manifestes
ni commande de configuration Git n’est nécessaire après un nouveau clone.

## 1. Configurer sur l’hôte

Prérequis : Git, Python 3, Docker, VS Code et son extension **Dev Containers**.
Cette configuration utilise Linux avec X11/XWayland : `DISPLAY` doit être défini
et `XAUTHORITY` doit pointer vers un fichier d’authentification existant.
Pour NVIDIA, installer aussi le NVIDIA Container Toolkit sur l’hôte. Se référer
au `README.md` à la racine du repo pour gérer ces problèmes si besoin.

À la racine du dépôt cloné :

1. Mettre `GPU = "intel"` ou `GPU = "nvidia"` en première ligne de
   `.devcontainer/select_gpu.py`.
2. Pour Intel, vérifier l’accès à `/dev/dri` et adapter si nécessaire
   `--group-add=105` dans `.devcontainer/devcontainer.json.intel` au GID du groupe GPU.
3. Exécuter **avant d’ouvrir le conteneur** :

```bash
python3 .devcontainer/select_gpu.py
```

Le script crée `.devcontainer/devcontainer.json` et `.devcontainer/Dockerfile`
à partir de la variante choisie. Les fichiers actifs sont ignorés par Git ;
les modèles avec suffixe restent versionnés.

Ouvrir le dépôt dans VS Code, puis **Dev Containers: Reopen in Container**.
Docker construit l’image et VS Code ouvre le terminal dans le conteneur.

## 2. Installer, compiler et lancer

Dans le terminal du conteneur, à la racine du dépôt :

```bash
source /opt/ros/jazzy/setup.bash
./setup.sh
./build.sh
source install/setup.bash
ros2 launch arcl_simulation sim.launch.py
```

`./setup.sh` complète les dépendances déclarées dans les `package.xml` via rosdep.
Il est recommandé dans un conteneur neuf (réseau et sudo requis), mais peut être
omis si toutes les dépendances sont déjà installées. Les deux Dockerfiles
installent `tf_transformations` et `transforms3d`, utilisés par ArUco.
Le `if` du script ignore uniquement l’import optionnel de `src/ros2.repos` quand
ce fichier est absent ; les erreurs d’installation arrêtent toujours le script.
La compilation
produit les cinq packages. Le lancement ouvre Gazebo et RViz.
Dans RViz, utiliser **2D Goal Pose** pour donner une cible au robot.
Arrêter la simulation avec `Ctrl+C` dans le terminal.

## 3. Relancer

Dans un nouveau terminal du conteneur :

```bash
source /opt/ros/jazzy/setup.bash
./build.sh # si des fichiers ont été modifiés
source install/setup.bash
ros2 launch arcl_simulation sim.launch.py
```