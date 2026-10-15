# Compilation et organisation de l’installation

`build.sh` utilise `--merge-install` : tous les paquets partagent le répertoire
`install/`, au lieu d'avoir chacun `install/<paquet>/`. Garde ce même mode pour
les commandes `colcon build` et `colcon test`. Si colcon signale un conflit de
layout, réutilise `--merge-install` plutôt que de supprimer `install/`. Pour
changer volontairement de mode, repars de répertoires `build/` et `install/`
propres dans un terminal qui n'a pas chargé l'ancien environnement.
