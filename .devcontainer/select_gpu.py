GPU = "nvidia"  # "intel" ou "nvidia" ; exécuter sur l'hôte avant d'ouvrir le conteneur.

from pathlib import Path
import shutil
import subprocess


def main():
    if GPU not in ("intel", "nvidia"):
        raise SystemExit('GPU doit valoir "intel" ou "nvidia".')
    folder = Path(__file__).resolve().parent
    names = ("devcontainer.json", "Dockerfile")
    sources = [folder / f"{name}.{GPU}" for name in names]
    for source in sources:
        if not source.is_file():
            raise SystemExit(f"Modèle absent : {source}")
    # Refuser un ancien clone où les fichiers actifs sont encore versionnés.
    tracked = subprocess.run(
        ["git", "ls-files", "--", *[f".devcontainer/{name}" for name in names]],
        cwd=folder.parent, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if tracked:
        raise SystemExit(
            "Fichiers actifs encore versionnés. Voir la migration Git dans README.md."
        )
    for source, name in zip(sources, names):
        shutil.copyfile(source, folder / name)
    print(f"Configuration {GPU} activée. Ouvrir/reconstruire le Dev Container.")


if __name__ == "__main__":
    main()
