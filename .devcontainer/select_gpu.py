from pathlib import Path
import shutil
import subprocess


def select_gpu():
    choice = input("Choisissez le GPU (1 : Intel, 2 : NVIDIA) : ").strip()
    if choice == "1":
        return "intel"
    if choice == "2":
        return "nvidia"
    raise SystemExit("Choix invalide : entrez 1 (Intel) ou 2 (NVIDIA).")


def main():
    gpu = select_gpu()
    folder = Path(__file__).resolve().parent
    names = ("devcontainer.json", "Dockerfile")
    sources = [folder / f"{name}.{gpu}" for name in names]
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
    print(f"Configuration {gpu} activée. Ouvrir/reconstruire le Dev Container.")


if __name__ == "__main__":
    main()
