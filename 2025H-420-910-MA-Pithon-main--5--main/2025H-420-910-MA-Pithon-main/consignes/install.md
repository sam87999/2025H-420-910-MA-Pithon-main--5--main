# Installer le projet Pithon

Une fois le projet [cloné](clone.md), il faut installer les dépendances et le projet lui-même.
Le plus simple est d'utiliser le gestionnaire de projet [`uv`](https://docs.astral.sh/uv/).

## Installer `uv`

Les instructions pour installer `uv` sont disponibles sur la [page d'installation](https://docs.astral.sh/uv/getting-started/installation/).

Sur Mac, vous pouvez aussi utiliser `brew`.

## Démarrer le projet
Une fois `uv` installé, il suffit de lancer la commande suivante à la racine du projet :

```bash
uv run pithon
```

Avec cette commande, `uv` va créer un environnement virtuel, installer les
dépendances et lancer le projet. Nous verrons durant le cours ce qu'est un
environnement virtuel et comment il fonctionne.