# Lancer les tests unitaires

Pour lancer les tests unitaires du projet Pithon, vous pouvez utiliser la commande suivante à la racine du projet :

```bash
uv run pytest -q
```

L'option `-q` permet d'afficher les résultats de manière concise. Si vous souhaitez voir plus de détails sur les tests, vous pouvez omettre cette option.

## Limitation du nombre de tests

Si jamais vous avez trop de tests qui échouent, vous pouvez limiter le nombre de tests exécutés avec l'option `--maxfail`.

```bash
uv run pytest --maxfail=5
```

ou simplement l'option `-x` pour arrêter l'exécution après le premier échec :

```bash
uv run pytest -x
```

## Exécuter un test spécifique

Pour exécuter un test spécifique, par exemple le fichier `addition.py`,, vous pouvez utiliser la syntaxe suivante :

```bash
uv run pytest -k 'addition.py'
```

ou pour tous les fichiers qui contiennent `while` :

```bash
uv run pytest -k 'while'
```