import pytest
from pathlib import Path

# Importation de la fonction à tester
from pithon.cli import run_file

def collect_test_cases():
    """
    Parcourt le dossier tests/fixtures/programs/ pour tous les fichiers .py
    et retourne une liste de tuples (chemin_source, chemin_attendu).
    """
    cases_dir = Path(__file__).parent / "fixtures" / "programs"
    test_cases = []
    # Pour chaque fichier .py, on cherche le fichier .out correspondant
    for py_path in cases_dir.glob("*.py"):
        out_path = py_path.with_suffix(".out")
        if not out_path.exists():
            raise FileNotFoundError(f"Fichier de sortie attendu manquant pour {py_path}")
        test_cases.append((py_path, out_path))
    return test_cases


# Paramétrage du test pour chaque couple (source, attendu)
test_cases = collect_test_cases()
id_list = [x[0].name for x in test_cases]
@pytest.mark.parametrize("source_path, expected_path",
                         test_cases,
                         ids=id_list)
def test_file_outputs_match(source_path: Path,
                            expected_path: Path,
                            capfd):
    """
    Pour chaque fichier .py (mini-Python), exécute run_file(...) sur son contenu,
    capture la sortie standard, et compare avec le contenu du fichier .out correspondant.
    """
    run_file(source_path)

    # capfd est un fixture pytest qui permet de capturer stdout/stderr pendant le test
    # On récupère la sortie capturée
    captured = capfd.readouterr()
    actual_stdout = captured.out

    # On lit la sortie attendue depuis le fichier .out
    expected_stdout = expected_path.read_text(encoding="utf-8")

    # Vérification que la sortie réelle correspond à la sortie attendue
    assert actual_stdout == expected_stdout, (
        f"\nDifférence de sortie pour {source_path.name}:\n"
        f"--- obtenu ---\n{actual_stdout!r}\n"
        f"--- attendu ---\n{expected_stdout!r}\n"
    )
