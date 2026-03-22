"""
Tests de l'environnement Othello.
"""
import numpy as np
import sys
import os

# Ajout du parent au path pour import relatif
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.game_env import OthelloEnv, BOARD_SIZE

# helpers
def section(title: str):
    print(f"  {title}")

def ok(msg: str):
    print(f"  [OK] {msg}")

def fail(msg: str):
    print(f"  [FAIL] {msg}")
    sys.exit(1)

# test 1 : reset & etat initial
def test_reset():
    section("Test 1 — Reset & état initial")

    env = OthelloEnv(mode="logic")
    obs, info = env.reset()

    mid = BOARD_SIZE // 2

    assert obs.shape == (BOARD_SIZE, BOARD_SIZE), "Mauvaise shape observation"
    ok(f"Shape observation : {obs.shape}")

    # Les 4 pièces centrales doivent être en place
    assert env.board[mid - 1][mid - 1] == 2, "Blanc manquant en (3,3)"
    assert env.board[mid - 1][mid]     == 1, "Noir manquant en (3,4)"
    assert env.board[mid][mid - 1]     == 1, "Noir manquant en (4,3)"
    assert env.board[mid][mid]         == 2, "Blanc manquant en (4,4)"
    ok("Position initiale des 4 pièces centrales correcte")

    assert env.current_player == 1, "Le joueur initial doit être Noir (1)"
    ok("Joueur initial = Noir (1)")

    assert env.pass_count == 0
    ok("pass_count initialisé à 0")

# test 2 : coups légaux et départ
def test_legal_actions_start():
    section("Test 2 — Coups légaux au départ")

    env = OthelloEnv(mode="logic")
    env.reset()

    legal = env.get_legal_actions()

    # Au départ, Noir a exactement 4 coups possibles
    expected = {
        2 * BOARD_SIZE + 3,  # (2,3)
        3 * BOARD_SIZE + 2,  # (3,2)
        4 * BOARD_SIZE + 5,  # (4,5)
        5 * BOARD_SIZE + 4,  # (5,4)
    }

    assert set(legal) == expected, (
        f"Coups attendus : {sorted(expected)}, obtenus : {sorted(legal)}"
    )
    ok(f"4 coups légaux au départ : {sorted(legal)}")

# test 3 : application d'un coup & retournement
def test_apply_move_and_flip():
    section("Test 3 — Application d'un coup & retournement de pièces")

    env = OthelloEnv(mode="logic")
    env.reset()

    # Noir joue en (2, 3) -> doit retourner la pièce blanche en (3, 3)
    action = 2 * BOARD_SIZE + 3  # case (2,3)
    obs, reward, terminated, truncated, info = env.step(action)

    assert env.board[2][3] == 1, "La pièce noire doit être en (2,3)"
    ok("Pièce noire posée en (2,3)")

    assert env.board[3][3] == 1, "La pièce (3,3) doit être retournée en Noir"
    ok("Pièce (3,3) retournée correctement")

    assert env.current_player == 2, "C'est maintenant au Blanc de jouer"
    ok("Tour passé au joueur Blanc (2)")

    assert not terminated
    ok("Partie non terminée après le premier coup")

# test 4 : coup invalide 
def test_invalid_move():
    section("Test 4 — Coup invalide")

    env = OthelloEnv(mode="logic")
    env.reset()

    # La case (0,0) n'est jamais légale au début
    invalid_action = 0
    obs, reward, terminated, truncated, info = env.step(invalid_action)

    assert reward == -0.5, f"Récompense attendue -0.5, obtenue {reward}"
    ok(f"Coup invalide pénalisé (reward = {reward})")

    assert not terminated
    ok("Partie non terminée après coup invalide")

    assert info.get("illegal_move") is True
    ok("Flag 'illegal_move' présent dans info")

# test 5 : clone (pour MCTS)
def test_clone():
    section("Test 5 — Clone de l'environnement (pour MCTS)")

    env = OthelloEnv(mode="logic")
    env.reset()

    # On joue un coup
    action = env.get_legal_actions()[0]
    env.step(action)

    clone = env.clone()

    assert np.array_equal(clone.board, env.board), "Le plateau cloné doit être identique"
    ok("Plateau cloné identique à l'original")

    assert clone.current_player == env.current_player
    ok(f"Joueur courant cloné correct ({clone.current_player})")

    # Modifier le clone ne doit pas affecter l'original
    clone_action = clone.get_legal_actions()[0]
    clone.step(clone_action)

    assert not np.array_equal(clone.board, env.board), \
        "Modifier le clone ne doit pas affecter l'original"
    ok("Clone indépendant de l'original ")

# test 6 : partie complète aléatoire
def test_random_game():
    section("Test 6 — Partie complète aléatoire")

    env = OthelloEnv(mode="logic")
    env.reset()

    rng = np.random.default_rng(42)
    steps = 0
    max_steps = 200  # garde-fou

    done = False
    while not done and steps < max_steps:
        legal = env.get_legal_actions()
        action = rng.choice(legal)
        _, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    score = env._score()
    ok(f"Partie terminée en {steps} coups")
    ok(f"Score final -> Noir: {score[1]}  Blanc: {score[2]}")
    ok(f"Reward final : {reward}")

    total_pieces = score[1] + score[2]
    assert total_pieces > 4, "Il doit y avoir plus que 4 pièces à la fin"
    ok(f"Total pièces sur le plateau : {total_pieces}")

# test 7 : set_state (chargemeent état externe)
def test_set_state():
    section("Test 7 — set_state (pour MCTS)")

    env = OthelloEnv(mode="logic")
    env.reset()

    # Jouer quelques coups
    for _ in range(5):
        legal = env.get_legal_actions()
        if not legal:
            break
        env.step(legal[0])

    saved_board = env.board.copy()
    saved_player = env.current_player

    # Créer un nouvel env et charger l'état
    env2 = OthelloEnv(mode="logic")
    env2.reset()
    env2.set_state(saved_board, saved_player)

    assert np.array_equal(env2.board, saved_board)
    ok("État chargé correctement via set_state")

    assert env2.current_player == saved_player
    ok(f"Joueur courant restauré : {saved_player}")

# test 8 : render (visuel)
def test_render():
    section("Test 8 — Render (affichage plateau)")

    env = OthelloEnv(mode="logic")
    env.reset()
    env.render()
    ok("Render exécuté sans erreur")

# demo interactive : partie aléatoire afficher
def demo_random_game_verbose():
    section("DÉMO — Partie aléatoire (5 premiers coups affichés)")

    env = OthelloEnv(mode="logic")
    env.reset()
    env.render()

    rng = np.random.default_rng(0)

    for turn in range(5):
        legal = env.get_legal_actions()
        action = rng.choice(legal)
        player = env.current_player

        if action == env.PASS_ACTION:
            print(f"Tour {turn + 1} — Joueur {player} PASSE son tour\n")
        else:
            row, col = divmod(action, BOARD_SIZE)
            print(f"Tour {turn + 1} — Joueur {player} joue en ({row}, {col})")

        _, reward, terminated, _, info = env.step(action)
        env.render()

        if terminated:
            score = info.get("score", env._score())
            print(f"Partie terminée ! Score -> {score}  Reward: {reward}")
            break

# main
if __name__ == "__main__":
    test_reset()
    test_legal_actions_start()
    test_apply_move_and_flip()
    test_invalid_move()
    test_clone()
    test_random_game()
    test_set_state()
    test_render()
    demo_random_game_verbose()

    section("TOUS LES TESTS PASSÉS ")



#  from game_env import TicTacToeEnv
# import numpy as np

# env = TicTacToeEnv()

# state, info = env.reset()
# env.render()

# action = 0  # exemple : joueur 1 joue case 0
# state, reward, done, info = env.step(action)
# env.render()
# print("Reward:", reward, "Done:", done)

# done = False

# while not done:
#     valid_actions = np.where(env.board.flatten() == 0)[0]
#     action = np.random.choice(valid_actions)
#     state, reward, done, info = env.step(action)
#     env.render()

# print("Fin de partie | Reward:", reward)