# Rapport DQN - Projet #2

## Architecture de l'Agent
L'agent DQN implémenté est flexible et supporte deux types d'architectures :
1. **MLP (Multi-Layer Perceptron)** : Utilisé pour les états de faible dimension (ex: plateau de Morpion 3x3).
2. **CNN (Convolutional Neural Network)** : Prêt pour les jeux Atari (observations d'images).

### Composants clés
- **Replay Buffer** : Stocke les transitions $(s, a, r, s', done)$ pour briser la corrélation entre les données.
- **Target Network** : Utilisation d'un réseau cible pour stabiliser l'apprentissage.
- **Epsilon-Greedy** : Stratégie d'exploration avec décroissance exponentielle.
- **Loss** : Utilisation de `SmoothL1Loss` (Huber Loss) pour plus de robustesse.

## Environnement
Actuellement testé sur un environnement personnalisé `TicTacToeEnv` compatible avec l'API Gymnasium.
- **Observation** : Plateau 3x3 aplati (9 valeurs).
- **Actions** : 9 cases possibles (avec filtrage des actions légales).
- **Récompense** : +1 pour une victoire, 0 pour un nul, -10 pour un coup invalide (bien que filtré).

## Résultats Préliminaires
L'entraînement sur 2000 épisodes montre que l'agent apprend à jouer des coups valides et à chercher la victoire.
- **Device** : CUDA (si disponible).
- **Epsilon final** : 0.14.
- **Reward moyen** : ~0.8-0.9 (indiquant des victoires fréquentes en self-play).

## Prochains Défis
- Résoudre les dépendances pour `ale-py` (Atari) dans l'environnement de développement.
- Implémenter le **Double DQN** pour réduire la surestimation des Q-values.
- Intégrer le **MCTS** pour comparer les performances.
