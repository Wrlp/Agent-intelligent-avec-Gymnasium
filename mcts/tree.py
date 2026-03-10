"""
Structure de l'arbre pour Monte Carlo Tree Search (MCTS)
"""
import numpy as np
import math


class Node:
    """
    Représente un nœud dans l'arbre de recherche MCTS.
    
    Chaque nœud contient:
    - L'état du jeu
    - le noeud parent
    - la liste des noeuds enfants
    - Le nombre de visite 
    - le score cumulé
    - Les actions possibles non explorées
    """
    
    def __init__(self, state, parent=None, action=None, possible_actions=None):
        """
        Initialise un nouveau nœud de l'arbre MCTS.
        
        Args:
            state: L'état du jeu à ce nœud (numpy array)
            parent: Le nœud parent (None pour la racine)
            action: L'action qui a mené à ce nœud depuis le parent
            possible_actions: Liste des actions possibles depuis cet état
        """
        self.state = state
        self.parent = parent
        self.action = action  # Action qui a mené à ce nœud
        
        self.visits = 0  # Nombre de fois que ce nœud a été visité
        self.value = 0.0  # somme des récompenses
        
        self.children = []
        self.untried_actions = list(possible_actions) if possible_actions else []
        
    def is_fully_expanded(self):
        """
        Vérifie si toutes les actions possibles ont été explorées.
        
        Returns:
            bool: True si toutes les actions ont été essayées
        """
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_weight=1.41):
        """
        Sélectionne le meilleur enfant selon la formule UCB1 (Upper Confidence Bound).
        
        UCB1 = (valeur moyenne) + exploration_weight * sqrt(ln(visites_parent) / visites_enfant)
        
        Args:
            exploration_weight: Paramètre C de la formule UCB1 (par défaut √2 ≈ 1.41)
                              - Plus élevé = plus d'exploration
                              - Plus bas = plus d'exploitation
        
        Returns:
            Node: Le nœud enfant avec le meilleur score UCB1
        """
        if not self.children:
            return None
        
        # Calcule le score UCB1 pour chaque enfant
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                # Si un enfant n'a jamais été visité, lui donner priorité maximale
                return child
            
            # Calcul de UCB1
            exploitation = child.value / child.visits
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            ucb1_score = exploitation + exploration
            
            if ucb1_score > best_score:
                best_score = ucb1_score
                best_child = child
        
        return best_child
    
    def add_child(self, state, action, possible_actions):
        """
        Ajoute un nœud enfant et retire l'action de la liste des actions non essayées.
        
        Args:
            state: L'état du jeu après avoir pris l'action
            action: L'action qui mène à cet enfant
            possible_actions: Liste des actions possibles depuis le nouvel état
        
        Returns:
            Node: Le nouveau nœud enfant créé
        """
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        
        child_node = Node(state, parent=self, action=action, possible_actions=possible_actions)
        self.children.append(child_node)
        return child_node
    
    def update(self, reward):
        """
        Met à jour les statistiques du nœud (backpropagation).
        
        Args:
            reward: La récompense à ajouter à la valeur du nœud
        """
        self.visits += 1
        self.value += reward
    
    def most_visited_child(self):
        """
        Retourne l'enfant le plus visité (utilisé pour choisir l'action finale).
        
        Returns:
            Node: L'enfant avec le plus grand nombre de visites
        """
        if not self.children:
            return None
        
        return max(self.children, key=lambda child: child.visits)
    
    def __repr__(self):
        """
        Représentation textuelle du nœud pour le débogage.
        """
        return (f"Node(action={self.action}, visits={self.visits}, "
                f"value={self.value:.2f}, avg={self.get_average_value():.2f}, "
                f"children={len(self.children)}, untried={len(self.untried_actions)})")


class MCTSTree:
    """
    Représente l'arbre de recherche MCTS complet.
    
    Facilite la gestion de l'arbre et fournit des méthodes utilitaires.
    """
    
    def __init__(self, root_state, possible_actions):
        """
        Initialise l'arbre MCTS avec un nœud racine.
        
        Args:
            root_state: L'état initial du jeu
            possible_actions: Liste des actions possibles depuis l'état initial
        """
        self.root = Node(root_state, possible_actions=possible_actions)
    
    def get_root(self):
        """
        Retourne le nœud racine de l'arbre.
        
        Returns:
            Node: Le nœud racine
        """
        return self.root
    
    def set_root(self, node):
        """
        Définit un nouveau nœud racine (utile pour réutiliser l'arbre).
        
        Args:
            node: Le nouveau nœud racine
        """
        self.root = node
        if self.root.parent is not None:
            self.root.parent = None  # Détacher du parent
    
    def tree_size(self):
        """
        Calcule la taille totale de l'arbre (nombre de nœuds).
        
        Returns:
            int: Nombre total de nœuds dans l'arbre
        """
        return self._count_nodes(self.root)
    
    def _count_nodes(self, node):
        """
        Compte récursivement le nombre de nœuds à partir d'un nœud donné.
        
        Args:
            node: Le nœud de départ
        
        Returns:
            int: Nombre de nœuds dans le sous-arbre
        """
        if node is None:
            return 0
        
        count = 1  # Compte ce nœud
        for child in node.children:
            count += self._count_nodes(child)
        
        return count
    
    def max_depth(self):
        """
        Calcule la profondeur maximale de l'arbre.
        
        Returns:
            int: Profondeur maximale
        """
        return self._calculate_depth(self.root)
    
    def _calculate_depth(self, node):
        """
        Calcule récursivement la profondeur à partir d'un nœud donné.
        
        Args:
            node: Le nœud de départ
        
        Returns:
            int: Profondeur maximale du sous-arbre
        """
        if node is None or not node.children:
            return 0
        
        return 1 + max(self._calculate_depth(child) for child in node.children)
    
    def print_tree(self, node=None, depth=0, max_depth=3):
        """
        Affiche l'arbre de manière lisible (utile pour le débogage).
        
        Args:
            node: Le nœud de départ (None = racine)
            depth: Profondeur actuelle (pour l'indentation)
            max_depth: Profondeur maximale à afficher
        """
        if node is None:
            node = self.root
        
        if depth > max_depth:
            return
        
        indent = "  " * depth
        print(f"{indent}{node}")
        
        for child in node.children:
            self.print_tree(child, depth + 1, max_depth)
