# pour test

from mcts_agent import Node, rollout, backpropagate


class FakeEnv:
    """Environnement de test compatible avec le code ChatGPT"""
    
    def __init__(self):
        self.count = 0
    
    def set_state(self, state):
        self.count = state
    
    def get_legal_actions(self):
        return [1, 2]
    
    def step(self, action):
        self.count += action
        done = self.count >= 5
        reward = 1 if self.count == 5 else (0 if not done else -1)
        return self.count, reward, done, {}
    
    def is_terminal(self):
        return self.count >= 5


def test_ucb():
    print("=== TEST UCB ===")
    
    parent = Node(state=0)
    parent.visits = 10  # obligatoire sinon parent.visits plante
    
    node = Node(state=0, parent=parent)
    
    # Cas 1 : jamais visité → infini
    assert node.ucb_score() == float('inf')
    print("✅ Nœud non visité → infini OK")
    
    # Cas 2 : score calculé normalement
    node.visits = 5
    node.value = 4.0
    score = node.ucb_score()
    print(f"✅ Score UCB : {score:.3f}")
    
    # Cas 3 : exploration favorise le peu visité
    parent2 = Node(state=0)
    parent2.visits = 100
    
    node_peu = Node(state=0, parent=parent2)
    node_peu.visits = 2
    node_peu.value = 1.0
    
    node_bcp = Node(state=0, parent=parent2)
    node_bcp.visits = 50
    node_bcp.value = 25.0
    
    assert node_peu.ucb_score() > node_bcp.ucb_score()
    print("✅ Exploration fonctionne")


def test_simulation():
    print("\n=== TEST SIMULATION ===")
    
    env = FakeEnv()
    state = 0  # état initial = compteur à 0
    
    # Cas 1 : la simulation doit finir et retourner une récompense
    result = rollout(env, state)
    assert isinstance(result, (int, float)), "ERREUR : résultat invalide"
    print(f"✅ Simulation terminée, récompense : {result}")
    
    # Cas 2 : 100 simulations pour vérifier la distribution
    resultats = [rollout(env, state) for _ in range(100)]
    print(f"✅ Sur 100 simulations : moyenne = {sum(resultats)/len(resultats):.2f}")
    
    # Cas 3 : l'état original intact (deepcopy protège)
    assert env.count == 0, "ERREUR : l'environnement original a été modifié !"
    print("✅ Environnement original intact")

def test_backpropagate():
    print("\n=== TEST BACKPROPAGATION ===")
    
    racine = Node(state=0)
    enfant = Node(state=0, parent=racine)
    petit_enfant = Node(state=0, parent=enfant)
    
    backpropagate(petit_enfant, reward=1)
    
    assert racine.visits == 1
    assert enfant.visits == 1
    assert petit_enfant.visits == 1
    assert racine.value == 1.0
    print("✅ Tous les nœuds mis à jour")
    
    backpropagate(petit_enfant, reward=-1)
    assert racine.visits == 2
    assert racine.value == 0.0  # 1 + (-1) = 0
    print("✅ Accumulation correcte")

def test_apprentissage():
    print("\n=== TEST APPRENTISSAGE ===")
    
    for n_iter in [10, 100, 500, 1000]:
        resultats = [rollout(FakeEnv(), 0) for _ in range(n_iter)]
        moyenne = sum(resultats) / len(resultats)
        print(f"Itérations {n_iter:>4} → moyenne récompense : {moyenne:.2f}")

if __name__ == "__main__":
    test_ucb()
    test_simulation()
    test_backpropagate()
    test_apprentissage()