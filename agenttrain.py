import torch
import random
import numpy as np
from collections import deque
from snakegameDL import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os
import cv2
import pickle
import time
MAX_MEMORY =50_000
BATCH_SIZE = 512
LR = 0.002

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = self.epsilon = 80
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        if reward > 0 or random.random() < 0.3:  # 30% of low/negative reward moves
            self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        final_move = [0, 0, 0]
        # If epsilon=0, agent plays deterministically (demo mode)
        self.epsilon = max(5, 80 * np.exp(-0.001 * self.n_games))

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
    
    def save_checkpoint(self, path):
        # Save state_dict properly
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved checkpoint: {path}")


    def load_checkpoint(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            print(f"Loaded pretrained model: {path}")
        else:
            print(f"No pretrained model found: {path}")

def select_version_cv():
    import cv2
    import numpy as np

    img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.putText(img, "Snake RL Demo", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(img, "Choose model version:", (120, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Updated options
    options = [
        "0 → Untrained",
        "1 → 1000 games",
        "2 → 5000 games",
        "3 → 10000 games",
        "4 → 15000 games",
    ]
    
    y_start = 140
    y_gap = 40
    for i, text in enumerate(options):
        cv2.putText(img, text, (100, y_start + i * y_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    
    cv2.putText(img, "Press corresponding key (0-4)", (120, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

    cv2.imshow("Select Version", img)

    # Updated mapping
    version_map = {
        '0': 0,
        '1': 1000,
        '2': 5000,
        '3': 10000,
        '4': 15000,
    }

    selected_version = None

    while True:
        key = cv2.waitKey(1) & 0xFF
        if chr(key) in version_map:
            selected_version = version_map[chr(key)]
            break

    cv2.destroyAllWindows()
    return selected_version

def train(load_version=None, demo_only=False):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(render=True)
    
    # Load pretrained model if requested
    if load_version is not None:
        checkpoint_path = f'./model/model_{load_version}.pth'
        agent.load_checkpoint(checkpoint_path)
        score_file = f'./model/score_{load_version}.pkl'
        if os.path.exists(score_file):
            with open(score_file, 'rb') as f:
                data = pickle.load(f)
            plot_scores = data['scores']
            plot_mean_scores = data['mean_scores']

    if demo_only:
        agent.epsilon = 0
        print(f"\nRunning demo for version {load_version}...\n")
        while True:
            state = agent.get_state(game)
            move = agent.get_action(state)
            reward, done, score = game.play_step(move)
            if done:
                game.reset()
                plot_scores.append(score)
                mean_score = sum(plot_scores) / len(plot_scores)
                plot_mean_scores.append(mean_score)
                print(f"Demo Score: {score}")
                plot(plot_scores, plot_mean_scores, demo=True)
        return
    # TRAINING MODE
    MAX_GAMES = 25000 # Optional: limit training
    while agent.n_games < MAX_GAMES:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.save_checkpoint('./model/best_model.pth')

            checkpoint_games = [25, 50, 75, 100, 150,200,250,300,500,700,1000,1500,2000,3000,4000,5000,10000,15000,20000,25000]
            if agent.n_games in checkpoint_games:
                path = f'./model/model_{agent.n_games}.pth'
                agent.save_checkpoint(path)
                scores_data = {'scores': plot_scores, 'mean_scores': plot_mean_scores}
                with open(f'./model/score_{agent.n_games}.pkl', 'wb') as f:
                    pickle.dump(scores_data, f)
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

def run_demo():
    # Let user select which trained version to display
    selected_version = select_version_cv()  # OpenCV window for version selection
    
    agent = Agent()
    game = SnakeGameAI()

    # Load selected checkpoint
    checkpoint_path = f'./model/model_{selected_version}.pth'
    agent.load_checkpoint(checkpoint_path)

    agent.epsilon = 0  # deterministic play
    plot_scores = []
    plot_mean_scores = []
    total_score = 0

    print(f"\nRunning demo for version {selected_version}...\n")
    while True:
        state = agent.get_state(game)
        move = agent.get_action(state)
        reward, done, score = game.play_step(move)
        if done:
            game.reset()
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / len(plot_scores)
            plot_mean_scores.append(mean_score)
            print(f"Demo Score: {score}")
            plot(plot_scores, plot_mean_scores, demo=True)

if __name__ == '__main__':
    train(load_version=None, demo_only=False)
        