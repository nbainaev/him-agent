import numpy as np
import pygame
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from hima.common.sdr import sparse_to_dense

def get_color_with_alpha_rgb(hex_color, alpha, background_color=(255, 255, 255)):

    if hex_color.startswith('#'):
        hex_color = hex_color[1:]

    r_obj = int(hex_color[0:2], 16)
    g_obj = int(hex_color[2:4], 16)
    b_obj = int(hex_color[4:6], 16)

    r_bg, g_bg, b_bg = background_color
    
    r = int(r_obj * alpha + r_bg * (1 - alpha))
    g = int(g_obj * alpha + g_bg * (1 - alpha))
    b = int(b_obj * alpha + b_bg * (1 - alpha))
    
    return (r, g, b)

def compute_successor_features(transition_matrix, emission_matrix, init_label, n_steps=5, gamma=0.9):

    T = transition_matrix.mean(axis=0)
    E = np.eye(T.shape[0])
    current_state = sparse_to_dense([init_label], size=T.shape[0])
    perfect_sf = E[init_label].copy()

    discount = gamma
    for _ in range(n_steps):
        current_state = current_state @ T
        obs_dist = current_state @ E
        perfect_sf += discount * obs_dist
        discount *= gamma
    
    return perfect_sf

def generate_emission_matrix_symbols(field):

    n = len(field)
    total_states = n * n
    
    unique_symbols = sorted(set(np.array(field).flatten()))
    num_symbols = len(unique_symbols)
    
    symbol_to_index = {symbol: idx for idx, symbol in enumerate(unique_symbols)}
    index_to_symbol = {idx: symbol for idx, symbol in enumerate(unique_symbols)}
    

    emission_matrix = np.zeros((total_states, num_symbols), dtype=float)
    

    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    for i in range(n):
        for j in range(n):
            current_state = i * n + j
            current_value = field[i][j]

            for dx, dy in directions:
                ni, nj = i + dx, j + dy

                if 0 <= ni < n and 0 <= nj < n:

                    observed_symbol = field[ni][nj]
                else:

                    observed_symbol = -1
                

                if observed_symbol not in symbol_to_index:

                    new_idx = len(unique_symbols)
                    symbol_to_index[observed_symbol] = new_idx
                    index_to_symbol[new_idx] = observed_symbol
                    unique_symbols.append(observed_symbol)
                    

                    new_emission = np.zeros((total_states, len(unique_symbols)), dtype=float)
                    new_emission[:, :-1] = emission_matrix
                    emission_matrix = new_emission
                    num_symbols = len(unique_symbols)
                
                symbol_idx = symbol_to_index[observed_symbol]
                emission_matrix[current_state, symbol_idx] += 0.25 
    
    return {
        'emission_matrix': emission_matrix,
        'symbol_to_index': symbol_to_index,
        'index_to_symbol': index_to_symbol,
        'unique_symbols': unique_symbols
    }

def generate_true_transition_matrix(field):

    n = len(field)
    total_states = n * n

    transition_matrix = np.zeros((4, total_states, total_states), dtype=float)
    

    direction_vectors = [
        (-1, 0),  # 0: up
        (0, 1),   # 1: right
        (1, 0),   # 2: down
        (0, -1)   # 3: left
    ]
    
    for i in range(n):
        for j in range(n):
            current_state = i * n + j

            if field[i][j] < 0:
                continue

            for action in range(4):
                dx, dy = direction_vectors[action]
                ni, nj = i + dx, j + dy

                if 0 <= ni < n and 0 <= nj < n:

                    if field[ni][nj] >= 0:

                        target_state = ni * n + nj
                        transition_matrix[action, current_state, target_state] = 1.0
                    else:

                        transition_matrix[action, current_state, current_state] = 1.0
                else:

                    transition_matrix[action, current_state, current_state] = 1.0
    
    sums = transition_matrix.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    transition_matrix /= sums
    return transition_matrix

def plot_training_metrics(episodes, accuracy, total_loss=None, energy=None, accuracy_eval=None,
                          accuracy_time=None, accuracy_time_eval=None,
                          config=None, new_fig=True, fig=None, axs=None,
                          figsize=(10, 8), style='seaborn-v0_8', label='', show=True,
                          colors=None, linewidth=2.5, title_name='', fig_name=None,
                          grid_alpha=0.3, title_fontsize=12,
                          save_path=None):
    from scipy.signal import savgol_filter
    from scipy.interpolate import make_smoothing_spline
    ['#2ca02c', '#d62728', '#ff7f0e', '#1f77b4']
    if colors is None:
        colors = ['#1f77b4'] * 4

    plt.style.use(style)
    mpl.rcParams['font.family'] = 'DejaVu Sans' 
    metrics = [accuracy, accuracy_eval, accuracy_time['acc'], accuracy_time_eval['acc']]
    labels = ['Training Accuracy', 'Accuracy Eval', 'Accuracy Time', 'Accuracy Time Eval']
    n_subplots = 0
    for i, metric in enumerate(metrics):
        if metric is None:
            metrics.pop(i)
            labels.pop(i)
        else:
            n_subplots += 1
    if new_fig:
        fig, axs = plt.subplots(n_subplots // 2, 2, figsize=figsize, 
                            facecolor='#f5f5f5' if style != 'dark_background' else '#2b2b2b',
                            constrained_layout=True)
    
        fig.suptitle(f'Training Metrics Analysis {title_name.upper()}', fontsize=14)
    
    for i, (metric, Label) in enumerate(zip(metrics, labels)):
        if 'time' in Label.lower():
            x = list(range(len(metric['mean'])))
        elif 'eval' in Label.lower():
            x = episodes[::config['eval_every']]
        else:
            x = episodes
        if 'loss' not in Label.lower():
            spline_mean = make_smoothing_spline(x, np.array(metric['mean'])) 
            smoothed_mean = np.array(spline_mean(x))
            smoothed_mean = savgol_filter(smoothed_mean, window_length=5, polyorder=3)

            spline_std = make_smoothing_spline(x, np.array(metric['std']))
            smoothed_std = np.array(spline_std(x))
            smoothed_std = savgol_filter(smoothed_std, window_length=5, polyorder=3)
            axs[i // 2, i % 2].fill_between(x, 
                                np.maximum(smoothed_mean - smoothed_std, 0), 
                                np.minimum(smoothed_mean + smoothed_std, 1), 
                                color=colors[i], 
                                alpha=0.3,
                                )
        else:
            spline_mean = make_smoothing_spline(x, np.array(metric), lam=0.3)
            smoothed_mean = np.array(spline_mean(x))
            smoothed_mean = savgol_filter(smoothed_mean, window_length=5, polyorder=3)
        axs[i // 2, i % 2].plot(x, smoothed_mean, 
                    linewidth=linewidth, 
                    color=colors[i],
                    label=Label if not label else label)
        axs[i // 2, i % 2].set_title(Label, fontsize=title_fontsize)
        axs[i // 2, i % 2].set_xlabel('Time step' if 'Time' in Label else 'Epoch', fontsize=10)
        axs[i // 2, i % 2].set_ylabel('Accuracy' if 'Accuracy' in Label else 'Loss', fontsize=10)
        axs[i // 2, i % 2].grid(alpha=grid_alpha)
        axs[i // 2, i % 2].legend(loc='lower right')
        

    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    if save_path:
        name = f"epochs - {config['n_epochs']}, batch_size - {config['batch_size']}, n_steps - {config['n_steps']}, seq_len - {config['sequence_length']}.jpg" if not fig_name else fig_name
        plt.savefig(save_path + "/" + name, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    if show:
        plt.show()
    
    return fig, axs

class RealTimeVisualizer:
    def __init__(self, field, terminal_states=None, rewards=None, cell_size=60, trace_len=3):
        self.field = field
        self.terminal_states = terminal_states
        self.cell_size = cell_size
        self.trace_len = trace_len
        self.position_history = []
        self.rewards = rewards

        unique_values = np.unique(field)
        pos_values = unique_values[unique_values >= 0]
        neg_values = unique_values[unique_values < 0]
        
        colors_pos = cm.Pastel1(np.linspace(0, 1, len(pos_values)+1))[1:]
        colors_neg = cm.Dark2(np.linspace(0, 1, len(neg_values)+1))[1:]
        
        self.color_map = {}
        for val in sorted(unique_values):
            if val >= 0:
                idx = np.where(pos_values == val)[0][0]
                self.color_map[val] = tuple(int(c * 255) for c in colors_pos[idx][:3])
            else:
                idx = np.where(neg_values == val)[0][0]
                self.color_map[val] = tuple(int(c * 255) for c in colors_neg[idx][:3])

        self.sf_cmap = cm.Blues
        self.sf_norm = Normalize(vmin=0, vmax=1)
        

        pygame.init()
        self.width = field.shape[1] * cell_size
        self.height = field.shape[0] * cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('GridWorld with Successor Features')
        self.clock = pygame.time.Clock()
        

        self.font = pygame.font.SysFont('Arial', 12)
        self.large_font = pygame.font.SysFont('Arial', 20, bold=True)
        self.sf_font = pygame.font.SysFont('Arial', 10)
        
        self.direction_arrows = {0: '←', 1: '→', 2: '↑', 3: '↓'}

        self.current_sf = None

        self.sf_color = "#6600ff"
        self.alpha_func = lambda x: min(1, x)
    
    def update(self, position, direction=None, step=None, successor_features=None, reward=None):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        

        if successor_features is not None:
            self.current_sf = successor_features.reshape(int(len(successor_features.squeeze()) ** 0.5), -1)
            self.current_sf /= self.current_sf.max()
        

        self.position_history.append(position)
        if len(self.position_history) > self.trace_len:
            self.position_history.pop(0)
        

        self.screen.fill((0, 0, 0))
        

        for y in range(self.field.shape[0]):
            for x in range(self.field.shape[1]):
                cell_value = self.field[y, x]
                

                if self.current_sf is not None and cell_value >= 0:
                    sf_value = self.current_sf[y, x]
                    sf_color = get_color_with_alpha_rgb(self.sf_color, self.alpha_func(sf_value))

                    pygame.draw.rect(self.screen, sf_color,
                                    (x * self.cell_size, y * self.cell_size, 
                                     self.cell_size, self.cell_size))
                    
                    # Отображаем значение successor feature
                    text = self.font.render(str(cell_value), True, 
                                       (0, 0, 0) if cell_value >= 0 else (255, 255, 255))
                    self.screen.blit(text, 
                                    (x * self.cell_size + self.cell_size // 2 - text.get_width() // 2,
                                    y * self.cell_size + self.cell_size // 2 - text.get_height() // 2))
                else:

                    color = self.color_map.get(cell_value, (200, 200, 200))
                    pygame.draw.rect(self.screen, color, 
                                    (x * self.cell_size, y * self.cell_size, 
                                     self.cell_size, self.cell_size))
                    

                    text = self.font.render(str(cell_value), True, 
                                           (0, 0, 0) if cell_value >= 0 else (255, 255, 255))
                    self.screen.blit(text, 
                                    (x * self.cell_size + self.cell_size // 2 - text.get_width() // 2,
                                     y * self.cell_size + self.cell_size // 2 - text.get_height() // 2))
                

                pygame.draw.rect(self.screen, (0, 0, 0), 
                                (x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size), 1)
                

                if self.terminal_states is not None and self.terminal_states[y, x] == 1:
                    term_text = self.large_font.render('G', True, (255, 0, 0)) 
                    self.screen.blit(term_text,
                                    (x * self.cell_size + self.cell_size - term_text.get_width() - 5,
                                     y * self.cell_size + 5))
                
                if self.rewards is not None and self.rewards[y, x] != 0:
                    term_text = self.large_font.render(f'{self.rewards[y, x]}', True, (21,71,52))
                    self.screen.blit(term_text,
                                    (x * self.cell_size + self.cell_size - term_text.get_width() - 5,
                                     y * self.cell_size + 5))
        

        if len(self.position_history) > 1:
            for i, pos in enumerate(self.position_history[:-1]):
                alpha = int(255 * (0.3 + 0.5 * (i / len(self.position_history))))
                pygame.draw.circle(self.screen, (255, 0, 0, alpha),
                                  (pos[1] * self.cell_size + self.cell_size // 2,
                                   pos[0] * self.cell_size + self.cell_size // 2),
                                  self.cell_size // 6)
            

            if len(self.position_history) > 1:
                points = [(p[1] * self.cell_size + self.cell_size // 2,
                           p[0] * self.cell_size + self.cell_size // 2) 
                          for p in self.position_history]
                pygame.draw.lines(self.screen, (0, 0, 255), False, points, 3)

        pygame.draw.circle(self.screen, (0, 0, 255),
                          (position[1] * self.cell_size + self.cell_size // 2,
                           position[0] * self.cell_size + self.cell_size // 2),
                          self.cell_size // 3)
        

        if direction is not None:
            arrow_text = self.large_font.render(self.direction_arrows.get(direction, '?'), True, (0, 0, 0))
            self.screen.blit(arrow_text,
                            (position[1] * self.cell_size + self.cell_size - arrow_text.get_width(),
                             position[0] * self.cell_size))
        

        if step is not None:
            step_text = self.font.render(f'Step: {step}', True, (0, 0, 0))
            self.screen.blit(step_text, (10, 10))

        if reward is not None:
            legend_text = self.font.render(f'Reward: {reward}', True, (0, 0, 0))
            self.screen.blit(legend_text, (10, 30))
        

        pygame.display.flip()
        self.clock.tick(5)
    
    def close(self):
        pygame.quit()