"""Breakout with a from-scratch NumPy DQN and a live neural-network display."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np
import pygame


WIDTH = 800
HEIGHT = 700
PANEL_WIDTH = 390
WINDOW_SIZE = (WIDTH + PANEL_WIDTH, HEIGHT + 40)
FPS = 60
STATE_SIZE = 7
ACTION_VALUES = (-1, 0, 1)
MODEL_PATH = Path(__file__).with_name("breakout_model.npz")

BACKGROUND = (13, 19, 34)
PLAYFIELD = (18, 27, 46)
PANEL = (22, 31, 51)
TEXT = (226, 235, 248)
MUTED = (140, 157, 181)
CYAN = (68, 220, 230)
ORANGE = (255, 166, 87)
GREEN = (111, 230, 160)
PINK = (255, 102, 144)


class QNetwork:
    """A fully connected ReLU network trained with explicit backpropagation."""

    def __init__(
        self,
        layer_sizes: tuple[int, ...] = (STATE_SIZE, 6, 2, 3),
        seed: int | None = None,
    ) -> None:
        if len(layer_sizes) < 2 or any(size <= 0 for size in layer_sizes):
            raise ValueError("A network needs at least two non-empty layers.")
        self.layer_sizes = layer_sizes
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)).astype(np.float32)
            for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [np.zeros((1, size), dtype=np.float32) for size in layer_sizes[1:]]
        self._first_moment = [np.zeros_like(value) for value in self.weights + self.biases]
        self._second_moment = [np.zeros_like(value) for value in self.weights + self.biases]
        self._adam_step = 0
        self.last_activations = [np.zeros(size, dtype=np.float32) for size in layer_sizes]
        self.last_weight_deltas = [np.zeros_like(value) for value in self.weights]

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Return Q-values for one state or a batch of states."""
        activations = np.asarray(states, dtype=np.float32)
        if activations.ndim == 1:
            activations = activations.reshape(1, -1)
        if activations.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Expected {self.layer_sizes[0]} inputs, got {activations.shape[1]}.")

        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            activations = activations @ weight + bias
            if index < len(self.weights) - 1:
                activations = np.maximum(activations, 0.0)
        return activations

    def activate_for_display(self, state: np.ndarray) -> np.ndarray:
        """Run one state through the network and retain every layer's activations."""
        activation = np.asarray(state, dtype=np.float32).reshape(1, -1)
        if activation.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Expected {self.layer_sizes[0]} inputs, got {activation.shape[1]}.")
        self.last_activations = [activation[0].copy()]
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            activation = activation @ weight + bias
            if index < len(self.weights) - 1:
                activation = np.maximum(activation, 0.0)
            self.last_activations.append(activation[0].copy())
        return activation[0].copy()

    def train_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        learning_rate: float = 0.001,
    ) -> float:
        """Update selected action values using the DQN squared TD-error loss."""
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != self.layer_sizes[0]:
            raise ValueError("states must be a batch with the network's input width.")
        if actions.shape != (len(states),) or targets.shape != (len(states),):
            raise ValueError("actions and targets must have one value per state.")
        if np.any(actions < 0) or np.any(actions >= self.layer_sizes[-1]):
            raise ValueError("An action index is outside the output layer.")
        if len(states) == 0:
            raise ValueError("Cannot train on an empty batch.")

        activations = [states]
        pre_activations = []
        current = states
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            pre_activation = current @ weight + bias
            pre_activations.append(pre_activation)
            current = (
                np.maximum(pre_activation, 0.0)
                if index < len(self.weights) - 1
                else pre_activation
            )
            activations.append(current)

        row_indices = np.arange(len(states))
        errors = current[row_indices, actions] - targets
        loss = float(np.mean(errors**2))
        gradient = np.zeros_like(current)
        gradient[row_indices, actions] = (2.0 / len(states)) * errors

        weight_gradients: list[np.ndarray] = [np.empty_like(value) for value in self.weights]
        bias_gradients: list[np.ndarray] = [np.empty_like(value) for value in self.biases]
        for index in range(len(self.weights) - 1, -1, -1):
            weight_gradients[index] = activations[index].T @ gradient
            bias_gradients[index] = np.sum(gradient, axis=0, keepdims=True)
            gradient = gradient @ self.weights[index].T
            if index > 0:
                gradient *= pre_activations[index - 1] > 0.0

        parameters = self.weights + self.biases
        gradients = weight_gradients + bias_gradients
        gradient_norm = np.sqrt(sum(float(np.sum(value**2)) for value in gradients))
        if gradient_norm > 5.0:
            gradients = [value * (5.0 / gradient_norm) for value in gradients]

        self._adam_step += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        old_weights = [value.copy() for value in self.weights]
        for index, (parameter, grad) in enumerate(zip(parameters, gradients)):
            self._first_moment[index] = beta1 * self._first_moment[index] + (1 - beta1) * grad
            self._second_moment[index] = (
                beta2 * self._second_moment[index] + (1 - beta2) * grad**2
            )
            first = self._first_moment[index] / (1 - beta1**self._adam_step)
            second = self._second_moment[index] / (1 - beta2**self._adam_step)
            parameter -= learning_rate * first / (np.sqrt(second) + epsilon)
        self.last_weight_deltas = [
            value - old for value, old in zip(self.weights, old_weights)
        ]
        return loss

    def copy_from(self, other: QNetwork) -> None:
        if self.layer_sizes != other.layer_sizes:
            raise ValueError("Cannot copy weights between networks with different shapes.")
        self.weights = [value.copy() for value in other.weights]
        self.biases = [value.copy() for value in other.biases]

    def save(self, path: Path) -> None:
        """Persist only model parameters; replay memory is intentionally transient."""
        values: dict[str, np.ndarray] = {
            "layer_sizes": np.asarray(self.layer_sizes, dtype=np.int64)
        }
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            values[f"weight_{index}"] = weight
            values[f"bias_{index}"] = bias
        np.savez(path, **values)

    def load(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            saved_sizes = tuple(int(value) for value in data["layer_sizes"])
            if saved_sizes != self.layer_sizes:
                raise ValueError(
                    f"Model architecture {saved_sizes} does not match {self.layer_sizes}."
                )
            weights = [data[f"weight_{i}"].astype(np.float32) for i in range(len(self.weights))]
            biases = [data[f"bias_{i}"].astype(np.float32) for i in range(len(self.biases))]
        for actual, saved in zip(self.weights + self.biases, weights + biases):
            if actual.shape != saved.shape:
                raise ValueError("Saved model contains a parameter with an invalid shape.")
        self.weights = weights
        self.biases = biases
        self._first_moment = [np.zeros_like(value) for value in self.weights + self.biases]
        self._second_moment = [np.zeros_like(value) for value in self.weights + self.biases]
        self._adam_step = 0


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class DQNAgent:
    """DQN agent with epsilon-greedy actions, replay memory, and a target network."""

    def __init__(self, seed: int | None = None, training_enabled: bool = True) -> None:
        self.training_enabled = training_enabled
        self.rng = random.Random(seed)
        self.online = QNetwork(seed=seed)
        self.target = QNetwork(seed=None if seed is None else seed + 1)
        self.target.copy_from(self.online)
        self.replay: deque[Transition] = deque(maxlen=50_000)
        self.gamma = 0.99
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.epsilon = 1.0 if training_enabled else 0.0
        self.minimum_epsilon = 0.05
        self.epsilon_decay_steps = 30_000
        self.environment_steps = 0
        self.training_updates = 0
        self.last_loss: float | None = None
        self.last_action = 1
        self.scores: deque[int] = deque(maxlen=100)
        self.wins: deque[int] = deque(maxlen=100)
        self.episodes = 0
        if not training_enabled:
            initialize_manual_breakout_weights(self.online)
        self.target.copy_from(self.online)

    def choose_action(self, state: np.ndarray, explore: bool = True) -> int:
        if self.training_enabled and explore and self.rng.random() < self.epsilon:
            action_index = self.rng.randrange(len(ACTION_VALUES))
        else:
            action_index = int(np.argmax(self.online.predict(state)[0]))
        self.last_action = action_index
        return ACTION_VALUES[action_index]

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if not self.training_enabled:
            return
        action_index = ACTION_VALUES.index(action)
        self.replay.append(
            Transition(state.copy(), action_index, reward, next_state.copy(), done)
        )
        self.environment_steps += 1
        self.epsilon = max(
            self.minimum_epsilon,
            1.0 - (1.0 - self.minimum_epsilon) * self.environment_steps / self.epsilon_decay_steps,
        )

        if len(self.replay) < self.batch_size or self.environment_steps % 4:
            return

        batch = self.rng.sample(self.replay, self.batch_size)
        states = np.stack([item.state for item in batch])
        next_states = np.stack([item.next_state for item in batch])
        actions = np.fromiter((item.action for item in batch), dtype=np.int64)
        rewards = np.fromiter((item.reward for item in batch), dtype=np.float32)
        not_done = np.fromiter((not item.done for item in batch), dtype=np.float32)
        next_q = self.target.predict(next_states).max(axis=1)
        targets = rewards + self.gamma * next_q * not_done

        self.last_loss = self.online.train_batch(
            states, actions, targets, learning_rate=self.learning_rate
        )
        self.training_updates += 1
        if self.training_updates % 250 == 0:
            self.target.copy_from(self.online)


def initialize_manual_breakout_weights(network: QNetwork) -> None:
    """Set a hand-designed paddle controller without using training data."""
    if (
        len(network.layer_sizes) != 4
        or network.layer_sizes[0] != STATE_SIZE
        or network.layer_sizes[1] < 4
        or network.layer_sizes[2] < 2
        or network.layer_sizes[3] != len(ACTION_VALUES)
    ):
        raise ValueError(
            "Manual controller requires 7 inputs, hidden widths of at least 4 and 2, "
            "and 3 outputs."
        )

    network.weights = [np.zeros_like(weight) for weight in network.weights]
    network.biases = [np.zeros_like(bias) for bias in network.biases]
    first_weight, second_weight, output_weight = network.weights
    first_bias, second_bias, output_bias = network.biases

    for neuron, input_index, direction in (
        (0, 6, 1.0),
        (1, 6, -1.0),
        (2, 4, 1.0),
        (3, 4, -1.0),
    ):
        first_weight[input_index, neuron] = direction
        first_bias[0, neuron] = 1.0

    delta_coefficients = np.asarray((0.5, -0.5, -0.5, 0.5), dtype=np.float32)
    margin = 0.08
    second_weight[:4, 0] = -delta_coefficients
    second_weight[:4, 1] = delta_coefficients
    second_bias[0, :2] = -margin

    # Outputs are ordered left, stay, right; the stay bias creates a dead zone.
    output_weight[0, 0] = 1.0
    output_bias[0, 1] = 0.07
    output_weight[1, 2] = 1.0
    network.last_weight_deltas = [np.zeros_like(weight) for weight in network.weights]


class BreakoutGame:
    """Small deterministic game environment whose action is left, stay, or right."""

    paddle_width = 112
    paddle_height = 14
    paddle_y = HEIGHT - 48
    ball_radius = 8
    paddle_speed = 8.0
    ball_speed = 5.0
    episode_time_limit = FPS * 180

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.brick_columns = 10
        self.brick_rows = 5
        self.brick_gap = 5
        self.brick_height = 23
        self.brick_top = 74
        self.brick_width = (
            WIDTH - (self.brick_columns + 1) * self.brick_gap
        ) // self.brick_columns
        self.initial_bricks = self.brick_columns * self.brick_rows
        self.reset()

    def reset(self) -> np.ndarray:
        self.paddle_x = WIDTH / 2
        self.lives = 3
        self.score = 0
        self.episode_frames = 0
        self.bricks = [
            pygame.Rect(
                self.brick_gap + column * (self.brick_width + self.brick_gap),
                self.brick_top + row * (self.brick_height + self.brick_gap),
                self.brick_width,
                self.brick_height,
            )
            for row in range(self.brick_rows)
            for column in range(self.brick_columns)
        ]
        self._serve_ball()
        return self.observe()

    def _serve_ball(self) -> None:
        self.ball_x = self.paddle_x + self.rng.uniform(-18.0, 18.0)
        self.ball_y = self.paddle_y - 20
        self.ball_vx = self.rng.choice((-1, 1)) * self.rng.uniform(1.0, 2.5)
        self.ball_vy = -self.ball_speed

    def observe(self) -> np.ndarray:
        """Encode game state plus a wall-reflected estimate of the ball's paddle intercept."""
        return np.asarray(
            (
                2.0 * self.ball_x / WIDTH - 1.0,
                2.0 * self.ball_y / HEIGHT - 1.0,
                self.ball_vx / self.ball_speed,
                self.ball_vy / self.ball_speed,
                2.0 * self.paddle_x / WIDTH - 1.0,
                len(self.bricks) / self.initial_bricks,
                2.0 * self._predict_paddle_intercept() / WIDTH - 1.0,
            ),
            dtype=np.float32,
        )

    def _predict_paddle_intercept(self) -> float:
        if self.ball_vy <= 0:
            return self.ball_x

        frames_to_paddle = max(0.0, (self.paddle_y - self.ball_y) / self.ball_vy)
        left_wall = self.ball_radius
        travel_width = WIDTH - 2 * self.ball_radius
        reflected_distance = (
            self.ball_x - left_wall + self.ball_vx * frames_to_paddle
        ) % (2 * travel_width)
        if reflected_distance > travel_width:
            reflected_distance = 2 * travel_width - reflected_distance
        return left_wall + reflected_distance

    def step(self, action: int) -> tuple[np.ndarray, float, bool, str | None]:
        if action not in ACTION_VALUES:
            raise ValueError(f"Action must be one of {ACTION_VALUES}, got {action}.")

        self.episode_frames += 1
        reward = -0.001
        event = None
        self.paddle_x = min(
            WIDTH - self.paddle_width / 2,
            max(self.paddle_width / 2, self.paddle_x + action * self.paddle_speed),
        )
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_x - self.ball_radius <= 0:
            self.ball_x = self.ball_radius
            self.ball_vx = abs(self.ball_vx)
        elif self.ball_x + self.ball_radius >= WIDTH:
            self.ball_x = WIDTH - self.ball_radius
            self.ball_vx = -abs(self.ball_vx)
        if self.ball_y - self.ball_radius <= 0:
            self.ball_y = self.ball_radius
            self.ball_vy = abs(self.ball_vy)

        ball_rect = pygame.Rect(
            round(self.ball_x - self.ball_radius),
            round(self.ball_y - self.ball_radius),
            self.ball_radius * 2,
            self.ball_radius * 2,
        )
        paddle_rect = pygame.Rect(
            round(self.paddle_x - self.paddle_width / 2),
            self.paddle_y,
            self.paddle_width,
            self.paddle_height,
        )
        if self.ball_vy > 0 and ball_rect.colliderect(paddle_rect):
            self.ball_y = self.paddle_y - self.ball_radius
            self.ball_vy = -abs(self.ball_vy)
            offset = (self.ball_x - self.paddle_x) / (self.paddle_width / 2)
            offset += self.rng.uniform(-0.25, 0.25)
            self.ball_vx = max(-4.5, min(4.5, offset * 4.2))
            if abs(self.ball_vx) < 2.0:
                self.ball_vx = 2.0 if offset >= 0 else -2.0
            reward += 0.12
            event = "PADDLE"

        ball_rect = pygame.Rect(
            round(self.ball_x - self.ball_radius),
            round(self.ball_y - self.ball_radius),
            self.ball_radius * 2,
            self.ball_radius * 2,
        )
        for brick in self.bricks:
            if not ball_rect.colliderect(brick):
                continue
            overlap_x = min(ball_rect.right, brick.right) - max(ball_rect.left, brick.left)
            overlap_y = min(ball_rect.bottom, brick.bottom) - max(ball_rect.top, brick.top)
            if overlap_x < overlap_y:
                self.ball_vx *= -1
            else:
                self.ball_vy *= -1
            self.bricks.remove(brick)
            self.score += 10
            reward += 1.0
            event = "BRICK"
            break

        done = False
        if self.ball_y - self.ball_radius > HEIGHT:
            self.lives -= 1
            reward -= 1.0
            event = "LIFE LOST"
            if self.lives == 0:
                done = True
            else:
                self._serve_ball()

        if not self.bricks:
            reward += 5.0
            done = True
            event = "CLEARED"
        elif not done and self.episode_frames >= self.episode_time_limit:
            reward -= 1.0
            done = True
            event = "TIME LIMIT"

        return self.observe(), reward, done, event


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = TEXT,
) -> None:
    surface.blit(font.render(text, True, color), position)


def draw_network(
    surface: pygame.Surface,
    network: QNetwork,
    fonts: dict[str, pygame.font.Font],
    origin_x: int,
    weights_frozen: bool = False,
    show_weights: bool = False,
) -> None:
    """Render activations, signed weights, and optional weight-update markers."""
    graph_top, graph_bottom = 198, 626
    layer_x = [
        origin_x + 22 + round(index * (PANEL_WIDTH - 44) / (len(network.layer_sizes) - 1))
        for index in range(len(network.layer_sizes))
    ]
    captions = ("INPUT", "H1", "H2", "H3", "H4", "Q / ACTION")
    node_positions: list[list[tuple[int, int]]] = []
    for column, (size, x) in enumerate(zip(network.layer_sizes, layer_x)):
        count = size
        usable_height = graph_bottom - graph_top
        spacing = (
            usable_height / max(1, count - 1)
            if count <= 10
            else min(9.0, usable_height / (count - 1))
        )
        total_height = spacing * (count - 1)
        start_y = (graph_top + graph_bottom - total_height) / 2
        positions = [(x, round(start_y + index * spacing)) for index in range(count)]
        node_positions.append(positions)
        caption = captions[column] if len(network.layer_sizes) == len(captions) else (
            "INPUT" if column == 0 else "OUTPUT" if column == len(network.layer_sizes) - 1 else f"H{column}"
        )
        label = fonts["tiny"].render(caption, True, MUTED)
        surface.blit(label, label.get_rect(center=(x, graph_top - 17)))

    changed_edges: list[tuple[float, tuple[int, int], tuple[int, int], int]] = []
    for layer_index, (weight, delta) in enumerate(
        zip(network.weights, network.last_weight_deltas)
    ):
        max_weight = max(float(np.max(np.abs(weight))), 1e-6)
        for source in range(weight.shape[0]):
            for target in range(weight.shape[1]):
                value = float(weight[source, target])
                strength = abs(value) / max_weight
                base = (51, 194, 198) if value >= 0 else (243, 140, 85)
                brightness = 0.18 + 0.72 * strength
                color = tuple(round(component * brightness) for component in base)
                start = node_positions[layer_index][source]
                end = node_positions[layer_index + 1][target]
                pygame.draw.line(surface, color, start, end, 1)
                if show_weights:
                    label_x = round(start[0] * 0.42 + end[0] * 0.58)
                    label_x += ((source * 2 + target * 3) % 5 - 2) * 10
                    label_y = round(start[1] * 0.42 + end[1] * 0.58)
                    label_y += ((source * 3 + target * 5) % 3 - 1) * 5
                    weight_label = fonts["weight"].render(f"{value:+.2f}", True, TEXT)
                    label_rect = weight_label.get_rect(center=(label_x, label_y))
                    pygame.draw.rect(
                        surface,
                        (15, 22, 38),
                        label_rect.inflate(3, 1),
                        border_radius=2,
                    )
                    surface.blit(weight_label, label_rect)
                change = float(delta[source, target])
                changed_edges.append((abs(change), start, end, 1 if change >= 0 else -1))

    if not weights_frozen:
        for _, start, end, direction in sorted(
            changed_edges, reverse=True, key=lambda item: item[0]
        )[:18]:
            midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            pygame.draw.circle(surface, GREEN if direction > 0 else PINK, midpoint, 3)

    for layer_index, positions in enumerate(node_positions):
        values = network.last_activations[layer_index]
        max_value = max(float(np.max(np.abs(values))), 1e-6)
        for index, center in enumerate(positions):
            level = min(1.0, abs(float(values[index])) / max_value)
            fill = (int(34 + 70 * level), int(74 + 160 * level), int(105 + 145 * level))
            radius = 3 if len(positions) > 10 else 7
            pygame.draw.circle(surface, (6, 12, 24), center, radius + 1)
            pygame.draw.circle(surface, fill, center, radius)

    for index, name in enumerate(("LEFT", "STAY", "RIGHT")):
        y = node_positions[-1][index][1]
        draw_text(surface, fonts["tiny"], name, (layer_x[-1] - 23, y + 8), MUTED)

    pygame.draw.circle(surface, (51, 194, 198), (origin_x + 20, 658), 4)
    draw_text(surface, fonts["tiny"], "positive weight", (origin_x + 29, 650), MUTED)
    pygame.draw.circle(surface, (243, 140, 85), (origin_x + 155, 658), 4)
    draw_text(surface, fonts["tiny"], "negative weight", (origin_x + 164, 650), MUTED)
    pygame.draw.circle(surface, MUTED if weights_frozen else GREEN, (origin_x + 284, 658), 3)
    legend = "weights frozen" if weights_frozen else "weight update"
    draw_text(surface, fonts["tiny"], legend, (origin_x + 293, 650), MUTED)


def draw_game(
    screen: pygame.Surface,
    game: BreakoutGame,
    agent: DQNAgent,
    state: np.ndarray,
    fonts: dict[str, pygame.font.Font],
    paused: bool,
    episode: int,
    episode_result: str | None = None,
    show_weights: bool = False,
) -> None:
    screen.fill(BACKGROUND)
    pygame.draw.rect(screen, PLAYFIELD, (0, 0, WIDTH, HEIGHT))
    pygame.draw.line(screen, (42, 57, 82), (WIDTH, 0), (WIDTH, screen.get_height()), 2)
    brick_colors = ((255, 107, 124), (255, 160, 96), (255, 209, 99), (98, 212, 171), (104, 173, 255))
    for brick_index, brick in enumerate(game.bricks):
        color = brick_colors[(brick_index // game.brick_columns) % len(brick_colors)]
        pygame.draw.rect(screen, color, brick, border_radius=4)
        pygame.draw.line(screen, (255, 255, 255), brick.topleft, (brick.right - 4, brick.top), 1)

    paddle = pygame.Rect(
        round(game.paddle_x - game.paddle_width / 2),
        game.paddle_y,
        game.paddle_width,
        game.paddle_height,
    )
    pygame.draw.rect(screen, (71, 199, 224), paddle, border_radius=7)
    pygame.draw.circle(screen, (255, 245, 220), (round(game.ball_x), round(game.ball_y)), game.ball_radius)
    pygame.draw.circle(screen, (255, 192, 106), (round(game.ball_x), round(game.ball_y)), game.ball_radius, 2)

    draw_text(screen, fonts["title"], "BREAKOUT  /  NEURAL PILOT", (25, 16))
    draw_text(
        screen,
        fonts["small"],
        f"SCORE  {game.score:04d}     LIVES  {'● ' * game.lives}     BRICKS  {len(game.bricks):02d}",
        (25, 45),
        MUTED,
    )
    if paused or episode_result:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((6, 10, 20, 155))
        screen.blit(overlay, (0, 0))
        label_text = (
            f"{episode_result}  -  PRESS R TO PLAY AGAIN"
            if episode_result
            else "PAUSED  -  SPACE TO RESUME"
        )
        label = fonts["title"].render(label_text, True, TEXT)
        screen.blit(label, label.get_rect(center=(WIDTH // 2, HEIGHT // 2)))

    panel_x = WIDTH
    pygame.draw.rect(screen, PANEL, (panel_x, 0, PANEL_WIDTH, screen.get_height()))
    draw_text(screen, fonts["title"], "NEURAL CONTROL", (panel_x + 20, 16))
    mode = "WEIGHTS FROZEN" if not agent.training_enabled else f"replay {len(agent.replay):05d}"
    draw_text(
        screen,
        fonts["small"],
        f"Episode {episode:04d}   |   {game.episode_frames // FPS:02d}s   |   {mode}",
        (panel_x + 20, 48),
        MUTED,
    )
    draw_network(
        screen,
        agent.online,
        fonts,
        panel_x,
        weights_frozen=not agent.training_enabled,
        show_weights=show_weights,
    )

    average_score = float(np.mean(agent.scores)) if agent.scores else 0.0
    win_rate = 100.0 * float(np.mean(agent.wins)) if agent.wins else 0.0
    q_values = agent.online.predict(state)[0]
    stats_top = 679
    training_status = (
        f"weights frozen   updates {agent.training_updates}"
        if not agent.training_enabled
        else f"epsilon {agent.epsilon:.3f}   updates {agent.training_updates}"
    )
    draw_text(screen, fonts["tiny"], training_status, (panel_x + 18, stats_top), TEXT)
    draw_text(
        screen,
        fonts["tiny"],
        f"avg score {average_score:5.1f}   win rate {win_rate:4.0f}%",
        (panel_x + 18, stats_top + 18),
        MUTED,
    )
    details = (
        f"Q  L {q_values[0]:+.2f}   S {q_values[1]:+.2f}   R {q_values[2]:+.2f}   |   S save  R restart"
        if not agent.training_enabled
        else f"loss {agent.last_loss or 0:.4f}   Q L {q_values[0]:+.2f} S {q_values[1]:+.2f} R {q_values[2]:+.2f}"
    )
    draw_text(screen, fonts["tiny"], details, (panel_x + 18, stats_top + 36), MUTED)


def run() -> None:
    pygame.init()
    pygame.display.set_caption("Breakout — NumPy Neural Network")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    fonts = {
        "title": pygame.font.SysFont("segoeui", 22, bold=True),
        "small": pygame.font.SysFont("segoeui", 15),
        "tiny": pygame.font.SysFont("consolas", 12),
        "weight": pygame.font.SysFont("consolas", 9),
    }
    game = BreakoutGame()
    agent = DQNAgent(training_enabled=False)
    try:
        agent.online.load(MODEL_PATH)
        agent.target.copy_from(agent.online)
        print(f"Loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print("No saved model found; starting with a fresh network.")
    except (OSError, ValueError, KeyError) as error:
        print(f"Could not load saved model ({error}); starting with a fresh network.", file=sys.stderr)

    state = game.observe()
    episode = 1
    paused = False
    episode_result: str | None = None
    show_weights = False
    running = True
    try:
        while running:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if episode_result is None:
                            paused = not paused
                    elif event.key == pygame.K_w:
                        show_weights = not show_weights
                    elif event.key == pygame.K_s:
                        agent.online.save(MODEL_PATH)
                        print(f"Saved model to {MODEL_PATH}")
                    elif event.key == pygame.K_r:
                        game.reset()
                        state = game.observe()
                        episode += 1
                        episode_result = None
                        paused = False

            if not paused and episode_result is None:
                action = agent.choose_action(state, explore=False)
                next_state, reward, done, _ = game.step(action)
                state = next_state
                if done:
                    agent.episodes += 1
                    agent.scores.append(game.score)
                    agent.wins.append(int(not game.bricks))
                    episode_result = (
                        "CLEARED!"
                        if not game.bricks
                        else "TIME LIMIT"
                        if game.episode_frames >= game.episode_time_limit
                        else "GAME OVER"
                    )

            agent.online.activate_for_display(state)
            draw_game(
                screen,
                game,
                agent,
                state,
                fonts,
                paused,
                episode,
                episode_result=episode_result,
                show_weights=show_weights,
            )
            pygame.display.flip()
    finally:
        agent.online.save(MODEL_PATH)
        pygame.quit()
    print(
        f"Session ended after {agent.episodes} completed episodes "
        f"with {agent.training_updates} weight updates."
    )


if __name__ == "__main__":
    run()
