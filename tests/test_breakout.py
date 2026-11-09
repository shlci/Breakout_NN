import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pygame

from breakout import (
    ACTION_VALUES,
    BreakoutGame,
    DQNAgent,
    QNetwork,
    STATE_SIZE,
    draw_game,
    draw_network,
    initialize_manual_breakout_weights,
)


class QNetworkTests(unittest.TestCase):
    def test_default_architecture_has_compact_hidden_layers(self):
        network = QNetwork(seed=8)

        self.assertEqual(network.layer_sizes, (STATE_SIZE, 6, 2, 3))

    def test_training_updates_weights_and_reduces_fixed_target_error(self):
        network = QNetwork((STATE_SIZE, 16, 3), seed=12)
        states = np.ones((8, STATE_SIZE), dtype=np.float32) * 0.25
        actions = np.zeros(8, dtype=np.int64)
        targets = np.full(8, 2.0, dtype=np.float32)
        original = [weight.copy() for weight in network.weights]
        initial_loss = float(np.mean((network.predict(states)[:, 0] - targets) ** 2))

        for _ in range(120):
            network.train_batch(states, actions, targets, learning_rate=0.005)

        final_loss = float(np.mean((network.predict(states)[:, 0] - targets) ** 2))
        self.assertTrue(any(not np.array_equal(before, after) for before, after in zip(original, network.weights)))
        self.assertLess(final_loss, initial_loss)

    def test_handwritten_network_matches_previous_architecture(self):
        previous = QNetwork((STATE_SIZE, 64, 48, 3), seed=11)
        initialize_manual_breakout_weights(previous)
        compact = QNetwork(seed=12)
        initialize_manual_breakout_weights(compact)
        states = np.random.default_rng(13).uniform(-1.0, 1.0, (128, STATE_SIZE)).astype(
            np.float32
        )

        np.testing.assert_allclose(compact.predict(states), previous.predict(states), atol=1e-6)

    def test_save_and_load_round_trip(self):
        network = QNetwork(seed=3)
        state = np.zeros(STATE_SIZE, dtype=np.float32)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            expected = network.predict(state)
            network.save(path)
            restored = QNetwork(seed=9)
            restored.load(path)
        np.testing.assert_allclose(restored.predict(state), expected)


class GameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        pygame.display.set_mode((1, 1))

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def test_state_shape_and_actions_keep_game_in_bounds(self):
        game = BreakoutGame(seed=1)
        state = game.observe()
        self.assertEqual(state.shape, (STATE_SIZE,))
        self.assertEqual(len(game.bricks), game.initial_bricks)

        for action in ACTION_VALUES:
            state, reward, done, _ = game.step(action)
            self.assertEqual(state.shape, (STATE_SIZE,))
            self.assertGreaterEqual(game.paddle_x, game.paddle_width / 2)
            self.assertLessEqual(game.paddle_x, 800 - game.paddle_width / 2)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(done, bool)

    def test_brick_collision_removes_brick_and_rewards_agent(self):
        game = BreakoutGame(seed=2)
        brick = game.bricks[0]
        game.ball_x = float(brick.centerx)
        game.ball_y = float(brick.bottom + 5)
        game.ball_vx = 0.0
        game.ball_vy = -game.ball_speed

        _, reward, done, event = game.step(0)

        self.assertEqual(len(game.bricks), game.initial_bricks - 1)
        self.assertEqual(game.score, 10)
        self.assertGreater(reward, 0)
        self.assertFalse(done)
        self.assertEqual(event, "BRICK")

    def test_agent_outputs_only_supported_actions(self):
        agent = DQNAgent(seed=4)
        action = agent.choose_action(np.zeros(STATE_SIZE, dtype=np.float32))
        self.assertIn(action, ACTION_VALUES)

    def test_frozen_agent_does_not_change_weights_or_store_transitions(self):
        agent = DQNAgent(seed=9, training_enabled=False)
        state = np.zeros(STATE_SIZE, dtype=np.float32)
        original_weights = [weight.copy() for weight in agent.online.weights]

        agent.learn(state, 0, 1.0, state, False)

        self.assertEqual(len(agent.replay), 0)
        self.assertEqual(agent.training_updates, 0)
        self.assertEqual(agent.epsilon, 0.0)
        for before, after in zip(original_weights, agent.online.weights):
            np.testing.assert_array_equal(after, before)

    def test_handwritten_controller_tracks_ball_and_velocity(self):
        agent = DQNAgent(seed=10, training_enabled=False)
        left_state = np.asarray((-0.7, 0.0, 0.0, 1.0, 0.5, 1.0, -0.7), dtype=np.float32)
        right_state = np.asarray((0.7, 0.0, 0.0, 1.0, -0.5, 1.0, 0.7), dtype=np.float32)
        velocity_lead_state = np.asarray((0.0, 0.0, 0.5, 1.0, 0.0, 1.0, 0.5), dtype=np.float32)
        aligned_state = np.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0), dtype=np.float32)

        self.assertEqual(agent.choose_action(left_state, explore=False), -1)
        self.assertEqual(agent.choose_action(right_state, explore=False), 1)
        self.assertEqual(agent.choose_action(velocity_lead_state, explore=False), 1)
        self.assertEqual(agent.choose_action(aligned_state, explore=False), 0)

    def test_predictor_estimates_intercept_with_wall_reflection(self):
        game = BreakoutGame(seed=14)
        game.ball_x = 790.0
        game.ball_y = 350.0
        game.ball_vx = 4.0
        game.ball_vy = 5.0

        intercept = game._predict_paddle_intercept()
        travel_time = (game.paddle_y - game.ball_y) / game.ball_vy
        unreflected_x = game.ball_x + game.ball_vx * travel_time
        self.assertGreater(unreflected_x, 800)
        self.assertGreaterEqual(intercept, game.ball_radius)
        self.assertLessEqual(intercept, 800 - game.ball_radius)
        self.assertNotAlmostEqual(intercept, unreflected_x)

    def test_episode_has_a_finite_time_limit(self):
        game = BreakoutGame(seed=15)
        game.episode_frames = game.episode_time_limit - 1

        _, reward, done, event = game.step(0)

        self.assertTrue(done)
        self.assertEqual(event, "TIME LIMIT")
        self.assertLess(reward, 0)

    def test_frozen_ai_can_clear_a_complete_episode(self):
        game = BreakoutGame(seed=2)
        agent = DQNAgent(seed=2, training_enabled=False)
        done = False
        event = None

        while not done:
            action = agent.choose_action(game.observe(), explore=False)
            _, _, done, event = game.step(action)

        self.assertEqual(event, "CLEARED")
        self.assertEqual(game.score, game.initial_bricks * 10)

    def test_dqn_learns_from_replay_batch(self):
        agent = DQNAgent(seed=5)
        agent.batch_size = 2
        state = np.zeros(STATE_SIZE, dtype=np.float32)
        for _ in range(4):
            agent.learn(state, 0, 1.0, state, False)

        self.assertEqual(len(agent.replay), 4)
        self.assertEqual(agent.training_updates, 1)
        self.assertIsNotNone(agent.last_loss)

    def test_game_and_network_visualization_render(self):
        game = BreakoutGame(seed=6)
        agent = DQNAgent(seed=7)
        state = game.observe()
        agent.online.activate_for_display(state)
        fonts = {
            "title": pygame.font.Font(None, 24),
            "small": pygame.font.Font(None, 18),
            "tiny": pygame.font.Font(None, 14),
            "weight": pygame.font.Font(None, 10),
        }
        screen = pygame.Surface((1190, 740))

        draw_game(
            screen,
            game,
            agent,
            state,
            fonts,
            paused=False,
            episode=1,
            episode_result="GAME OVER",
            show_weights=True,
        )

        self.assertEqual(screen.get_size(), (1190, 740))

    def test_weight_labels_toggle_changes_network_render(self):
        agent = DQNAgent(seed=19, training_enabled=False)
        agent.online.activate_for_display(np.zeros(STATE_SIZE, dtype=np.float32))
        fonts = {
            "tiny": pygame.font.Font(None, 14),
            "weight": pygame.font.Font(None, 10),
        }
        hidden = pygame.Surface((1190, 740))
        visible = pygame.Surface((1190, 740))
        hidden.fill((22, 31, 51))
        visible.fill((22, 31, 51))

        draw_network(hidden, agent.online, fonts, 800, weights_frozen=True)
        draw_network(visible, agent.online, fonts, 800, weights_frozen=True, show_weights=True)

        self.assertFalse(np.array_equal(pygame.surfarray.array3d(hidden), pygame.surfarray.array3d(visible)))


if __name__ == "__main__":
    unittest.main()
