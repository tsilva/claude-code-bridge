"""
Unit tests for client pool functionality.

These tests mock the Claude SDK to test pool logic in isolation.

Usage:
- pytest tests/test_pool.py -v
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_agent_sdk import ResultMessage

from claude_code_bridge.pool import ClientPool, make_options


def _make_mock_client():
    """Create a mock ClaudeSDKClient that handles /clear correctly."""
    client = AsyncMock()
    # receive_response needs to yield a real ResultMessage for isinstance checks
    result_msg = MagicMock(spec=ResultMessage)

    async def _receive():
        yield result_msg

    client.receive_response = _receive
    return client


@pytest.mark.unit
class TestMakeOptions:
    """Tests for make_options function."""

    def test_make_options_opus(self):
        """Options for opus model."""
        opts = make_options("opus")
        assert opts.model == "opus"
        assert opts.max_turns == 1
        assert opts.setting_sources is None
        assert opts.tools == []

    def test_make_options_sonnet(self):
        """Options for sonnet model."""
        opts = make_options("sonnet")
        assert opts.model == "sonnet"

    def test_make_options_haiku(self):
        """Options for haiku model."""
        opts = make_options("haiku")
        assert opts.model == "haiku"

    def test_make_options_system_prompt(self):
        """Options include Claude Code preset system prompt."""
        opts = make_options("opus")
        assert opts.system_prompt["type"] == "preset"
        assert opts.system_prompt["preset"] == "claude_code"

    def test_make_options_env_variable(self):
        """Options include bridge environment variable."""
        opts = make_options("opus")
        assert opts.env["CLAUDE_CODE_BRIDGE"] == "1"


@pytest.mark.unit
class TestClientPoolInit:
    """Tests for ClientPool initialization."""

    def test_default_pool_size(self):
        """Default pool size is 3."""
        pool = ClientPool()
        assert pool.size == 3

    def test_custom_pool_size(self):
        """Custom pool size."""
        pool = ClientPool(size=5)
        assert pool.size == 5

    def test_default_model(self):
        """Default model is opus."""
        pool = ClientPool()
        assert pool.default_model == "opus"

    def test_custom_default_model(self):
        """Custom default model."""
        pool = ClientPool(default_model="sonnet")
        assert pool.default_model == "sonnet"

    def test_initial_state(self):
        """Initial state is empty."""
        pool = ClientPool(size=2)
        assert len(pool._available) == 0
        assert len(pool._client_models) == 0
        assert pool._initialized is False
        assert pool._in_use == 0


@pytest.mark.unit
@pytest.mark.asyncio
class TestClientPoolInitialize:
    """Tests for pool initialization."""

    async def test_initialize_creates_clients(self):
        """Initialize creates correct number of clients."""
        pool = ClientPool(size=2, default_model="opus")

        with patch("claude_code_bridge.pool.ClaudeSDKClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            await pool.initialize()

            assert MockClient.call_count == 2
            assert mock_instance.connect.call_count == 2
            assert len(pool._available) == 2
            assert pool._initialized is True

    async def test_initialize_sets_model(self):
        """Initialize sets model for all clients."""
        pool = ClientPool(size=2, default_model="sonnet")

        with patch("claude_code_bridge.pool.ClaudeSDKClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            await pool.initialize()

            for client in pool._client_models:
                assert pool._client_models[client] == "sonnet"

    async def test_double_initialize_no_op(self):
        """Double initialization is a no-op."""
        pool = ClientPool(size=2)

        with patch("claude_code_bridge.pool.ClaudeSDKClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            await pool.initialize()
            await pool.initialize()  # Second call

            # Should only create clients once
            assert MockClient.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
class TestClientPoolAcquire:
    """Tests for pool acquire functionality."""

    async def test_acquire_matching_model(self):
        """Acquire returns a client for matching model."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claude_code_bridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            await pool.initialize()

            async with pool.acquire("opus") as client:
                assert client is mock_client

    async def test_acquire_different_model_replaces(self):
        """Acquire with different model replaces client."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claude_code_bridge.pool.ClaudeSDKClient") as MockClient:
            mock_opus = _make_mock_client()
            mock_sonnet = _make_mock_client()
            MockClient.side_effect = [mock_opus, mock_sonnet]

            await pool.initialize()

            async with pool.acquire("sonnet") as client:
                assert client is mock_sonnet
                mock_opus.disconnect.assert_called_once()

    async def test_acquire_returns_client_to_pool(self):
        """Client is returned to pool after use."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claude_code_bridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            await pool.initialize()

            async with pool.acquire("opus"):
                assert len(pool._available) == 0
                assert pool._in_use == 1

            assert len(pool._available) == 1
            assert pool._in_use == 0

    async def test_acquire_concurrent_limit(self):
        """Concurrent acquisitions limited by pool size."""
        pool = ClientPool(size=2, default_model="opus")

        with patch("claude_code_bridge.pool.ClaudeSDKClient") as MockClient:
            clients = [_make_mock_client() for _ in range(2)]
            MockClient.side_effect = clients

            await pool.initialize()

            concurrent = 0
            max_concurrent = 0
            lock = asyncio.Lock()

            async def use_client(delay: float):
                nonlocal concurrent, max_concurrent
                async with pool.acquire("opus"):
                    async with lock:
                        concurrent += 1
                        max_concurrent = max(max_concurrent, concurrent)
                    await asyncio.sleep(delay)
                    async with lock:
                        concurrent -= 1

            tasks = [
                asyncio.create_task(use_client(0.05)),
                asyncio.create_task(use_client(0.05)),
                asyncio.create_task(use_client(0.05)),
            ]
            await asyncio.gather(*tasks)

            assert max_concurrent <= 2


@pytest.mark.unit
@pytest.mark.asyncio
class TestClientPoolShutdown:
    """Tests for pool shutdown."""

    async def test_shutdown_disconnects_all(self):
        """Shutdown disconnects all clients."""
        pool = ClientPool(size=2, default_model="opus")

        with patch("claude_code_bridge.pool.ClaudeSDKClient") as MockClient:
            mock_clients = [_make_mock_client() for _ in range(2)]
            MockClient.side_effect = mock_clients

            await pool.initialize()
            await pool.shutdown()

            for mock_client in mock_clients:
                mock_client.disconnect.assert_called_once()

            assert len(pool._available) == 0
            assert len(pool._client_models) == 0
            assert pool._initialized is False

    async def test_shutdown_clears_state(self):
        """Shutdown clears all pool state."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claude_code_bridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            await pool.initialize()

            async with pool.acquire("opus"):
                pass

            await pool.shutdown()

            assert pool._in_use == 0
            assert pool._initialized is False


@pytest.mark.unit
@pytest.mark.asyncio
class TestClientPoolErrorHandling:
    """Tests for pool error handling."""

    async def test_acquire_handles_clear_error(self):
        """Pool handles error during /clear command."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claude_code_bridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.query.side_effect = Exception("Connection lost")
            # Need replacement client
            mock_replacement = _make_mock_client()
            MockClient.side_effect = [mock_client, mock_replacement]

            await pool.initialize()

            with pytest.raises(Exception, match="Connection lost"):
                async with pool.acquire("opus"):
                    pass

            mock_client.disconnect.assert_called()

    async def test_acquire_client_returned_on_success(self):
        """Client is returned to pool on successful use."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claude_code_bridge.pool.ClaudeSDKClient") as MockClient:
            mock_client = _make_mock_client()
            MockClient.return_value = mock_client

            await pool.initialize()

            async with pool.acquire("opus"):
                pass

            assert mock_client in pool._available


@pytest.mark.unit
@pytest.mark.asyncio
class TestClientPoolModelTracking:
    """Tests for model tracking in pool."""

    async def test_model_tracked_per_client(self):
        """Each client tracks its model."""
        pool = ClientPool(size=2, default_model="opus")

        with patch("claude_code_bridge.pool.ClaudeSDKClient") as MockClient:
            mock_clients = [_make_mock_client() for _ in range(2)]
            MockClient.side_effect = mock_clients

            await pool.initialize()

            for client in pool._available:
                assert pool._client_models[client] == "opus"

    async def test_model_updated_on_replacement(self):
        """Model is updated when client is replaced."""
        pool = ClientPool(size=1, default_model="opus")

        with patch("claude_code_bridge.pool.ClaudeSDKClient") as MockClient:
            mock_opus = _make_mock_client()
            mock_sonnet = _make_mock_client()
            MockClient.side_effect = [mock_opus, mock_sonnet]

            await pool.initialize()

            async with pool.acquire("sonnet"):
                pass

            assert pool._client_models[mock_sonnet] == "sonnet"
