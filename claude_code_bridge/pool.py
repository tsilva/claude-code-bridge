"""Client pool for reduced latency via connection reuse."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ResultMessage

logger = logging.getLogger(__name__)


def make_options(model: str | None = None) -> ClaudeAgentOptions:
    """Create ClaudeAgentOptions with optional model override."""
    return ClaudeAgentOptions(
        max_turns=1,
        setting_sources=["user"],
        system_prompt={"type": "preset", "preset": "claude_code"},
        model=model,
    )


class ClientPool:
    """Pool of persistent ClaudeSDKClient instances for reduced latency.

    Maintains a pool of pre-connected clients that can be reused across requests.
    Uses /clear command between requests to reset conversation state while keeping
    the subprocess warm.
    """

    def __init__(self, size: int = 3, model: str | None = None):
        """Initialize client pool.

        Args:
            size: Number of clients to maintain in the pool.
            model: Model name for all clients in this pool (opus, sonnet, haiku).
        """
        self.size = size
        self.model = model
        self._available: asyncio.Queue[ClaudeSDKClient] = asyncio.Queue()
        self._all_clients: list[ClaudeSDKClient] = []
        self._initialized = False
        self._in_use = 0  # Track clients currently in use

    async def initialize(self) -> None:
        """Pre-spawn all clients and connect them."""
        if self._initialized:
            return

        model_label = self.model or "default"
        logger.info(f"[pool:{model_label}] Initializing pool with {self.size} clients...")
        for i in range(self.size):
            client = ClaudeSDKClient(make_options(self.model))
            await client.connect()
            self._all_clients.append(client)
            await self._available.put(client)
            logger.info(f"[pool:{model_label}] Client {i + 1}/{self.size} connected")

        self._initialized = True
        logger.info(f"[pool:{model_label}] Pool ready: {self.size} clients available")

    async def _clear_client_state(self, client: ClaudeSDKClient) -> None:
        """Clear conversation state using /clear command."""
        try:
            await client.query("/clear")
            # Drain the clear acknowledgment
            async for msg in client.receive_response():
                if isinstance(msg, ResultMessage):
                    break
        except Exception:
            # If /clear fails, the client might be in a bad state
            # We'll handle this in acquire by replacing the client
            raise

    async def _replace_client(self, old_client: ClaudeSDKClient) -> ClaudeSDKClient:
        """Replace an unhealthy client with a fresh one."""
        try:
            await old_client.disconnect()
        except Exception:
            pass  # Ignore disconnect errors

        if old_client in self._all_clients:
            self._all_clients.remove(old_client)

        new_client = ClaudeSDKClient(make_options(self.model))
        await new_client.connect()
        self._all_clients.append(new_client)
        return new_client

    def _log_status(self, action: str) -> None:
        """Log current pool status."""
        model_label = self.model or "default"
        available = self._available.qsize()
        total = len(self._all_clients)
        logger.info(f"[pool:{model_label}] {action} | in_use={self._in_use} available={available} total={total}")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[ClaudeSDKClient]:
        """Get a client from the pool, clear its state, yield it, return to pool.

        Yields:
            A ClaudeSDKClient ready for use with clean conversation state.
        """
        # Check if we'll need to wait
        model_label = self.model or "default"
        if self._available.empty():
            logger.warning(f"[pool:{model_label}] All {len(self._all_clients)} clients in use, request will wait...")

        client = await self._available.get()
        self._in_use += 1
        self._log_status("Acquired")
        client_to_return: ClaudeSDKClient | None = client

        try:
            # Clear conversation state before use
            await self._clear_client_state(client)
            yield client
        except Exception as e:
            # If something went wrong, replace the client
            model_label = self.model or "default"
            logger.warning(f"[pool:{model_label}] Client error, replacing: {e}")
            try:
                client_to_return = await self._replace_client(client)
                logger.info(f"[pool:{model_label}] Client replaced successfully")
            except Exception:
                # If replacement also fails, try creating a fresh client
                logger.warning(f"[pool:{model_label}] Replacement failed, creating fresh client...")
                try:
                    new_client = ClaudeSDKClient(make_options(self.model))
                    await new_client.connect()
                    self._all_clients.append(new_client)
                    client_to_return = new_client
                    logger.info(f"[pool:{model_label}] Fresh client created")
                except Exception:
                    # Can't create a new client - pool capacity reduced
                    logger.error(f"[pool:{model_label}] Failed to create client, pool capacity reduced")
                    client_to_return = None
            raise e
        finally:
            self._in_use -= 1
            if client_to_return is not None:
                await self._available.put(client_to_return)
            self._log_status("Released")

    async def shutdown(self) -> None:
        """Disconnect all clients and clean up."""
        model_label = self.model or "default"
        logger.info(f"[pool:{model_label}] Shutting down {len(self._all_clients)} clients...")
        for client in self._all_clients:
            try:
                await client.disconnect()
            except Exception:
                pass  # Ignore disconnect errors

        self._all_clients.clear()
        self._initialized = False
        self._in_use = 0

        # Clear the queue
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info(f"[pool:{model_label}] Shutdown complete")


class ModelPoolManager:
    """Manages multiple model-specific client pools.

    Creates and manages separate pools for each model type, enabling
    model selection without losing the performance benefits of pooling.
    """

    def __init__(self, pool_size: int = 3):
        """Initialize the pool manager.

        Args:
            pool_size: Number of clients per model pool.
        """
        self.pool_size = pool_size
        self._pools: dict[str, ClientPool] = {}
        self._lock = asyncio.Lock()

    async def get_pool(self, model: str) -> ClientPool:
        """Get or create a pool for the specified model.

        Args:
            model: Model name (opus, sonnet, haiku).

        Returns:
            Initialized ClientPool for the model.
        """
        async with self._lock:
            if model not in self._pools:
                logger.info(f"[manager] Creating pool for model: {model}")
                pool = ClientPool(size=self.pool_size, model=model)
                await pool.initialize()
                self._pools[model] = pool
            return self._pools[model]

    @asynccontextmanager
    async def acquire(self, model: str) -> AsyncIterator[ClaudeSDKClient]:
        """Acquire a client for the specified model.

        Args:
            model: Model name (opus, sonnet, haiku).

        Yields:
            A ClaudeSDKClient configured for the specified model.
        """
        pool = await self.get_pool(model)
        async with pool.acquire() as client:
            yield client

    async def shutdown(self) -> None:
        """Shutdown all pools."""
        logger.info(f"[manager] Shutting down {len(self._pools)} pools...")
        for model, pool in self._pools.items():
            await pool.shutdown()
        self._pools.clear()
        logger.info("[manager] All pools shut down")
