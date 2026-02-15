"""Smoke tests for demo entry points (import and minimal run with mocks)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


def test_demo_module_imports():
    """Main demo module can be imported without error."""
    import demo  # noqa: F401
    assert hasattr(demo, "main")
    assert hasattr(demo, "create_personalities")
    assert hasattr(demo, "run_situations")


async def test_demo_create_personalities_mocked(sample_kernel):
    """create_personalities returns kernels when create_personality is mocked."""
    from demo import create_personalities
    from unittest.mock import AsyncMock, patch

    with patch("demo.create_personality", new_callable=AsyncMock) as create_mock:
        create_mock.return_value = sample_kernel
        kernels = await create_personalities(n=2)
    assert len(kernels) == 2
    assert all(k.name == sample_kernel.name for k in kernels)
