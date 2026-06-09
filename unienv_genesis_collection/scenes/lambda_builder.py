"""Generic scene builder that delegates construction to a user-supplied callable."""

from typing import Callable, Optional

from unienv_genesis import GenesisWorld
from unienv_interface.world.node import WorldNode


class LambdaBuilder(WorldNode):
    """Minimal generic scene builder.

    Wraps an arbitrary ``build_fn`` callable so that custom scene elements can
    be plugged into the standard reload lifecycle without writing a full
    :class:`WorldNode` subclass.

    During ``reload(priority=...)`` the builder calls::

        build_fn(world, name, **build_kwargs)

    whenever ``priority`` is in :pyattr:`reload_priorities`, and stores the
    return value on both :pyattr:`result` and :pyattr:`entity`.

    Args:
        world: GenesisWorld instance.
        name: Name identifier for this builder.
        build_fn: Callable invoked during reload.  Must accept at least
            ``(world, name)`` and may accept additional keyword arguments
            supplied via ``**build_kwargs``.
        reload_priority: Priority value at which :meth:`reload` triggers the
            build.  Defaults to ``100``.
        **build_kwargs: Extra keyword arguments forwarded to *build_fn*.

    Raises:
        TypeError: If *build_fn* is not callable.

    Example:
        >>> def make_box(world, name, size=1.0):
        ...     return world.scene.add_entity(gs.morphs.Box(size=size), name=name)
        >>> builder = LambdaBuilder(world, name="box", build_fn=make_box, size=0.5)
    """

    has_reward = False
    has_termination_signal = False
    has_truncation_signal = False

    def __init__(
        self,
        world: GenesisWorld,
        name: str = "lambda_builder",
        build_fn: Callable = None,
        reload_priority: int = 100,
        **build_kwargs,
    ):
        if not callable(build_fn):
            raise TypeError(
                f"build_fn must be callable, got {type(build_fn).__name__}"
            )

        self.world: GenesisWorld = world
        self._name = name
        self.build_fn = build_fn
        self.reload_priorities = {reload_priority}
        self.build_kwargs = build_kwargs

        self.result = None
        self.entity = None

    @property
    def name(self) -> str:
        return self._name

    def reload(
        self,
        *,
        priority: int = 0,
        seed: Optional[int] = None,
        mask=None,
        **kwargs,
    ):
        if priority in self.reload_priorities:
            self.result = self.build_fn(self.world, self._name, **self.build_kwargs)
            self.entity = self.result

    def get_observation(self):
        return None
