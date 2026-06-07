"""Utilities for handling batched vs unbatched tensor shapes in Genesis.

When batch_size is None (unbatched), Genesis returns tensors without a batch dimension.
These utilities help handle both cases consistently.
"""

from typing import Optional, Tuple, Union, Sequence
import torch

__all__ = [
    "is_batched",
    "try_scalar",
    "get_batch_aware_shape",
    "broadcast_to_batch",
    "index_batch_dimension",
    "reduce_and",
    "reduce_or",
]

# Type alias for index types supported by index_batch_dimension
IndexType = Union[
    int,                    # Single index: 5
    slice,                  # Slice: slice(1, 10)
    Sequence[int],          # List/tuple of indices: [1, 3, 5] or (1, 3, 5)
    torch.Tensor,           # Tensor of indices: torch.tensor([1, 3, 5])
    Tuple[Union[int, slice, Sequence[int], torch.Tensor], ...],  # Multi-dim indices
]


def is_batched(batch_size: Optional[int]) -> bool:
    """Check if the environment is batched.
    
    Args:
        batch_size: Batch size, or None for unbatched.
        
    Returns:
        True if batch_size is a positive integer, False if None.
    """
    return batch_size is not None and batch_size > 0

def try_scalar(data : torch.Tensor) -> Union[int, float, torch.Tensor]:
    """Check if the data is batched or unbatched, and return a scalar if unbatched.
    """
    if len(data.shape) == 0:
        return data.item()
    else:
        return data

def get_batch_aware_shape(
    batch_size: Optional[int], 
    dims: Union[int, Sequence[int]]
) -> Tuple[int, ...]:
    """Get shape tuple for tensors, conditionally including batch dimension.
    
    Args:
        batch_size: Batch size, or None for unbatched.
        dims: Feature dimension(s). Can be a single int or a sequence of ints.
        
    Returns:
        Shape tuple with batch dimension if batched, otherwise just dims.
        
    Example:
        >>> get_batch_aware_shape(4, 7)  # batched, single dim
        (4, 7)
        >>> get_batch_aware_shape(None, 7)  # unbatched, single dim
        (7,)
        >>> get_batch_aware_shape(4, (3, 4))  # batched, multiple dims as tuple
        (4, 3, 4)
        >>> get_batch_aware_shape(4, [3, 4])  # batched, multiple dims as list
        (4, 3, 4)
    """
    # Normalize dims to a tuple
    if isinstance(dims, int):
        dims_tuple = (dims,)
    else:
        dims_tuple = tuple(dims)
    
    if is_batched(batch_size):
        return (batch_size,) + dims_tuple
    return dims_tuple


def broadcast_to_batch(
    scalar_or_array: Union[float, torch.Tensor],
    batch_size: Optional[int],
    n_features: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Broadcast scalar or 1D array to appropriate shape for batched/unbatched.
    
    Args:
        scalar_or_array: Scalar value or 1D/2D tensor.
        batch_size: Batch size, or None for unbatched.
        n_features: Number of features (e.g., DOFs).
        device: PyTorch device.
        
    Returns:
        Tensor with shape (batch_size, n_features) if batched,
        or (n_features,) if unbatched.
        
    Example:
        >>> broadcast_to_batch(1.0, 4, 7)  # batched
        tensor([[1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1.]])
        >>> broadcast_to_batch(1.0, None, 7)  # unbatched
        tensor([1., 1., 1., 1., 1., 1., 1.])
    """
    if isinstance(scalar_or_array, float):
        if is_batched(batch_size):
            return torch.full((batch_size, n_features), scalar_or_array, device=device)
        else:
            return torch.full((n_features,), scalar_or_array, device=device)
    
    # It's a tensor
    tensor = scalar_or_array
    
    if is_batched(batch_size):
        # Batched case: ensure shape is (batch_size, n_features)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0).expand(batch_size, -1)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    else:
        # Unbatched case: ensure shape is (n_features,)
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor


def index_batch_dimension(
    tensor: torch.Tensor, 
    index: IndexType, 
    batch_size: Optional[int]
) -> torch.Tensor:
    """Index into a tensor, handling both batched and unbatched cases.
    
    For batched tensors (shape: (batch_size, D1, D2, ...)): 
        - Prepends ':' to the index tuple to keep batch dimension
        - e.g., index=5 -> tensor[:, 5], index=(1, 2) -> tensor[:, 1, 2]
    
    For unbatched tensors (shape: (D1, D2, ...)):
        - Uses the index directly
        - e.g., index=5 -> tensor[5], index=(1, 2) -> tensor[1, 2]
    
    Args:
        tensor: Input tensor, either (batch_size, ...) or (...)
        index: Index to select. Can be:
            - int: Single index (e.g., 5)
            - slice: Slice object (e.g., slice(1, 10))
            - Sequence[int]: List/tuple of indices (e.g., [1, 3, 5])
            - torch.Tensor: Tensor of indices (e.g., torch.tensor([1, 3, 5]))
            - Tuple: Multi-dimensional indices (e.g., (1, slice(None), 2))
        batch_size: Batch size, or None for unbatched.
        
    Returns:
        Indexed tensor.
        
    Examples:
        >>> # Batched tensor shape: (4, 10, 3)
        >>> index_batch_dimension(tensor, 5, 4)  # -> shape: (4, 3)
        >>> index_batch_dimension(tensor, (5, 1), 4)  # -> shape: (4,)
        >>> index_batch_dimension(tensor, slice(2, 8), 4)  # -> shape: (4, 6, 3)
        
        >>> # Unbatched tensor shape: (10, 3)
        >>> index_batch_dimension(tensor, 5, None)  # -> shape: (3,)
        >>> index_batch_dimension(tensor, (5, 1), None)  # -> shape: ()
        >>> index_batch_dimension(tensor, slice(2, 8), None)  # -> shape: (6, 3)
    """
    # Normalize index to a tuple
    if isinstance(index, tuple):
        index_tuple = index
    else:
        index_tuple = (index,)
    
    if is_batched(batch_size):
        # Batched: prepend slice(None) to keep batch dimension
        full_index = (slice(None),) + index_tuple
    else:
        # Unbatched: use index as-is
        full_index = index_tuple
    
    return tensor[full_index]


def reduce_and(*values):
    """Logical AND across a list of bool / Tensor values."""
    result = values[0]
    for v in values[1:]:
        if isinstance(result, torch.Tensor) or isinstance(v, torch.Tensor):
            result = torch.logical_and(
                result,
                v
            )
        else:
            result = result and v
    return result

def reduce_or(*values):
    """Logical OR across a list of bool / Tensor values."""
    result = values[0]
    for v in values[1:]:
        if isinstance(result, torch.Tensor) or isinstance(v, torch.Tensor):
            result = torch.logical_or(
                result,
                v
            )
        else:
            result = result or v
    return result
