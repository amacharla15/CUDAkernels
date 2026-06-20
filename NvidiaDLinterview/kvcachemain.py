from dataclasses import dataclass, field


@dataclass
class RequestState:
    request_id: str
    num_tokens: int = 0
    blocks: list[int] = field(default_factory=list)


class KVCacheManager:
    def __init__(self, num_blocks: int, tokens_per_block: int):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        if tokens_per_block <= 0:
            raise ValueError("tokens_per_block must be positive")

        self.num_blocks = num_blocks
        self.tokens_per_block = tokens_per_block

        # Free physical KV block IDs.
        self.free_blocks = list(range(num_blocks))

        # request_id -> RequestState
        self.request_map = {}

    def blocks_needed(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0

        return (num_tokens + self.tokens_per_block - 1) // self.tokens_per_block

    def allocate_request(self, request_id: str, initial_tokens: int) -> bool:
        if request_id in self.request_map:
            raise ValueError("request already exists")

        if initial_tokens < 0:
            raise ValueError("initial_tokens cannot be negative")

        needed_blocks = self.blocks_needed(initial_tokens)

        if needed_blocks > len(self.free_blocks):
            return False

        owned_blocks = []

        for _ in range(needed_blocks):
            block_id = self.free_blocks.pop()
            owned_blocks.append(block_id)

        self.request_map[request_id] = RequestState(
            request_id=request_id,
            num_tokens=initial_tokens,
            blocks=owned_blocks,
        )

        return True

    def ensure_capacity(self, request_id: str, target_tokens: int) -> bool:
        if request_id not in self.request_map:
            raise ValueError("unknown request")

        if target_tokens < 0:
            raise ValueError("target_tokens cannot be negative")

        state = self.request_map[request_id]

        if target_tokens < state.num_tokens:
            raise ValueError("target_tokens cannot be smaller than current tokens")

        current_blocks = len(state.blocks)
        needed_blocks = self.blocks_needed(target_tokens)

        extra_blocks = needed_blocks - current_blocks

        if extra_blocks <= 0:
            state.num_tokens = target_tokens
            return True

        if extra_blocks > len(self.free_blocks):
            return False

        for _ in range(extra_blocks):
            block_id = self.free_blocks.pop()
            state.blocks.append(block_id)

        state.num_tokens = target_tokens
        return True

    def append_tokens(self, request_id: str, num_new_tokens: int) -> bool:
        if request_id not in self.request_map:
            raise ValueError("unknown request")

        if num_new_tokens < 0:
            raise ValueError("num_new_tokens cannot be negative")

        state = self.request_map[request_id]
        target_tokens = state.num_tokens + num_new_tokens

        return self.ensure_capacity(request_id, target_tokens)

    def append_one_token(self, request_id: str) -> bool:
        return self.append_tokens(request_id, 1)

    def free_request(self, request_id: str) -> None:
        if request_id not in self.request_map:
            return

        state = self.request_map.pop(request_id)

        for block_id in state.blocks:
            self.free_blocks.append(block_id)

    def get_block_table(self, request_id: str) -> list[int]:
        if request_id not in self.request_map:
            raise ValueError("unknown request")

        return list(self.request_map[request_id].blocks)

    def get_physical_location(self, request_id: str, token_position: int):
        if request_id not in self.request_map:
            raise ValueError("unknown request")

        state = self.request_map[request_id]

        if token_position < 0 or token_position >= state.num_tokens:
            raise IndexError("token_position out of range")

        block_index = token_position // self.tokens_per_block
        offset = token_position % self.tokens_per_block

        physical_block_id = state.blocks[block_index]

        return physical_block_id, offset

    def stats(self):
        used_blocks = self.num_blocks - len(self.free_blocks)

        return {
            "num_blocks": self.num_blocks,
            "used_blocks": used_blocks,
            "free_blocks": len(self.free_blocks),
            "tokens_per_block": self.tokens_per_block,
            "active_requests": len(self.request_map),
        }


cache = KVCacheManager(num_blocks=5, tokens_per_block=4)

print(cache.stats())

print(cache.allocate_request("A", initial_tokens=6))
print(cache.get_block_table("A"))
print(cache.stats())

print(cache.append_one_token("A"))
print(cache.append_one_token("A"))
print(cache.get_block_table("A"))
print(cache.stats())

print(cache.append_one_token("A"))
print(cache.get_block_table("A"))
print(cache.stats())

print(cache.get_physical_location("A", token_position=5))

cache.free_request("A")
print(cache.stats()) 