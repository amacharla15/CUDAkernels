class RequestState:
    def __init__(self, id : str, num_tokens : int , owned_blocks : list):
        self.id = id
        self.num_tokens = num_tokens
        self.owned_blocks=owned_blocks

class KVCacheManager:
    def __init__(self, num_blocks, tokens_per_block):
        self.num_blocks=num_blocks
        self.tokens_per_block=tokens_per_block
    
        self.request_map= {}

        #self.prefix_cache={}
        self.free_blocks=[]
        for i in range(0,num_blocks):
            self.free_blocks.append(i)

    def blocks_needed(self, tokens_number): # to check hwo many blocks we need (for both prefill or decode)
        needed_blocks= (tokens_number+self.tokens_per_block-1)//self.tokens_per_block
        return needed_blocks

    def requestallocator(self, request_id, initial_tokens): # during initial prefill
        temp_blocks=[]
        if request_id in self.request_map:
            return False
        needed_blocks = self.blocks_needed(initial_tokens)
        if needed_blocks> len(self.free_blocks):
            return False
        for i in range(0,needed_blocks):
            block_id= self.free_blocks.pop()
            temp_blocks.append(block_id)
        self.request_map[request_id]=RequestState(request_id, initial_tokens, temp_blocks)
        return True

    def ensure_capacity(self, request_id, target_tokens): #during decode

        state = self.request_map[request_id]
        needed_blocks=self.blocks_needed(target_tokens)
        more_blocks=needed_blocks- len(state.owned_blocks)
        if more_blocks==0:
            return True
        if more_blocks > len(self.free_blocks):
            return False
        for i in range(0,more_blocks):
            block_id=self.free_blocks.pop()
            state.owned_blocks.append(block_id)
        state.num_tokens = target_tokens
        return True

    def append_tokens(self, request_id, extra_tokens):
        state = self.request_map[request_id]
        target_tokens=extra_tokens+state.num_tokens
        return self.ensure_capacity(request_id, target_tokens)

    def append_one_token(self, request_id,one_token):
        return self.append_tokens(request_id, 1)

    def free_request(self, req_id): 
        state= self.request_map[req_id]
        for i in range(0,len(state.owned_blocks)):
            self.free_blocks.append(state.owned_blocks.pop())

    def block_table(self, request_id):
        state=self.request_map[req_id]
        for i in state.owned_blocks:
            print(state.owned_blocks)

    def physical_location(self, request_id, token_position):
        state=self.request_map[req_id]
        #logical block
        logical_block_location = token_position//self.tokens_per_block
        block=state[logical_block_location]
        offset = token_position%self.tokens_per_block

if __name__ == "__main__":
    cache=KVCacheManager(6, 4)
    print(cache.requestallocator("A", 3))
    print(cache.append_tokens("A", 6))


    


