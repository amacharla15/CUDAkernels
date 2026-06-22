from dataclasses import dataclass

@dataclass
class RequestState:
    id : str
    initial_tokens : int =0
    blocks_owned : list = []

class KVCacheManager:
    def __init__(self, num_blocks, block_limit):
        self.num_blocks=num_blocks
        self.block_limit=block_limit
        self.free_blocks=list(range(num_blocks))
        self.request_map = {} #no duplicate allocation , tracks request / request state 


    def blocks_needed_per_request(self, num_tokens):
        return (self.block_limit+num_tokens-1)//self.block_limit


    def allocate_request(self,request_id, initial_tokens):
        blocks_needed=self.blocks_needed_per_request(initial_tokens)
        temporary_block_ids=[]
        if blocks_needed>len(self.free_blocks):
            return False
        else:
            for i in range(0,blocks_needed):
                block_id= self.free_blocks.pop()
                temporary_block_ids.append(block_id)
            self.request_map[request_id]=RequestState(request_id, initial_tokens, temporary_block_ids)
        return True

    def ensure_capacity(self, request_id, target_tokens):
        blocks_needed= self.blocks_needed_per_request(target_tokens+self.request_map[request_id].initial_tokens)
        if blocks_needed == len(self.request_map[request_id].blocks_owned):
            return True
        if blocks_needed> len(self.free_blocks):
            return False
        for i in range(0,blocks_needed-len(self.request_map[request_id].temporary_block_ids)):
            block_id=self.free_blocks.pop()
            self.request_map[request_id].temporary_block_ids.append(block_id)
        self.request_map[request_id]+=target_tokens
        return True

    # useful during speculative decoding
    def append_tokens(self, request_id, target_tokens):
        state = self.request_map[request_id]
        return self.ensure_capacity(request_id, target_tokens)
    
    def append_one_token(self, request_id: str) -> bool:
        return self.append_tokens(request_id, 1)

    def free_request(self,request_id):
        state = self.request_map.pop(request_id)
        for block_id in state.blocks_owned:
            self.free_blocks.append(block_id)

    def get_block_table(self, request_id: str) -> list[int]:
        return list(self.request_map[request_id].blocks_owned)

    def get_physical_location():
        
        

        
                
                
                
        





