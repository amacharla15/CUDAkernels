@dataclass
class request:
    id : str
    prompt_len : int

gpu_blocks=[] #entire gpu blocks of gpu 
free_blocks={} #we add all the free blocks indices to this hashmap to track


def blocks_needed(num_of_tokens : int, block_limit : int):
    num_blocks= (num_of_tokens//block_limit)+1
    return num_blocks

def allocate_request(request):
    if request.id in requestmap:
        return "this request id already exists"
    needed = blocks_needed(request.prompt_len, 5)
    owned_blocks=[]
    temp=0
    for i in free_blocks:
        if temp == needed:
            break
        if free_blocks[i] == "free":
            owned_blocks.append(free_blocks[i])
            free_blocks[i]="not free":
            temp=temp+1
    







