class Request:
    id : str
    initial_tokens : int 
    prompt_done : int  # chunked prefill 
    prompt_len : int 
    arrival_time : int

class Scheduler:
    def __init(self, batch_budget,token_budget, requests): #assuming requests are sorted
        self.requests = requests 

        self.time : int 
        self.waiting : deque()


    def arrivals(self):

    def decode_function():

    def prefill_function():

    def step():

    def run():

if __name__== "__main__":

    scheduler= Scheduler()