from dataclasses import dataclass
from collections import deque

@dataclass
class Request:
    id : str
    prompt_len : int
    max_new_tokens : int
    arrival_time : int
    req_status: str ="nothing" #will change later according to status such as waitying, prefill , decode state
    prompt_done : int = 0
    generated_tokens : int =0

class Scheduler:
    def __init__(self,requests, max_batch_size, max_num_tokens):  # maxbatchsize is how many req, max num tokens is how many tokens schjeduler can handle per itertion
        self.requests= requests.sort(key=arrival_time)
        self.max_batch_size=max_batch_size
        self.max_num_tokens=max_num_tokens

        self.waiting = deque()
        self.decode=[]
        self.finished=[]
        self.time=0
        self.next_arrival =0

        #we will use this as arrivals which means at this point of time or lesser we will add to arrivals queue
    def arrivals(self,):
            while(self.next_arrival<len(requests)):
                request=self.requests[self.next_arrival]
                if request.arrival_time>self.time:
                    break
                self.waiting.append(request)
                request.req_status="waiting"
                self.next_arrival += 1

    def decode_function(self,token_budget, batch_budget):
        decode_work=[]
        next_decode=[]
        for i in self.decode:
            if batch_budget ==0 or token_budget ==0:
                next_decode.append(self.decode[i])
                self.decode[i].req_status="decode"
            req=i
            req.generated_tokens+=1
            decode_work.append(req.request_id)
            if req.max_new_tokens==req.generated_tokens:
                finished.append(req)
                req.req_status="finished"
            else:
                next_decode.append(req)
                req.req_status="decode"
        self.decode=next_decode
        return token_budget,batch,budget


    def prefill_function(self,token_budget, batch_budget):
            # chunk to trach if the promptlen> token budget
            # model does prefill 
            prefill_tracker=[]
            while(token_budget>0 and batch_budget>0 and len(self.waiting)>0):
                prompt_remaining=req.prompt_len-req.prompt_done
                if token_budget==0 or batch_budget ==0:
                    break
                req = waiting.popleft()
                req.req_status="prefill"
                chunk = min(prompt_remaining,token_budget)
                req.prompt_done = req.prompt_done+chunk
                batch_budget-=1
                token_budget-=chunk
                if prompt_remaining <=0:
                    self.decode.append(req)
                    req.req_status="decode"
                    prefill_tracker.append((req.id,prompt_remaining))
                else:
                    self.waiting.append(req)
                    req.req_status="waiting"
                
    def step(self):
            token_budget= self.max_num_tokens
            batch_budget = self.max_batch_size
            token_budget,batch_budget=self.decode_function(token_budget,batch_budget)
            token_budget,batch_budget=self.prefill_function(token_budget, batch_budget)

    def run(self):
        while(len(self.fininshed)<len(requests)):
            self.step()
            self.time+=1

def main():
    requests =[]
    scheduler=Scheduler(requests, 3, 8)
    scheduler.run()




