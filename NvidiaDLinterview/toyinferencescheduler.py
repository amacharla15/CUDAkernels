from dataclasses import dataclass
from collections import deque

@dataclass
class Request:
    id: str
    arrival_time: int
    prompt_len: int
    max_new_tokens: int

    prompt_done: int = 0
    generated_tokens: int = 0
    state: str = "waiting"   # waiting / decode / finished

    first_token_time: int | None = None
    finish_time: int | None = None


def run_scheduler(requests, max_batch_size, max_num_tokens):
    time = 0

    waiting = deque()
    decode = []
    finished = []

    # sort by arrival time
    requests = sorted(requests, key=lambda r: r.arrival_time)
    next_arrival = 0

    trace = []

    while len(finished) < len(requests):
        # TODO 1:
        # Add all requests whose arrival_time <= time into waiting queue.
        # Increase next_arrival as you add them.
        while next_arrival < len(requests) and requests[next_arrival].arrival_time <= time:
            waiting.append(requests[next_arrival])
            next_arrival += 1

        token_budget = max_num_tokens
        request_budget = max_batch_size

        decode_scheduled = []
        prefill_scheduled = []

        # TODO 2:
        # Schedule decode requests first.
        # For each request in decode:
        #   if request_budget > 0 and token_budget > 0:
        #       schedule 1 token
        #       reduce budgets
        #       remember it in decode_scheduled
        for request in decode:
            if request_budget == 0 or token_budget == 0:
                break
            decode_scheduled.append(request)
            token_budget -= 1
            request_budget -= 1

        # TODO 3:
        # Schedule prefill requests second.
        # For each waiting request:
        #   remaining_prompt = prompt_len - prompt_done
        #   chunk = min(remaining_prompt, token_budget)
        #   schedule chunk
        #   reduce budgets
        #   if prompt_done reaches prompt_len:
        #       move it to decode

        while waiting and request_budget > 0 and token_budget > 0:
            request = waiting.popleft()

            remaining_prompt = request.prompt_len - request.prompt_done
            chunk = min(remaining_prompt, token_budget)

            request.prompt_done += chunk
            token_budget -= chunk
            request_budget -= 1

            prefill_scheduled.append((request.id, chunk))

            if request.prompt_done == request.prompt_len:
                request.state = "decode"
                decode.append(request)
            else:
                waiting.appendleft(request)




        # TODO 4:
        # Apply decode updates:
        #   generated_tokens += 1
        #   if first_token_time is None:
        #       first_token_time = time
        #   if generated_tokens == max_new_tokens:
        #       state = "finished"
        #       finish_time = time + 1
        #       move to finished

        

        # TODO 5:
        # Save one trace row:
        # time, decode_scheduled, prefill_scheduled, token usage, waiting ids, decode ids, finished ids

        time += 1

    total_generated = sum(r.max_new_tokens for r in requests)
    total_iterations = time

    metrics = {
        "throughput": total_generated / total_iterations,
        "requests": {
            r.id: {
                "TTFT": r.first_token_time - r.arrival_time,
                "latency": r.finish_time - r.arrival_time,
                "first_token_time": r.first_token_time,
                "finish_time": r.finish_time,
            }
            for r in requests
        }
    }

    return trace, metrics