from dataclasses import dataclass, field
from collections import deque


@dataclass
class Request:
    request_id: str
    arrival_time: int
    prompt_len: int #number of tokens 
    max_new_tokens: int

    # Progress fields
    prompt_done: int = 0
    generated_tokens: int = 0

    # State: waiting / prefill / decode / finished
    state: str = "waiting"

    # Metrics
    first_token_time: int | None = None
    finish_time: int | None = None


class Scheduler:
    def __init__(self, requests: list[Request], max_batch_size: int, max_num_tokens: int):
        self.requests = sorted(requests, key=lambda r: r.arrival_time)

        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens

        self.time = 0
        self.next_arrival = 0

        self.waiting = deque()
        self.decode = []
        self.finished = []

        self.trace = []

    def can_allocate(self, request: Request, tokens: int) -> bool:
        """
        Fake KV-cache check.

        In real TensorRT-LLM / vLLM style serving:
        - prefill needs KV space for prompt tokens
        - decode needs KV space for one new generated token

        For now, always return True.
        """
        return True

    def admit_arrivals(self) -> list[str]:
        """
        Move all requests whose arrival_time <= current time
        into the waiting queue.
        """
        arrivals = []

        while self.next_arrival < len(self.requests):
            req = self.requests[self.next_arrival]

            if req.arrival_time > self.time:
                break

            req.state = "waiting"
            self.waiting.append(req)
            arrivals.append(req.request_id)

            self.next_arrival += 1

        return arrivals

    def schedule_decode(self, token_budget: int, batch_slots: int):
        """
        Decode-first scheduling.

        Each active decode request can generate exactly 1 token
        in this iteration.

        Returns:
            updated token_budget
            updated batch_slots
            list of decode work done
        """
        decode_work = []
        next_decode = []

        i = 0
        while i < len(self.decode):
            req = self.decode[i]

            if batch_slots == 0:
                # No more requests can run this iteration.
                # Keep this request and the rest for future decode.
                while i < len(self.decode):
                    next_decode.append(self.decode[i])
                    i += 1
                break

            if token_budget == 0:
                # No token budget left.
                # Keep this request and the rest for future decode.
                while i < len(self.decode):
                    next_decode.append(self.decode[i])
                    i += 1
                break

            if not self.can_allocate(req, 1):
                # KV cache full or unavailable.
                # Do not schedule this request this iteration.
                next_decode.append(req)
                i += 1
                continue

            # Schedule one decode token.
            req.generated_tokens += 1
            token_budget -= 1
            batch_slots -= 1

            decode_work.append(req.request_id)

            # First generated token metric.
            if req.first_token_time is None:
                req.first_token_time = self.time

            # Check whether request finished.
            if req.generated_tokens == req.max_new_tokens:
                req.state = "finished"
                req.finish_time = self.time
                self.finished.append(req)
            else:
                req.state = "decode"
                next_decode.append(req)

            i += 1

        self.decode = next_decode

        return token_budget, batch_slots, decode_work

    def schedule_prefill(self, token_budget: int, batch_slots: int):
        """
        Schedule prefill using leftover token budget and batch slots.

        Prefill can be chunked:
            chunk = min(prompt_remaining, token_budget)

        If prompt is complete after this chunk:
            waiting/prefill -> decode

        Else:
            request goes back to waiting queue.
        """
        prefill_work = []

        while batch_slots > 0 and token_budget > 0 and len(self.waiting) > 0:
            req = self.waiting.popleft()

            prompt_remaining = req.prompt_len - req.prompt_done

            if prompt_remaining <= 0:
                # Defensive case: prompt already done.
                req.state = "decode"
                self.decode.append(req)
                continue

            chunk = min(prompt_remaining, token_budget)

            if not self.can_allocate(req, chunk):
                # Fake KV check failed.
                # In a real scheduler, maybe try another request or evict KV blocks.
                # For now, put it back and stop prefill to avoid infinite rotation.
                self.waiting.appendleft(req)
                break

            # Process prompt chunk.
            req.state = "prefill"
            req.prompt_done += chunk

            token_budget -= chunk
            batch_slots -= 1

            prefill_work.append((req.request_id, chunk))

            # If full prompt is processed, request becomes decode-ready.
            if req.prompt_done == req.prompt_len:
                req.state = "decode"
                self.decode.append(req)
            else:
                # Chunked prefill not done.
                # Put it back for future prefill.
                req.state = "waiting"
                self.waiting.append(req)

        return token_budget, batch_slots, prefill_work

    def step(self):
        """
        Run one scheduler iteration.
        """
        arrivals = self.admit_arrivals()

        token_budget = self.max_num_tokens
        batch_slots = self.max_batch_size

        token_budget_before = token_budget

        # 1. Decode first.
        token_budget, batch_slots, decode_work = self.schedule_decode(
            token_budget,
            batch_slots,
        )

        # 2. Prefill with remaining budget.
        token_budget, batch_slots, prefill_work = self.schedule_prefill(
            token_budget,
            batch_slots,
        )

        tokens_used = token_budget_before - token_budget

        trace_row = {
            "time": self.time,
            "arrivals": arrivals,
            "decode": decode_work,
            "prefill": prefill_work,
            "tokens_used": tokens_used,
            "waiting": self.get_waiting_ids(),
            "decode_list": self.get_decode_ids(),
            "finished": self.get_finished_ids(),
        }

        self.trace.append(trace_row)

        self.time += 1

    def jump_if_idle(self):
        """
        If there is no active work but future requests exist,
        jump time directly to the next arrival.

        This avoids wasting empty iterations.
        """
        if len(self.waiting) > 0:
            return

        if len(self.decode) > 0:
            return

        if self.next_arrival >= len(self.requests):
            return

        next_time = self.requests[self.next_arrival].arrival_time

        if next_time > self.time:
            self.time = next_time

    def run(self):
        """
        Run until all requests are finished.
        """
        while len(self.finished) < len(self.requests):
            self.jump_if_idle()
            self.step()

        return self.finished, self.trace

    def get_waiting_ids(self):
        ids = []
        for req in self.waiting:
            ids.append(req.request_id)
        return ids

    def get_decode_ids(self):
        ids = []
        for req in self.decode:
            ids.append(req.request_id)
        return ids

    def get_finished_ids(self):
        ids = []
        for req in self.finished:
            ids.append(req.request_id)
        return ids

    def print_trace(self):
        print()
        print("SCHEDULER TRACE")
        print("-" * 90)

        for row in self.trace:
            print(f"t={row['time']}")
            print(f"  arrivals:     {row['arrivals']}")
            print(f"  decode:       {row['decode']}")
            print(f"  prefill:      {row['prefill']}")
            print(f"  tokens_used:  {row['tokens_used']}")
            print(f"  waiting:      {row['waiting']}")
            print(f"  decode_list:  {row['decode_list']}")
            print(f"  finished:     {row['finished']}")
            print("-" * 90)

    def print_metrics(self):
        print()
        print("REQUEST METRICS")
        print("-" * 90)

        total_output_tokens = 0
        first_arrival = None
        last_finish = None

        for req in self.requests:
            total_output_tokens += req.max_new_tokens

            if first_arrival is None or req.arrival_time < first_arrival:
                first_arrival = req.arrival_time

            if last_finish is None or req.finish_time > last_finish:
                last_finish = req.finish_time

            ttft = req.first_token_time - req.arrival_time

            if req.max_new_tokens <= 1:
                tpot = None
            else:
                tpot = (req.finish_time - req.first_token_time) / (req.max_new_tokens - 1)

            print(f"Request {req.request_id}")
            print(f"  arrival_time:      {req.arrival_time}")
            print(f"  prompt_len:        {req.prompt_len}")
            print(f"  max_new_tokens:    {req.max_new_tokens}")
            print(f"  first_token_time:  {req.first_token_time}")
            print(f"  finish_time:       {req.finish_time}")
            print(f"  TTFT:              {ttft}")
            print(f"  TPOT:              {tpot}")
            print()

        total_time = last_finish - first_arrival + 1
        throughput = total_output_tokens / total_time

        print("GLOBAL METRICS")
        print(f"  total_output_tokens: {total_output_tokens}")
        print(f"  total_time:          {total_time}")
        print(f"  output throughput:   {throughput:.2f} tokens / iteration")
        print("-" * 90)


def build_toy_requests():
    requests = [
        Request(request_id="A", arrival_time=0, prompt_len=6, max_new_tokens=3),
        Request(request_id="B", arrival_time=0, prompt_len=2, max_new_tokens=4),
        Request(request_id="C", arrival_time=1, prompt_len=5, max_new_tokens=2),
        Request(request_id="D", arrival_time=2, prompt_len=1, max_new_tokens=2),
    ]

    return requests


def main():
    requests = build_toy_requests()

    scheduler = Scheduler(
        requests=requests,
        max_batch_size=3,
        max_num_tokens=8,
    )

    scheduler.run()
    scheduler.print_trace()
    scheduler.print_metrics()


if __name__ == "__main__":
    main()