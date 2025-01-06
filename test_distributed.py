import torch
import torch.distributed as dist

def main():
    # 1) Initialize the process group (using GLOO to avoid single-GPU conflicts)
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank {rank}/{world_size}] initialized.")

    # 2) Each rank has its "own" tensor
    if rank == 0:
        # Rank 0 holds the value 17
        val0 = torch.tensor([17], dtype=torch.int32)
        # We'll receive a value from rank 1 into val1
        val1 = torch.zeros(1, dtype=torch.int32)
        # Asynchronously send val0 to rank 1
        send_req = dist.isend(tensor=val0, dst=1)
        # Asynchronously receive val1 from rank 1
        recv_req = dist.irecv(tensor=val1, src=1)
        
        # 3) Wait for completion
        send_req.wait()
        recv_req.wait()
        
        # 4) Compute sum
        sum_val = val0.item() + val1.item()
        print(f"Rank {rank}: Received val1={val1.item()}, sum={sum_val}")

    else:
        # Rank 1 holds the value 13
        val1 = torch.tensor([13], dtype=torch.int32)
        # We'll receive a value from rank 0 into val0
        val0 = torch.zeros(1, dtype=torch.int32)
        # Asynchronously send val1 to rank 0
        send_req = dist.isend(tensor=val1, dst=0)
        # Asynchronously receive val0 from rank 0
        recv_req = dist.irecv(tensor=val0, src=0)
        
        # 3) Wait for completion
        send_req.wait()
        recv_req.wait()
        
        # 4) Compute sum
        sum_val = val0.item() + val1.item()
        print(f"Rank {rank}: Received val0={val0.item()}, sum={sum_val}")

    # 5) Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
