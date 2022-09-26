import torch
import torch.nn as nn


class KnnMemory(nn.Module):
    def __init__(self, memory_size=8192, dim=512) -> None:
        super().__init__()

        self.K = memory_size
        # create the queue
        self.register_buffer("queue", torch.randn(dim, memory_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        if len(keys.shape) == 3:
            keys = torch.flatten(keys, 0, 1)

        # gather keys before updating queue
        if torch.distributed.is_initialized():
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
    
    def forward(self, x, topk=100):
        r"""
        x: the query item with shape [B, N, dim]
            where B is the batch size, N is the sequence length
            dim is the dimension of the embedding
        topk: the nearest k items in the memory

        return:
            sampled features from the memory, with shape [B, N, topk, dim]
        """
        sim_matrix = torch.einsum('bnc,ck->bnk', [x, self.queue.clone().detach()])

        _, topk_inds = torch.topk(sim_matrix, k=topk, dim=-1)
        sampled_features = self.queue[:, topk_inds.view(-1)]
        sampled_features = sampled_features.view(-1, *(topk_inds.shape))
        sampled_features = sampled_features.permute(1, 2, 3, 0)
        
        self._dequeue_and_enqueue(x)

        return sampled_features, topk_inds


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    # test the function
    B = 4
    N_query = 32
    embed_dim = 512
    q = torch.randn(B, N_query, embed_dim).cuda()

    knn_memory = KnnMemory(memory_size=8192, dim=embed_dim).cuda()

    sampled_knn_features, sampled_knn_inds = knn_memory(q)
    print(q.shape, sampled_knn_features.shape, sampled_knn_inds.shape)
    