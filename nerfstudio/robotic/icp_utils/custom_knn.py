import torch



def knn(ref, nbr, k=1, ord=2, dim=-1,top_k_dim=-2, largest=False, sorted=True):

    """
    input:ref (Tensor): reference points, shape (..., N, D)
    input:nbr (Tensor): query points, shape (..., M, D)
    input:k (int): number of nearest neighbors to return
    input:ord (int): the order of the norm
    input:dim (int): the dimension to calculate the norm
    input:top_k_dim (int): the dimension to return the top k neighbors

    return: dist (Tensor): distance to the k nearest neighbors, shape (..., M, k)
    
    example:
    ref = torch.rand(2, 10, 3)
    nbr = torch.rand(2, 5, 3)
    idx = knn(ref, nbr,top_k_dim=-1) # -1 is ref length, -2 is nbr length

    if I want to get the binding from ref to nbr, I can use -1 as top_k_dim
    if I want to get the binding from nbr to ref, I can use -2 as top_k_dim
    

    reference point should be mesh
    query point should be point cloud

    setting: every point cloud should only have one mesh vertex, but one mesh vertex can have multiple point cloud


    if i want to pass the mesh vertex deformation to point cloud, I can use -2 as top_k_dim
    if i want to pass the point cloud deformation to mesh vertex, I can use -1 as top_k_dim
    """
    diff = ref.unsqueeze(-2) - nbr.unsqueeze(-3)
    dist = torch.linalg.norm(diff, dim=dim, ord=ord)

    return dist.topk(k, dim=top_k_dim, largest=largest, sorted=sorted)


def knn_pt2mesh(ref,nbr,nbr_deformation, idx):
    """

    nbr_deformation: point cloud deformation, shape (..., N, 4,4) 
    ref_deformation: mesh deformation, shape (..., N, 4,4)
    idx: the index that has same shape of ref

    example:
    if we have nbr_deformation and idx, we can pass the deformation from point cloud to mesh vertex(ref_deformation)
    """


    # # need to test method
    # # Example tensors for ref_deformation and nbr_deformation

    # ref_deformation = torch.zeros_like(nbr_deformation)
    # nbr_deformation = torch.rand(10, 3, 3)  # Replace this with your actual tensor

    # # Assuming idx is a tensor containing indices
    # idx = torch.tensor([2, 4, 6, 8, 1, 3, 5, 7, 9, 0])  # Replace with your actual indices

    # # Flatten the tensors to use scatter
    # ref_deformation_flat = ref_deformation.view(-1)
    # nbr_deformation_flat = nbr_deformation.view(-1)
    # idx_flat = idx.unsqueeze(-1).expand_as(nbr_deformation)

    # # Calculate the indices to scatter
    # scatter_indices = idx_flat.view(-1)

    # # Scatter the values
    # ref_deformation_flat.scatter_(0, scatter_indices, nbr_deformation_flat)

    # # Reshape back to the original shape
    # ref_deformation = ref_deformation_flat.view_as(ref_deformation)

    # stable method
    ref_deformation=torch.zeros(len(ref),4,4)



    for i in range(len(idx)):
        ref_idx = i
        nbr_idx = idx[i]

        ref_deformation[ref_idx,:,:] = nbr_deformation[nbr_idx,:,]

    return ref_deformation




def knn_mesh2pt(ref,nbr,ref_deformation, idx):
    """

    nbr_deformation: point cloud deformation, shape (..., N, 4,4) 
    ref_deformation: mesh deformation, shape (..., N, 4,4)
    idx: the index that has same shape of nbr

    example:
    if we have ref_deformation and idx, we can pass the deformation from mesh to point cloud(nbr_deformation)
    """

    # stable method
    nbr_deformation=torch.zeros(len(nbr),4,4)



    for i in range(len(idx)):
        ref_idx = idx[i]
        nbr_idx = i

        nbr_deformation[nbr_idx,:,]=ref_deformation[ref_idx,:,:] 

    return ref_deformation