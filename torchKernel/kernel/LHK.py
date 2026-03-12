"""
torch kernel calculation

Author: Daniel Deidda

Copyright 2024 National Physical Laboratory.

SPDX-License-Identifier: Apache-2.0
"""


import torch
import torch.nn as nn
import math
import gc
import numpy as np
from torchKernel.utils.system import check_reserved_memory, check_pytorch_gpu, clear_pytorch_cache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

class kernelise_image(torch.autograd.Function):
      
    @staticmethod
    def forward(ctx, im, a, im_templ, recon):
        ctx.org_shape=im.shape
        ctx.input=im
        ctx.current_alpha=a
        ctx.im_template = im_templ
        ctx.recon=recon
        ctx.save_for_backward(im)

        input_np = ctx.input.detach().cpu().numpy()
        cur_alpha = ctx.current_alpha.detach().cpu().numpy()

        if (ctx.input.dim()==4):
            in_image_sirf=ctx.im_template.fill(input_np[0,...])
            ctx.alpha_sirf=ctx.im_template.clone().fill(cur_alpha[0,...])
        else:
            in_image_sirf=ctx.im_template.fill(input_np[0,0,...])
            ctx.alpha_sirf=ctx.im_template.clone().fill(cur_alpha[0,0,...])

        ka_np=ctx.recon.compute_kernelised_image(in_image_sirf,ctx.alpha_sirf).as_array()
        ka=torch.from_numpy(ka_np).requires_grad_().to(device).reshape(im.shape)
    
        return ka
    
    @staticmethod
    def backward(ctx, grad_out) :
        im = ctx.saved_tensors
        
        if (ctx.current_alpha.dim()==4):
            grad_sirf=ctx.im_template.fill(grad_out[0,...].clone().detach().cpu().numpy())
        else:
            grad_sirf=ctx.im_template.fill(grad_out[0,0,...].clone().detach().cpu().numpy())

        kgrad_np=ctx.recon.compute_kernelised_image(grad_sirf,ctx.alpha_sirf).as_array()
        kgrad=torch.from_numpy(kgrad_np).requires_grad_().to(device).reshape(ctx.input.shape)

        #out=torch.autograd.grad(ctx.ka,(a,im))
        return kgrad, None, None, None
    
def set_KOSMAPOSL(recon, obj_fun, anatomical, curr_alpha, h, nghb, sm, sp, is2d, dsm=5):
    dsp=dsm
    recon.set_objective_function(obj_fun)
    recon.set_anatomical_prior(anatomical)
    recon.set_num_non_zero_features(1)
    recon.set_num_neighbours(nghb)
    recon.set_sigma_m(sm)
    recon.set_sigma_p(sp)
    recon.set_sigma_dm(dsm)
    recon.set_sigma_dp(dsp)
    recon.set_only_2D(is2d)
    recon.set_hybrid(h)

    sirf_at = anatomical.clone()
    
    if (curr_alpha.dim()==4):
        sirf_at.fill(curr_alpha[0,...].detach().cpu().numpy())
    else:
        sirf_at.fill(curr_alpha[0,0,...].detach().cpu().numpy())
    recon.set_current_estimate(sirf_at)
    recon.set_up(sirf_at)   
    
class BuildK(nn.Module):
    def __init__(self, sigma_m, is_voxelised, save_mem_k, spacing=None, is_iterative=0, sigma_p=None, sigma_dm=None, isHybrid=False, isDeepK=False):
        super(BuildK, self).__init__()
        self.sigma_m = sigma_m
        self.sigma_p = sigma_p if sigma_p else sigma_m
        self.sigma_dm = sigma_dm if sigma_dm else 5
        self.is_voxelised = is_voxelised
        self.save_mem_k = save_mem_k
        self.spacing = spacing
        self.is_iterative = is_iterative
        self.isHybrid = isHybrid
        self.isDeepK=isDeepK
        # STIR uses sigma_m, sigma_p, sigma_dm for anatomical, functional, and distance weighting
        self.ksigma = nn.Parameter(torch.tensor([[[[float(sigma_m)]]]], device=device))



       
    def extract_neighbour_indices(self,imageSize,w):
        """
        extract vector containg the indeces of the neighbourhood, w x w x w, of each voxel in an image of dimension 
        imageSize
        """
        assert w%2!=0, "The Neighbourhood window needs to be odd"

        h = imageSize[len(imageSize)-3]
        m = imageSize[len(imageSize)-2]
        n = imageSize[len(imageSize)-1]

        
        wlen = 2*np.floor(w/2)
        widx = xidx = yidx = torch.arange(-wlen/2,wlen/2+1)

        if h==1:
            zidx = [0]
            nN = w*w
        else:
            zidx = widx
            nN = w*w*w

        Z,Y,X = torch.meshgrid(torch.arange(0,h), torch.arange(0,n), torch.arange(0,m), indexing='ij')              
        N = torch.zeros((n*m*h, nN), dtype=torch.int)
        dz, dy, dx = self.spacing
        distances = torch.zeros(N.shape)
        l = 0
        for z in zidx:
            Znew = self.setBoundary(Z + z,h)
            for y in yidx:
                Ynew = self.setBoundary(Y + y,m)
                for x in xidx:
                    Xnew = self.setBoundary(X + x,n)
                    distances[:,l] = torch.sqrt((z * dz)**2 + (y * dy)**2 + (x * dx)**2)
                    N[:,l] = ((Xnew + (Ynew)*n + (Znew)*n*m)).reshape(-1)
                    l += 1
        return N, nN, distances

    def setBoundary(self,idX,num_x):
        """Boundary conditions along index idX with number of elements num_x."""
        idx = idX<0
        idX[idx] = idX[idx] + num_x
        idx = idX>num_x-1
        idX[idx] = idX[idx] - num_x
        return torch.flatten(idX)
    
    def get_knn(self, input, w, k, test=None):
        """
        use knn to find distances between voxel values in the neighbourhood and save a matrix (tensor)
        for the differences and a matrix for thheir relative indeces
        """


        ID,nN,distances = self.extract_neighbour_indices(input.shape,w)
        # input= input.detach().cpu()
        ID=ID.to(device)
        distances = distances.to(device)

        sizes=input.shape
        anat_v=input.flatten()
        
        # NN = anat_v[ID]
        # v =anat_v.reshape(-1,1)
        # Compute absolute intensity differences between central voxel and neighbours
        dist=abs(anat_v[ID]-anat_v.reshape(-1,1))


        #  Sort by intensity difference
        if (not self.is_voxelised):
            sorted_dist, indices = torch.sort(dist, dim=-1,stable=True,descending=False)
            del sorted_dist
            torch.cuda.empty_cache()

            # print( ID.shape)
            # Map sorted indices back to original voxel indices
            # the following is done because indices are based on the new dist tensor which lost the info that we had from ID
            sorted_indices = ID.gather(dim=1, index=indices.type(dtype=torch.int64).to(device))


            # Remap physical distances to match sorted neighbour order
            sorted_distances = distances.gather(dim=1, index=indices)
            self.knn_distances = sorted_distances[:, :k]
        else:
            self.knn_distances = distances
            sorted_indices = ID


        if test == None:
            del ID
            torch.cuda.empty_cache()

        matrix_shape= (1,1,input.shape[len(sizes)-3]*input.shape[len(sizes)-2]*input.shape[len(sizes)-1], k)
        # ID = ID.reshape((1,1,input.shape[len(sizes)-3]*input.shape[len(sizes)-2]*input.shape[len(sizes)-1],nN))
        # keep only knn
        new_NN = anat_v[sorted_indices]
        # W, D,IDs = new_NN[:, 0:k], dist[:, 0:k], sorted_indices[:, 0:k]
        W,IDs = new_NN[:, 0:k], sorted_indices[:, 0:k]


        # D=D.reshape(matrix_shape)
        W=W.reshape(matrix_shape)
        IDs=IDs.reshape(matrix_shape)
        # self.knn_distances = self.knn_distances.reshape(matrix_shape)


        if test!= None:
            return W, IDs, ID.reshape((1,1,input.shape[len(sizes)-3]*input.shape[len(sizes)-2]*input.shape[len(sizes)-1],nN)) #sr for sorted distance
        else:
            return W, IDs
        
    def get_features_STIR_like(self, input,w, functional_input=None):
        # STIR-aligned weight formula usefull when wanted to have voxel-wise kernel operations
        # Kw = torch.zeros((W.shape[2], W.shape[3]), device=device)
        
        sigma_anat = torch.std(input)  # global std of anatomical image

        # # print("The standard dev of the anatomical image is ",sigma_anat)

        # for i in range(W.shape[2]):

        #     # if W[0, 0, i, 0]==0:
        #     #     Kw[i, :]=1
        #     #     continue
            
        #     anat_diff = (W[0, 0, i, :] - W[0, 0, i, 0])/sigma_anat
        #     anat_term = torch.square(anat_diff / (math.sqrt(2) * self.sigma_m))
        #     dist_term = torch.square(self.knn_distances[i, :] / ( self.sigma_dm))

        #     if self.isHybrid and functional_W is not None:
        #         ref_value = W[0, 0, i, 0].clamp(min=1e-8)  # central voxel value for functional normalization
        #         func_diff = (functional_W[0, 0, i, :] - functional_W[0, 0, i, 0]) / ref_value
        #         func_term = torch.square(func_diff / (math.sqrt(2) * self.sigma_p))
        #     else:
        #         func_term = 0
        #     Kw[i, :] = torch.exp(-(anat_term + func_term + dist_term)).clamp(min=1e-12)
        # # Normalize rows to sum to 1 (STIR normalization)
        # Kw /= Kw.sum(dim=1, keepdim=True)
        # return Kw

        # Extract fixed neighborhood indices and distances
        ID, nN, distances = self.extract_neighbour_indices(input.shape, w)
        ID = ID.to(device)
        distances = distances.to(device)

        # Flatten anatomical and functional images
        anat_v = input.flatten()  # shape: (num_voxels,)
        functional_v = functional_input.flatten() if (self.isHybrid and functional_input is not None) else None

        # Gather neighborhood values for all voxels
        neigh_anat = anat_v[ID]  # shape: (num_voxels, num_neighbors)
        center_anat = anat_v.unsqueeze(1)  # shape: (num_voxels, 1)

        # Anatomical term
        anat_term = ((neigh_anat - center_anat)/sigma_anat / (math.sqrt(2) * self.sigma_m)) ** 2

        # Distance term
        dist_term = (distances / self.sigma_dm) ** 2  # shape: (num_voxels, num_neighbors)

        # Functional term (if hybrid)
        if functional_v is not None:
            neigh_func = functional_v[ID]
            center_func = functional_v.unsqueeze(1)
            func_term = ((neigh_func - center_func)/center_func.clamp(min=1e-8) / (math.sqrt(2) * self.sigma_p)) ** 2
        else:
            func_term = 0

        # Compute Gaussian weights for all voxels
        weights = torch.exp(-(anat_term + func_term + dist_term))

        # Normalize weights row-wise
        weights /= weights.sum(dim=1, keepdim=True)

        # Store kernel weights and neighborhood indices
        # Reshape Kw and ID for compatibility with existing kernelise functions
        self.Kw = weights.unsqueeze(0).unsqueeze(0)  # shape: (1,1,num_voxels,num_neighbors)
        self.ID = ID.unsqueeze(0).unsqueeze(0)       # shape: (1,1,num_voxels,num_neighbors)

        return self.Kw



    def get_K_save_mem_STIR_like(self,input, w,  functional_input=None):
        self.Kw = self.get_features_STIR_like(input, w, functional_input)
        

    def get_features(self,W,ID,ksigma):

        with torch.no_grad():
            Kw = torch.zeros((W.shape[2], W.shape[3]), device=device)
            id=torch.tensor(range(W.shape[2]),  dtype=torch.int64, device=device).reshape(1,1,W.shape[2],1)

            id= id.expand(W.shape).reshape(1,1,W.shape[2],W.shape[3],1)
            idx= torch.cat([id, ID.reshape(1,1,W.shape[2],W.shape[3],1)],4)
            del id
            gc.collect()
            torch.cuda.empty_cache()
            ID_shape2=ID.shape[2]
            # check_pytorch_gpu()     

            for i in range(W.shape[2]):
                
                sigma=torch.std(W[0,0,i,:])
                # print("sigma is ",sigma)

                if( sigma== 0 and not self.is_voxelised):        
                    Kw[i,:] = torch.zeros(W.shape[3], device=device)
                else:
                    if (self.is_voxelised):
                        sigma=torch.std(W)
                        norm = W[0,0,i,0]-W[0,0,ID[0,0,i,:],0]
                        # if (self.is_iterative==1): # this is the case of hybrid K
                            
                        #     # Kw[i, :] = torch.where(
                        #     #     W[0, 0, i, :] != 0,
                        #     #     -torch.square((norm / W[0, 0, i, 0]) / (math.sqrt(2) * ksigma)),
                        #     #     torch.zeros_like(W[0, 0, i, :])
                        #     #     )

                        #     if W[0, 0, i, 0] != 0:
                        #         value = -torch.square((norm / W[0, 0, i, 0].clamp(min=1e-8)) / (math.sqrt(2) * ksigma))
                        #         Kw[i, :] = value
                        #     else:
                        #         continue

                        #     # print(f"[Hybrid Kernel] W sum: {torch.sum(W).item()}, "
                        #     #   f"Kw sum: {torch.sum(Kw).item()}")

                        # else: 

                        Kw[i,:]=(-torch.square((norm/sigma)/(math.sqrt(2)*ksigma)))
                            
                    else:
                        norm = torch.nn.functional.pairwise_distance(W[0,0,i,:],W[0,0,ID[0,0,i,:],:],p=2)
                        Kw[i,:]=(-torch.square((norm/sigma)/(math.sqrt(2)*ksigma)))

        Kw = torch.nn.functional.softmax(Kw, dim=1)

        return Kw, idx
                

    def get_K_save_mem(self,W,ID,ksigma):

            Kw, idx = self.get_features(W,ID,ksigma)
            self.Kw=Kw
            self.ID=ID
    
    def get_K(self,W,ID,ksigma):

        Kw, idx = self.get_features(W,ID,ksigma)
                   
        clear_pytorch_cache()
        # check_pytorch_gpu()     
        # implementation of sparse_coo is apparently very slow in gpu so sending operation on cpu
        idxl=idx.reshape(Kw.numel(),2)#.to(device)
        ID=ID.reshape(Kw.shape)
        Kwl=Kw.reshape(Kw.numel())#.to(device)
        # print('### before sparse_coo creation',idxl.device,(Kwl.device))
        # check_pytorch_gpu()
        K = torch.sparse_coo_tensor(list(zip(*idxl)), Kwl, device=device, requires_grad=False)#(ID_shape2,ID_shape2),

        # print('###after kernel calculation the kernel is in ',Kw.device )
        # check_pytorch_gpu()  
        self.Kw=Kw  
        self.ID=ID

        return K
    
    #apply kernel to image
    def kernelise_image(self,K,a):
        org_shape=a.shape
        # K=K.to(device)
        return  torch.sparse.mm(K,a.reshape(a.numel(),1)).reshape(org_shape)#torch.mv(K,a.reshape(a.numel())).reshape(org_shape)#.to(torch.float32)
    
    def kernelise_image(self, Kw, ID, a):
        """
        Apply kernel weights Kw to image a using neighbor indices ID.
        Equivalent to K * a without creating a sparse matrix.
        """

        # Flatten image to [num_pixels]
        flat = a.reshape(-1)                     # [P]

        # Gather neighbor values: shape [P, K]
        vals = flat[ID.reshape(-1, ID.shape[-1])]

        # Multiply by weights and sum: shape [P]
        out = (Kw.reshape(-1, Kw.shape[-1]) * vals).sum(dim=1)

        # Reshape back to the original image shape
        return out.reshape(a.shape)
    
    def kernelise_image_t(self, Kw, ID, b):
        """
        Compute the adjoint of the kernel operator K^T b without sparse matrices.

        Kw : [P, K]    weights
        ID : [P, K]    neighbor indices
        b  : [P]       vector being backprojected

        returns: [P]
        """

        # 1. Expand b to match Kw (broadcasted j → j,k)
        #    shape: [P, K]
        b_expanded = b.unsqueeze(1).expand_as(Kw)

        # 2. Compute contribution for each (j,k)
        #    contrib[j,k] = Kw[j,k] * b[j]
        contrib = Kw * b_expanded              # [P, K]

        # 3. Scatter-add to accumulate contributions at ID[j,k]
        #    adj[i] += contrib[j,k]  where i = ID[j,k]
        P = Kw.shape[0]
        adj = torch.zeros(P, device=Kw.device)

        adj = adj.scatter_add(0, ID.reshape(-1), contrib.reshape(-1))

        return adj
    
    #apply kernel to image
    def kernelise_image_save_mem(self,a):

        org_shape=a.shape
        kim = a.clone().detach().reshape(a.numel())
        a_v = a.reshape(a.numel())
        NNa=a_v[self.ID]
        NNa = NNa.reshape(self.Kw.shape)
        # for i in range(a.numel()):
        #     kim[i] = torch.dot(self.Kw[i,:],NNa[0,0,i,:])
        # K=K.to(device)

        # the following calculates  the dot product 
        # of each corresponding  row in Kw and NNa
        kim = torch.einsum("ij,ij->i", self.Kw.squeeze(), NNa.squeeze())
        #kim = torch.einsum("ij,ij->i", self.Kw, NNa)#self.Kw@NNa.transpose(0, 1)
        return  kim.reshape(org_shape)#torch.mv(K,a.reshape(a.numel())).reshape(org_shape)#.to(torch.float32)
    
    def kernelise_image_save_mem_t(self,a):
        
        """
        Compute K.T @ a using memory-saving approach with vectorized scatter_add.
        Equivalent to torch.sparse.mm(K.T, a).
        """
            
        org_shape=a.shape
        a_v = a.reshape(-1)
        num_voxels = a_v.shape[0]

        # Flatten all contributions

        flat_ID = self.ID.flatten().long()  # Ensure int64  # target indices (columns in K)
        flat_rows = torch.arange(self.ID.shape[2]).repeat_interleave(self.ID.shape[3])  # source rows
        # print('row_size',self.Kw.shape)
        flat_weights = self.Kw.flatten()
        flat_values = a_v[flat_rows] * flat_weights  # contributions

        # print('weights',flat_weights.shape)

        # print('rvalues',flat_values.shape)

        # Accumulate contributions
        output = torch.zeros_like(a_v)
        # basicallylly to reproduce K^T*a instead of multiplying each values of the neighbourhood for the associated Kij
        # we  multiply the voxel values in  i the indeces for the relevant weights but then scatter_add makes sure that where
        # a voxel appears as a neighbour is accumulated in the relevant index position of output
        output = output.scatter_add(0, flat_ID, flat_values)

        return output.reshape(org_shape)
    
    def deepK_forward(self, net_W, k,w, functional_input=None):
        ID,nN,distances = self.extract_neighbour_indices(input.shape,w)
        dim1,dim2,dim3 = net_W.shape[1], net_W.shape[2], net_W.shape[3] #(C,Jx,Jy) C = channels
        J,K = ID.shape
        
        W = net_W.contiguous().view(dim1, dim2 * dim3)
        W = W.t()  # dim: (J=Jx*Jy, C)

        #  1. Gather neighbor feature vectors:
        nn_W = W[ID]      #shape: (J, K, C)

        # 2. Expand reference features for voxel j, W[j] to (J, K, C)
        ref_W = W.unsqueeze(1)                     # shape: (J, 1, C)

        # 3. Compute squared vector differences 
        diff = ref_W - nn_W                 # shape: (J, K, C)
        sq_diff = diff ** 2

        #  4. Mean over feature channels C, sqrt, and negate
        D = -torch.sqrt(torch.mean(sq_diff, dim=2) + self.eps)   # shape: (J, K)

        # 5. Softmax across neighbors
        Kw = torch.softmax(D, dim=1)

        return Kw, ID


    def forward(self, input, k,w, functional_input=None):
        ## compute the weight
        # input = input1.to(input1.device)
        # print('###before kernel calculation')
        # check_pytorch_gpu()

        functional_W = None
        
        clear_pytorch_cache()

        if self.save_mem_k and not self.isHybrid :
            W, ID = self.get_knn(input,w,k)
            self.get_K_save_mem(W,ID,self.ksigma)

        elif self.isHybrid :
            # assert functional_input is None, "The input for the functional kernel is None, you need to pass it to use a hybrid kernel"
            # functional_W, _ = self.get_knn(functional_input, w, k)
            self.get_K_save_mem_STIR_like(input, w, functional_input)

        else:
            W, ID = self.get_knn(input,w,k)
            return self.get_K(W,ID,self.ksigma)

    
