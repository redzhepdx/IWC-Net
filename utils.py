import tensorflow as tf
import numpy as np
import cv2
from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_gaussian, unary_from_softmax, create_pairwise_bilateral, unary_from_labels
from skimage.segmentation import relabel_sequential
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse import csr_matrix
import time

def root_min_square_loss(y_pred, y_true):
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred))))

def distance(pos_i, pos_j):
    return np.sum(np.square(np.subtract(pos_i, pos_j)))

def min_square_error(y_pred, y_true):
    return tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))

def ReconstructionLoss(reconstructed_img, real_img):
    return min_square_error(reconstructed_img, real_img)

#E(X) = unary + gaussian_pairwise, CRF
def CRF(img, probabilities, K):
    #print("Probabilities shape : " , probabilities.shape)
    processed_probabilities = probabilities.squeeze()
    #print("Processed : " , processed_probabilities.shape)
    softmax                 = processed_probabilities.transpose((2, 0, 1))
    #print(softmax.shape)
    unary                   = unary_from_softmax(softmax)
    #print(unary.shape)
    unary                   = np.ascontiguousarray(unary)
    #print(unary.shape)
    d                       = dcrf.DenseCRF(img.shape[0] * img.shape[1], K)

    d.setUnaryEnergy(unary)
    #d.addPairwiseGaussian(sxy=3, compat=3)
    feats                   = create_pairwise_gaussian(sdims=(3, 3), shape=(img.shape[1], img.shape[0]))
    #feats                   = create_pairwise_bilateral(sdims=(5, 5), schan=(10, 10, 10), img=img.reshape(img.shape[1], img.shape[0], 3), chdim=2)
    #print("Feats : \n", feats)
    d.addPairwiseEnergy(feats, compat=5, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)

    res = np.argmax(Q, axis=0).reshape(img.shape[1], img.shape[0]).astype('float32')
    res *= (255. / res.max())
    res.reshape(img.shape[:2])
    #print("Res \n", res)
    return res

def CRF_N(img, gt_image):
    labels = relabel_sequential(gt_image)[0].flatten()
    #print(labels)

    M = labels.max() + 1

    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    GT_PROB = 0.5

    u_energy = -np.log(1.0 / M)
    n_energy = -np.log((1.0 - GT_PROB) / (M - 1))
    #print(n_energy.shape)
    p_energy = -np.log(GT_PROB)

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    #print(U[:, labels > 0].shape)
    U[:, labels > 0] = n_energy
    U[labels, np.arange(U.shape[1])] = p_energy
    U[:, labels == 0] = u_energy
    d.setUnaryEnergy(U)

    #d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)

    res = np.argmax(d.inference(5), axis=0).astype('float32')

    res *= 255 / res.max()
    res = res.reshape(img.shape[:2])

    return res.astype(np.uint8)

def CRF_MASTER(img, anno_rgb):
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    HAS_UNK = 0 in colors

    if HAS_UNK:
        print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
        print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]

    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

    print("Using 2D specialized functions")

    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
    d.setUnaryEnergy(U)

    #d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseBilateral(sxy=(5, 5), srgb=(3, 3, 3), rgbim=img, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)

    MAP = np.argmax(Q, axis=0)

    MAP = colorize[MAP,:]

    return MAP.reshape(img.shape)

#TODO OWT-HC
def H_Clustering():
    pass

#TODO gPb
def Probability_Boundary(edged_X_prime):
    pass

def compute_dissimilarity(img, x, y, n_x, n_y, ro_I_2, ro_X_2):
    pixel_distance = distance(img[x, y], img[n_x, n_y])
    position_distance = distance(np.array([x, y]), np.array([n_x, n_y]))
    return np.exp(-pixel_distance / ro_I_2) * np.exp(-position_distance / ro_X_2)

#Creation of Dissimilarity Matrix(Tensor Representation)
def create_dissimilarity_matrix(img, r=5, ro_I=10, ro_X=4):
    shape = img.shape
    dissim_matrix = np.zeros((shape[1], shape[2], shape[1], shape[2]))
    ro_I_2 = ro_I * ro_I
    ro_X_2 = ro_X * ro_X
    for i in range(shape[1]):
        for j in range(shape[2]):
            for off_i in range(-r, r + 1, 1):
                for off_j in range(-r, r + 1, 1):
                    n_i = i + off_i
                    n_j = j + off_j
                    if n_j >= 0 and n_j < shape[1] and n_i >= 0 and n_i < shape[2]:
                        dissim_matrix[i, j, n_i, n_j] = compute_dissimilarity(img[0], i, j, n_i, n_j, ro_I_2, ro_X_2)
    return dissim_matrix

#Creation of Dissimilarity Matrix(Sparse Matrix Graph Representation)
def create_sparse_dissimilary_matrix(img, r=5, ro_I=10, ro_X=4):
    shape   = img.shape
    indices = []
    values  = []
    for i in range(shape[1]):
        for j in range(shape[2]):
            for off_i in range(-r, r + 1, 1):
                for off_j in range(-r, r + 1, 1):
                    n_i = i + off_i
                    n_j = j + off_j
                    if n_j >= 0 and n_j < shape[1] and n_i >= 0 and n_i < shape[2]:
                        main_pixel_index = i * shape[2] + j
                        neig_pixel_index = n_i * shape[2] + n_j
                        dissimilarty_val = compute_dissimilarity(img[0], i, j, n_i, n_j, ro_I, ro_X)
                        indices.append([main_pixel_index, neig_pixel_index])
                        values.append(dissimilarty_val)
    return indices, values

#Normal and memory safe version of soft normalized cut loss but it is slow
def compute_loss(raw_data, volume, K, ro_I = 10, ro_X = 4, r = 5):
    shape = raw_data.shape
    v_shape = volume.shape
    total = 0
    for k in range(K):
        assoc_A = 0
        assoc_V = 0
        for i in range(shape[1]):
            for j in range(shape[2]):
                partial_assoc_A = 0
                partial_assoc_V = 0
                for off_i in range(-r, r + 1, 1):
                    for off_j in range(-r, r + 1, 1):
                        n_i = i + off_i
                        n_j = j + off_j
                        if n_j >= 0 and n_j < shape[1] and n_i >= 0 and n_i < shape[2]:
                            dissim = compute_dissimilarity(raw_data[0], i, j, n_i, n_j, ro_I, ro_X)
                            partial_assoc_A += (dissim * volume[n_i, n_j, k])
                            partial_assoc_V += dissim
                assoc_A += (volume[i, j, k] * partial_assoc_A)
                assoc_V += (volume[i, j, k] * partial_assoc_V)
        total += (assoc_A / assoc_V)
    return K - total


#TODO : Tensorflow Speed Up Version of N-Cut Loss
def n_cut_loss(volume, dissim_matrix, K):
    total     = tf.cast(tf.Variable(0), dtype=tf.float32)
    k_tensor  = tf.cast(tf.Variable(K), dtype=tf.float32)
    for step in range(K):
        total = tf.add(total,
                    tf.div(tf.reduce_sum(
                        tf.multiply(volume[0, :, :, step], tf.reduce_sum(
                            tf.multiply(dissim_matrix, volume[0, :, :, step]), [0,1]))),
                    tf.reduce_sum(
                        tf.multiply(volume[0, :, :, step], tf.reduce_sum(dissim_matrix, [0,1])))))
    return tf.subtract(k_tensor, total)

def n_cut_loss_sp(volume, dissim_matrix, K):
    total     = tf.cast(tf.Variable(0), dtype=tf.float32)
    k_tensor  = tf.cast(tf.Variable(K), dtype=tf.float32)
    for step in range(K):
        w_mult_pv = tf.sparse_tensor_dense_matmul(dissim_matrix, tf.reshape(volume[0, :, :, step], [-1]))
        total = tf.add(total, tf.reduce_sum(tf.multiply(volume[0, :, :, step], w_mult_pv)))

    return tf.subtract(k_tensor, total)

def PostProcess(U_Enc_x):
    x_1 = CRF(U_Enc_x)
    x_2 = Probability_Boundary(x_1)
    S   = H_Clustering(x_2)
    return S

def pairwise_dist_image(A, B):
    with tf.variable_scope('pairwise_dist'):
        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(A), 1)
        nb = tf.reduce_sum(tf.square(B), 1)

        # na as a row and nb as a co"lumn vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])

        # return pairwise euclidead difference matrix
        D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
        return D

def positional_sparse_matrix(row_size, col_size, radius):
    #p2 means euclidean distance
    nn   = NearestNeighbors(radius=radius, p=2)

    rows = np.arange(row_size)
    cols = np.arange(col_size)

    mesh_grid         = np.empty((row_size, row_size, 2), dtype=np.intp)
    mesh_grid[..., 0] = rows[:, None]
    mesh_grid[..., 1] = cols
    #print("MESH_GRID : \n", mesh_grid)
    mesh_grid         = mesh_grid.reshape(-1, 2)
    #print("MESH_GRID_RESHAPE : \n", mesh_grid)
    nn.fit(mesh_grid)

    pos_sparse_matrix = nn.radius_neighbors_graph(mesh_grid, radius=radius, mode='distance')

    return pos_sparse_matrix

def dissim_matrix_values(A, positions, pos_distances, ro_I_2=100, ro_X_2=16, shape=(224*224, 224*224)):
    vap_i = A[positions[... , 0]]
    #print("vap_i : \n", vap_i)
    vap_j = A[positions[... , 1]]
    #print("vap_j : \n", vap_j)
    #dissimilarity_values = np.exp(-np.sum(np.square(np.subtract(vap_i, vap_j))) / ro_I_2) * np.exp(-np.sum(np.square(pos_distances)) / ro_X_2).astype(np.float32)

    '''IMAGE VALUES'''
    diff                    = np.subtract(vap_i, vap_j)
    #print("diff : \n", diff)
    sq_diff                 = np.square(diff)
    #print("sq_diff : \n", sq_diff)

    sum_sq_diff             = np.sum(sq_diff, axis=1)
    #print("Sum sq diff : \n", sum_sq_diff)

    neg_sum_sq_diff         = np.negative(sum_sq_diff)
    #print("neg_sum_sq_diff : \n", neg_sum_sq_diff)
    div_neg_sum_sq_diff     = np.divide(neg_sum_sq_diff, ro_I_2)
    #print("div_neg_sum_sq_diff : \n", div_neg_sum_sq_diff)
    exp_div_neg_sum_sq_diff = np.exp(div_neg_sum_sq_diff)
    #print("exp_neg_sum_sq_diff : \n", exp_div_neg_sum_sq_diff)

    '''SPATIAL POSITION VALUES'''
    sq_pos                  = np.square(pos_distances).astype(np.float32)
    #print("Sq POs : \n", sq_pos)
    sum_sq_pos              = np.sum(sq_pos, 1)
    #print("sum_Sq POs : \n", sum_sq_pos)
    neg_sum_sq_pos          = np.negative(sum_sq_pos)
    #print("neg_sum_sq_pos : \n", neg_sum_sq_pos)
    div_neg_sum_sq_pos      = np.divide(neg_sum_sq_pos, ro_X_2)
    #print("div_neg_sum_sq_pos : \n", div_neg_sum_sq_pos)
    exp_div_neg_sum_sq_pos  = np.exp(div_neg_sum_sq_pos)
    #print("exp_div_neg_sum_sq_pos : \n", exp_div_neg_sum_sq_pos)

    '''DISSIMILARITY'''
    dissimilarity_values_tensor   = np.multiply(exp_div_neg_sum_sq_diff, exp_div_neg_sum_sq_pos)
    #print("dissim : \n", dissimilarity_values_tensor)
    #input()
    sparse_matrix                 = csr_matrix((dissimilarity_values_tensor, (positions[..., 0], positions[..., 1])), shape)
    sparse_matrix.setdiag(np.ones((224 * 224)))
    return dissimilarity_values_tensor, sparse_matrix


def ncut_loss_np(volume_tensors, images, positions, pos_distances, K, batch_size):
    total_loss = 0
    for m in range(batch_size):
        total = 0
        curr = np.reshape(images[m], [-1, 3])
        #print("CUR : \n", curr.shape)
        _, dissim_sparse_tensor = dissim_matrix_values(curr, positions, pos_distances)
        #print("sparse : \n", dissim_sparse_tensor.shape)
        for i in range(K):
            vol_tensor = np.reshape(volume_tensors[m, :, :, i], [-1, 1])
            spvm  = dissim_sparse_tensor * vol_tensor
            print("SPVM : ", spvm.T.shape)
            #mult = np.multiply(vol_tensor, spvm)
            mult = np.matmul(spvm.T, vol_tensor)
            print("MULT : ", mult.shape)
            assoc_A = np.sum(mult)

            #sum_sparse = np.sum(dissim_sparse_tensor, axis=1)
            #mult_sparse = np.multiply(vol_tensor, sum_sparse)
            #print("Vol : \n", vol_tensor.shape)
            mult_sparse = dissim_sparse_tensor * vol_tensor
            #print(mult_sparse.shape)
            assoc_V     = np.sum(mult_sparse)
            #print("Assoc_A : {} assoc_V : {}".format(assoc_A, assoc_V))
            total = np.add(total, np.divide(assoc_A, assoc_V))

        loss_per_image = np.subtract(K, total)
        print("Loss : ", loss_per_image)
        total_loss     = np.add(total_loss, loss_per_image)
    print(total_loss)
    return total_loss


def dissim_matrix_values_tf(image, positions, pos_distances, ro_I_2=100, ro_X_2=16, shape=(224*224, 224*224)):
    vap_i = tf.gather(image, positions[... , 0])
    vap_j = tf.gather(image, positions[... , 1])

    '''IMAGE VALUES'''
    diff                    = tf.subtract(vap_i, vap_j)
    sq_diff                 = tf.square(diff)
    sum_sq_diff             = tf.reduce_sum(sq_diff, 1)
    neg_sum_sq_diff         = tf.negative(sum_sq_diff)
    div_neg_sum_sq_diff     = tf.div(neg_sum_sq_diff, ro_I_2)
    exp_div_neg_sum_sq_diff = tf.exp(div_neg_sum_sq_diff)

    '''SPATIAL POSITION VALUES'''
    sq_pos                  = tf.cast(tf.square(pos_distances), dtype=tf.float32)
    sum_sq_pos              = tf.reduce_sum(sq_pos, 1)
    neg_sum_sq_pos          = tf.negative(sum_sq_pos)
    div_neg_sum_sq_pos      = tf.div(neg_sum_sq_pos, ro_X_2)
    exp_div_neg_sum_sq_pos  = tf.exp(div_neg_sum_sq_pos)

    '''DISSIMILARITY'''
    dissimilarity_values_tensor   = tf.multiply(exp_div_neg_sum_sq_diff, exp_div_neg_sum_sq_pos)

    return tf.SparseTensor(positions, dissimilarity_values_tensor, shape)


def ncut_loss_tf(volume_tensors, images, positions, pos_distances, K, batch_size, total_loss, total):
    #total_loss = tf.Variable(0, dtype=tf.float32)
    tf.assign(total_loss, 0)
    for m in range(batch_size):
        #total = tf.Variable(0, dtype=tf.float32)
        total = tf.assign(tota, 0.)
        curr = tf.reshape(images[m], [-1, 3])
        dissim_sparse_tensor = dissim_matrix_values_tf(curr, positions, pos_distances)
        for i in range(K):
            vol_tensor  = tf.reshape(volume_tensors[m, :, :, i], [-1, 1])
            spvm        = tf.sparse_tensor_dense_matmul(dissim_sparse_tensor, vol_tensor)
            #assoc_A     = tf.reduce_sum(tf.multiply(vol_tensor, spvm))
            assoc_A     = tf.multiply(tf.transpose(vol_tensor), spvm)
            #print(assoc_A.get_shape())
            sum_sparse  = tf.sparse_reduce_sum(dissim_sparse_tensor, 1, keep_dims=True)
            #print(sum_sparse.get_shape())
            mult_sparse = tf.multiply(vol_tensor, sum_sparse)
            #print(mult_sparse.get_shape())
            assoc_V     = tf.reduce_sum(mult_sparse)
            #print(assoc_V.get_shape())
            total = tf.add(total, tf.div(assoc_A, assoc_V))

        loss_per_image = tf.subtract(K, total)
        total_loss     = tf.add(total_loss, loss_per_image)
    return tf.div(total_loss, batch_size)

def weight_matrix(images, radius, ro_I_2=100, ro_X_2=16, shape=(224*224, 224*224)):
    shape  = images.shape
    images = images.reshape([shape[0], shape[3], shape[1], shape[2]])
    print(images)
    shape  = images.shape
    dissim = np.zeros((shape[0], shape[1], shape[2], shape[3], (radius - 1) * 2 + 1, (radius - 1) * 2 + 1))
    print(dissim.shape)
    padded = np.pad(images, ((0, 0), (0, 0), (radius - 1, radius - 1), (radius - 1, radius - 1), ), 'constant')
    print("pad : \n", padded)
    print(padded.shape)
    for m in range(2 * (radius - 1) + 1):
        for n in range(2 * (radius - 1) + 1):
            dissim[:, :, :, :, m, n] = np.subtract(images, padded[:, :, m:shape[2] + m, n:shape[3] + n])
            print("m : {} n : {}".format(m,n))
            print("img : \n", images)
            print("pad : \n", padded[:, :, m:shape[2] + m, n:shape[3] + n])
            print("Dissim : \n", dissim[:, :, :, :, m, n])
            input()

            #print(padded[:, :, m:shape[2] + m, n:shape[3] + n].shape)
    #print(  dissim[0, 0, 0, 0, 0, 0])
    print(dissim)
    dissim = np.exp(-np.power(dissim,2).sum(1, keepdims = True) / ro_I_2)
    dist   = np.zeros((2 * (radius - 1) + 1, 2 * (radius - 1) + 1))

    for m in range(1-radius, radius):
        for n in range(1-radius,radius):
            if m**2 + n**2 < radius**2:
                dist[m + radius - 1,n + radius - 1] = np.exp(-( m**2 + n**2) / ro_X_2**2)

    res = np.multiply(dissim, dist)
    return res

def ncut_loss_non_sparse_tf(volume_tensors, weights, radius, batch_size):
    volume_tensors = tf.reshape(volume_tensors, [batch_size, volume_tensors.get_shape()[3], volume_tensors.get_shape()[1], volume_tensors.get_shape()[2]])
    K              = tf.constant(volume_tensors.get_shape()[1], dtype=tf.float32)
    cropped_seg    = tf.zeros(volume_tensors.get_shape()[0], volume_tensors.get_shape()[1], volume_tensors.get_shape()[2], volume_tensors.get_shape()[3], (radius - 1) * 2 + 1, (radius - 1) * 2 + 1)
    paddings       = tf.constant((radius - 1, radius - 1), (radius - 1, radius - 1))
    padded         = tf.pad(volume_tensors, paddings, "CONSTANT")
    for m in range(2 * (radius - 1) + 1):
        for n in range(2 * (radius - 1) + 1):
             cropped_seg[:, :, :, :, m, n].copy_(padded_seg[:,:,m:m+volume_tensors.get_shape()[2],n:n+volume_tensors.get_shape()[3]])
    mult_1 = tf.multiply(cropped_seg, weights)
    print(mult_1.shape)
    pass

if __name__ == "__main__":
    img_shape = 224
    channel = 3
    batch_size = 10
    #img = np.random.randint(low=0, high=255, size=(batch_size, img_shape , img_shape, channel))
    img = np.ones((batch_size, img_shape, img_shape, channel))
    img = np.full((batch_size, img_shape, img_shape, channel), 7)
    #img[:, 10:40, 125:192] = 255
    if False:
        cv2.imshow("img", img[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    img = img.astype(np.float32) / 256.0
    pos_sparse_matrix = positional_sparse_matrix(img_shape, img_shape, 2)
    K = 4
    #volume_tensors = tf.zeros((batch_size, img_shape, img_shape, K))
    #volume_tensors[:, :, :] = np.array([1., 0.0, 0.0, 0.0])
    #dissim = weight_matrix(img, 2)

    #print(dissim.shape)

    #ncut_loss_non_sparse_tf(volume_tensors, dissim, 5, batch_size)
    '''
    print("\nSparse Definition : \n", x)
    print("\nSparse None zero Indices : \n", np.where(coo != 0))
    print("\nSparse Indices : \n", x.indices)
    print("\nSparse Row Pointers : \n", x.indptr)
    print("\nNon zero func : \n", np.array(list(zip(x.nonzero()[0], x.nonzero()[1]))))
    print("\n Values : \n", x.data[x.data > 0])

    positions     = np.array(list(zip(pos_sparse_matrix.nonzero()[0], pos_sparse_matrix.nonzero()[1])))
    #print("Positions : \n", positions)
    pos_distances = np.reshape(pos_sparse_matrix.data[pos_sparse_matrix.data > 0], (-1, 1)).astype(np.float32) ## FIXME: FIX AT SERVER
    #pos_distances = np.square(pos_distances)
    #print("Pos distances : \n", pos_distances)
    #print("img : ", img)
    #values, sp_matrix = dissim_matrix_values(img.reshape(-1, 3), positions, pos_distances)
    #positions_new = np.array(list(zip(sp_matrix.nonzero()[0], sp_matrix.nonzero()[1])))
    volume_tensors = np.zeros((batch_size, img_shape, img_shape, K))
    volume_tensors[:, :, :] = np.array([1., 0.0, 0.0, 0.0])
    print(volume_tensors)
    '''
    #total = ncut_loss_np(volume_tensors, [img], positions, pos_distances, K, batch_size)
    #print("New Positions : \n" , positions_new)
    #print(positions.shape)
    #print(pos_distances.shape)
    positions = np.array(list(zip(pos_sparse_matrix.nonzero()[0], pos_sparse_matrix.nonzero()[1])))
    pos_distances = np.reshape(pos_sparse_matrix.data[pos_sparse_matrix.data > 0], (-1, 1)).astype(np.float32)
    #values, sp_matrix = dissim_matrix_values_tf(img.reshape(-1, 3), positions, pos_distances)

    volume_tensors = np.zeros((batch_size, img_shape, img_shape, K))
    volume_tensors[:, :, :] = np.array([0.96, 0.01, 0.01, 0.01])#np.array([0.9, 0.02, 0.05, 0.03])
    #volume_tensors[:, 10:40, 125:192] = np.array([0.01, 0.99])
    l = ncut_loss_np(volume_tensors, img, positions, pos_distances, K, batch_size)

    #input()
    if False:
        positions_tensor     = tf.placeholder(tf.float32, [positions.shape[0], positions.shape[1]])
        pos_distances_tensor = tf.placeholder(tf.float32, [pos_distances.shape[0], pos_distances.shape[1]])

        #print(pos_distances)
        vol_values     = np.random.randint(5, size=(1, img_shape, img_shape, K))
        volume_tensors = tf.placeholder(tf.float32, [None, img_shape, img_shape, K])
        image_tensors   = tf.placeholder(tf.float32, [None, img_shape, img_shape, 3])

        print("Before Before")
        #total_loss = ncut_loss_tf(volume_tensors, [img], positions, pos_distances, K, 1)
        flat_img = img.reshape(-1, 3)



        #dissim_sparse_tensor = dissim_matrix_values(flat_img, positions, pos_distances)
        #print("Before")
        total_loss = ncut_loss_tf(volume_tensors, image_tensors, positions, pos_distances_tensor, K, 1)

        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            res = sess.run([total_loss], feed_dict={volume_tensors: vol_values, image_tensors : img, positions_tensor : positions, pos_distances_tensor : pos_distances})
            print("\nTotal Loss : \n", res)

    #print(time.time() - curr)
    #N_Cut Loss
    #CRF
    #Hierarchical Clustering
    #print("Fuck Yeah")
