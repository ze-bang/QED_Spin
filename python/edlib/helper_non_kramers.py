import numpy as np
import sys
import os
import matplotlib.pyplot as plt

omega = np.exp(2*1j*np.pi/3) #gamma phase factor

#vectors pointing from the center of an eta=+1 (upper,down-pointing) tetrahedron to the 4 sublattices
b0 = -1.0/8*np.array([1,1,1])
b1 = -1.0/8*np.array([1,-1,-1])
b2 = -1.0/8*np.array([-1,1,-1])
b3 = -1.0/8*np.array([-1,-1,1])
bb = np.array([b0, b1, b2, b3]) #list of b_i vectors to index into

#displacement of the 0,1,2,3 sublattices from the 0 sublattice
#note: the displacement between unit cells is twice ei
e0 = np.array([0,0,0])
e1 = 1.0/4*np.array([0,1,1])
e2 = 1.0/4*np.array([1,0,1])
e3 = 1.0/4*np.array([1,1,0])
ee = np.array([e0,e1,e2,e3]) #list of e_i vectors to index into

ee_coord = np.vstack(([0,0,0], np.eye(3)))

#local z axes
z = np.array([np.array([1,1,1])/np.sqrt(3), np.array([1,-1,-1])/np.sqrt(3), np.array([-1,1,-1])/np.sqrt(3), np.array([-1,-1,1])/np.sqrt(3)])

def clusters(name):
    xhat = ee_coord[1,:]
    yhat = ee_coord[2,:]
    zhat = ee_coord[3,:]
    
    if name == '1':
        coords = np.array([e0])
        faces = ['up']
    elif name == '2':
        coords = np.array([e0, e0])
        faces = ['up', 'down']
    elif name == '3':
        coords = np.array([e0, e0, xhat]) 
        faces = ['up', 'down', 'up']
    elif name == '4a':
        coords = np.array([e0, e0, xhat, yhat]) 
        faces = ['up', 'down', 'up', 'up']
    elif name == '4b':
        coords = np.array([e0, e0, xhat, -zhat]) 
        faces = ['up', 'down', 'up', 'down']
    elif name == '5a':
        coords = np.array([e0, e0, xhat, yhat, zhat]) 
        faces = ['up', 'down', 'up', 'up', 'up']
    elif name == '5b':
        coords = np.array([e0, e0, xhat, yhat, -zhat]) 
        faces = ['up', 'down', 'up', 'up', 'down']
    elif name == '5c':
        coords = np.array([e0, e0, xhat, -zhat, -zhat]) 
        faces = ['up', 'down', 'up', 'down', 'up']
    
    return coords, faces

def draw_tet(coords, faces, show_guides, show_label): #coord in units of e1, e2, e3
    N = len(faces) #number of tetrahedra
    vertices = np.zeros((4*N, 3))
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    for j in range(N):
        if faces[j] == 'up':
            eta = -1
        else:
            eta = 1
        coord = coords[j, :]
        tet_center = -eta*b0 + coord[0]*2*e1 + coord[1]*2*e2 + coord[2]*2*e3
        vertices[4*j:4*(j+1), :] = tet_center[None,:] + eta*bb
        
        xs = vertices[4*j:4*(j+1), 0]
        ys = vertices[4*j:4*(j+1), 1]
        zs = vertices[4*j:4*(j+1), 2]
        
        for ii in range(3):
            for jj in range(ii+1,4):
                ax.plot([xs[ii],xs[jj]],[ys[ii],ys[jj]],[zs[ii],zs[jj]], color='red')
            
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], marker='o', color='blue')
    
    if show_guides:
        #ax.scatter(b0[0],b0[1],b0[2], marker='o', color='green') #indicates tetrahedron 0_up
        #ax.plot([0,b0[0]], [0,b0[1]], [0,b0[2]], color= 'green') #indicates "up" facing tetrahedron
        ax.quiver(b0[0],b0[1],b0[2],-b0[0]/(3/2),-b0[1]/(3/2),-b0[2]/(3/2), color='green') #indicates tetrahedron 0_up, and the "up" direction
        
        #ax.plot([0,2*e1[0]], [0,2*e1[1]], [0,2*e1[2]], color= 'green') 
        #ax.plot([0,2*e2[0]], [0,2*e2[1]], [0,2*e2[2]], color= 'green') 
        #ax.plot([0,2*e3[0]], [0,2*e3[1]], [0,2*e3[2]], color= 'green') 
    
    if show_label:
        sipc = to_SIPC(coords, faces)
        N_pts = np.shape(sipc)[0] #number of vertices
        
        for j in range(N_pts):
            mu = int(sipc[j,3])
            r_mu = sipc[j,0]*2*e1 + sipc[j,1]*2*e2 + sipc[j,2]*2*e3 + ee[mu]
            text_position = r_mu + np.array([1,1,1])*1/50
            ax.text(*text_position, str(j), zdir=(1,1,0))
        
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_box_aspect((np.ptp(vertices[:,0]),np.ptp(vertices[:,1]),np.ptp(vertices[:,2])))
            
def to_SIPC(coords, faces):
    N = len(faces) #number of tetrahedra
    vertices = np.zeros((4*N, 4))    

    for j in range(N):
        up_factor = 0 #whether tetrahedron is pointing "down" (0) or "up" (-1)
        if faces[j] == 'up':
            up_factor = -1
        
        coord_0 = coords[j, :] #position of the 0 sublattice site in the given tetrahedron
        
        vertices[4*j:4*(j+1), :3] = coord_0[None,:] + up_factor*ee_coord
        vertices[4*j:4*(j+1), 3] = np.arange(0,4,1) #the sublattice index
        
    unique_vertices, idx = np.unique(vertices, axis=0, return_index=True) #removes duplicate coordinates
    
    return unique_vertices[np.argsort(idx)]

def mask(coords, faces):
    def make_mask(index_1, index_2, points): #puts ones if two sites are neighbours
        m = np.zeros(points)
        m[index_1] = 1
        m[index_2] = 1
        
        return m

    sipc = to_SIPC(coords, faces) 
    N = np.shape(sipc)[0] #number of sites (vertices)
    mask_list = np.zeros(N+1) #+1 for the gamma indicator column
    
    for p in range(N):
        coord_0 = sipc[p,:]
        sublattice = int(coord_0[3])
    
        #possible intra-tetrahedron neighbours
        possible_intra = np.outer(np.array([1,1,1,1]), coord_0) 
        possible_intra[:,3] = np.arange(0,4,1)
            
        #possible inter-tetrahedron neighbours
        e_sub = ee_coord[sublattice,:]
        possible_inter = coord_0[None,:3] + e_sub[None,:] - np.vstack((np.zeros(3), np.eye(3)))
        possible_inter = np.concatenate((possible_inter, np.array([[0,1,2,3]]).T), axis = 1)
    
        possible_neighbours = np.vstack((possible_intra, possible_inter))
    
        #checks if each possible neighbour is in the SIPC list
        for n in range(8):
            for k in range(N):
                if possible_neighbours[n,:].tolist() == sipc[k,:].tolist() and k != p:
                    gamma_indicator = -1
                    sublattice_neighbour = possible_neighbours[n,3]
                    
                    #sublattice indices of original point and neighour
                    mu_nu = set([sublattice, sublattice_neighbour])
                    
                    if mu_nu == set([0,1]) or mu_nu == set([2,3]):
                        gamma_indicator = 1
                    if mu_nu == set([0,2]) or mu_nu == set([1,3]):
                        gamma_indicator = 2
                    if mu_nu == set([0,3]) or mu_nu == set([1,2]):
                        gamma_indicator = 3
                    
                    mask = make_mask(p, k, N)
                    mask = np.append(mask, gamma_indicator)
                    
                    mask_list = np.vstack((mask_list, mask))
        
    mask_list = mask_list[1:, :] #removes first row of zeros
    mask_list = np.unique(mask_list, axis=0) #removes duplicates

    return mask_list

def inter_table(cluster_mask, Js):
    J_zz, J_pm, J_pmpm = Js 
    
    J_zz = J_zz/2
    J_pm = J_pm/2
    J_pmpm = J_pmpm/2

    edges = np.shape(cluster_mask)[0]
    
    #5 terms for each nearest neighbour pair: Sz Sz, S+ S-, S- S+, S+ S+, S- S-
    inter = np.zeros((5*edges, 6), dtype=float)
    
    for row in range(edges):
        sites = np.nonzero(cluster_mask[row,:-1])[0]
        gamma_index = cluster_mask[row,-1] - 1
        
        inter[5*row:5*(row+1),0] = [2,0,1,0,1]
        inter[5*row:5*(row+1),1] = sites[0]*np.ones(5)
        inter[5*row:5*(row+1),2] = [2,1,0,0,1]
        inter[5*row:5*(row+1),3] = sites[1]*np.ones(5)
        
        inter[5*row:5*(row+1),4] = [J_zz, -J_pm, -J_pm, np.real(J_pmpm*omega**gamma_index), np.real(J_pmpm*omega**gamma_index)]
        inter[5*row:5*(row+1),5] = [0, 0, 0, np.imag(J_pmpm*omega**gamma_index), np.imag(J_pmpm*omega**gamma_index)]
    
    return inter

def zeeman_table(B, sipc):
    N_sites = np.shape(sipc)[0]
    transfer = np.zeros((N_sites, 4),dtype=float)
    
    transfer[:,0] = 2*np.ones(N_sites) #Sz
    transfer[:,1] = np.arange(0,N_sites,1) #site index
    for j in range(N_sites):
        transfer[j,2] = B[sipc[j,-1]] #component of h along local z axis
    
    return transfer


cluster_name = str(sys.argv[9])
tet_center_coords, tet_faces = clusters(cluster_name)

#examples
#tet_center_coords = np.array([[0,0,0],[0,0,0]])
#tet_faces = ['up', 'down']
#tet_center_coords = np.array([[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
#tet_faces = ['up', 'down', 'up', 'up', 'up']

#make a plot for fun
#draw_tet(tet_center_coords, tet_faces, show_guides = False, show_label = True)

#exchange parameters
Jxx, Jyy, Jzz = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])

Jpm = -(Jxx+Jyy)/4
Jpmpm = (Jxx-Jyy)/4

#sublattice indexed pyrochlore coordinates of all the sites
cluster_sipc = to_SIPC(tet_center_coords, tet_faces).astype(int)
#mask
cluster_mask = mask(tet_center_coords, tet_faces).astype(int)

h = float(sys.argv[4])
fielddir = np.array([float(sys.argv[5]),float(sys.argv[6]), float(sys.argv[7])])
fielddir = fielddir/np.linalg.norm(fielddir)
B = np.einsum('r, ir->i', h*fielddir, z)

interALL = inter_table(cluster_mask, [Jzz, Jpm, Jpmpm])
transfer = zeeman_table(B, cluster_sipc)

output_dir = "./" + sys.argv[8] + "/"

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

max_site = np.shape(cluster_sipc)[0]
All_N = max_site
exct = 1

fstrength = np.zeros((1,1))
fstrength[0,0] = h
np.savetxt(output_dir+"field_strength.dat", fstrength)

def write_interALL(interALL, file_name):
    num_param = len(interALL)
    f        = open(output_dir+file_name, 'wt')
    f.write("==================="+"\n")
    f.write("num "+"{0:8d}".format(num_param)+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    for i in range(num_param):
        f.write(" {0:8d} ".format(int(interALL[i,0])) \
        +" {0:8d}   ".format(int(interALL[i,1]))     \
        +" {0:8d}   ".format(int(interALL[i,2]))     \
        +" {0:8d}   ".format(int(interALL[i,3]))     \
        +" {0:8f}   ".format(interALL[i,4]) \
        +" {0:8f}   ".format(interALL[i,5]) \
        +"\n")
    f.close()

def write_transfer(interALL, file_name):
    num_param = len(interALL)
    f        = open(output_dir+file_name, 'wt')
    f.write("==================="+"\n")
    f.write("num "+"{0:8d}".format(num_param)+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    f.write("==================="+"\n")
    for i in range(num_param):
        f.write(" {0:8d} ".format(int(interALL[i,0])) \
        +" {0:8d}   ".format(int(interALL[i,1]))     \
        +" {0:8f}   ".format(interALL[i,2])     \
        +" {0:8f}   ".format(interALL[i,3])
        +"\n")
    f.close()

write_interALL(interALL, 'InterAll.dat')
write_transfer(transfer, 'Trans.dat')
