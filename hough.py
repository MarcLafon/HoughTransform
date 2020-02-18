import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

def show_image(img):
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    
def mask_edges(edges):
    roi = np.array([[[0, edges.shape[0]],[0, edges.shape[0]/2],
                      [edges.shape[1] / 3, edges.shape[0] / 2],[2* edges.shape[1] / 3, edges.shape[0] / 2],
                      [edges.shape[1], edges.shape[0]/2],[edges.shape[1], edges.shape[0]]]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi, (255,))
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges

# def voting_hough_space_test(edges,accuracy_rho,accuracy_theta):
#     x_max, y_max = edges.shape
#     rho_max = int(np.hypot(x_max, y_max))
#     rhos = np.arange(-rho_max,rho_max,accuracy_rho)
#     thetas = np.arange(0,np.pi,accuracy_theta)
#     thetas = np.expand_dims(thetas,0)
#     rho_dim = rhos.shape[0]
#     theta_dim = thetas.shape[1]

#     H = np.zeros((rho_dim,theta_dim))
#     xs, ys = np.where(edges==255)
#     xs = np.expand_dims(xs, 1)
#     ys = np.expand_dims(ys, 1)
    
#     xy_rhos = (xs @ np.cos(thetas))+(ys @ np.sin(thetas))
#     xy_rhos = xy_rhos.astype(np.int)
#     H[rhos,np.array(range(theta_dim))] = H[rhos,np.array(range(theta_dim))] + 1
                
#     return H

def voting_hough_space(edges,accuracy_rho,accuracy_theta):
    x_max, y_max = edges.shape
    rho_max = int(np.hypot(x_max, y_max))
    rhos = np.arange(-rho_max,rho_max,accuracy_rho)
    thetas = np.arange(0,np.pi,accuracy_theta)
    rho_dim = rhos.shape[0]
    theta_dim = thetas.shape[0]

    H = np.zeros((rho_dim,theta_dim))
    
    for (x,y),edge in np.ndenumerate(edges):
        if edge == 255:
            xy_rho = (y * np.cos(thetas))+(x * np.sin(thetas))
            xy_rho = xy_rho.astype(np.int)
            irs = ((xy_rho-rho_max)/(accuracy_rho)-1).astype(np.int)+rhos.shape[0]
            H[irs,np.array(range(theta_dim))] = H[irs,np.array(range(theta_dim))] + 1
    return H

def get_neighbors(x,y,arr,size):
    x_max = arr.shape[0]
    y_max = arr.shape[1]
    x1 = max(x-size,0)
    x2 = min(x+size,x_max)
    y1 = max(y-size,0)
    y2 = min(y+size,y_max)
    
    return arr[x1:x2,y1:y2]

def retrieve_local_maxima(H,min_votes,neighborhood_size):
    local_maxima = []
    for (x,y),v in np.ndenumerate(H):
        neighbors = get_neighbors(x,y,H,size=neighborhood_size)
        if (np.max(neighbors)==v)&(v>=min_votes):
            local_maxima.append((x,y))
    return local_maxima

def hough_transform(edges, accuracy_rho=10, accuracy_theta=np.pi/180, min_votes=150, neighborhood_size=2):
    x_max, y_max = edges.shape
    rho_max = int(np.hypot(x_max, y_max))
    rhos = np.arange(-rho_max,rho_max,accuracy_rho)
    thetas = np.arange(0,np.pi,accuracy_theta)
    
    H = voting_hough_space(edges,accuracy_rho,accuracy_theta)
    local_maxima = retrieve_local_maxima(H,min_votes, neighborhood_size)
    
    theta_dim = int(np.pi/accuracy_theta)
    thetas = accuracy_theta * np.array(range(theta_dim))
    lines = [(rhos[irho] ,thetas[itheta]) for irho,itheta in local_maxima]
    return lines


def myconvovle2d(A,kernel,fillvalue=0):
    l = kernel.shape[0]-1
    A_pad = np.pad(A, pad_width=int(l/2))
    X = np.zeros(A.shape)
    n = A_pad.shape[0]
    
    for (i,j),v in np.ndenumerate(kernel):
        i_e = i-l if i<l else n
        j_e = j-l if j<l else n
        X+=A_pad[i:i_e,j:j_e]*v
    return X

if __name__=="__main__":
    input_img_file = "Terrain_propre1.png"
    img = cv2.imread(input_img_file)

    blured_img = cv2.GaussianBlur(img,(5,5),0)
    edges = cv2.Canny(blured_img,75,150)
    plt.imshow(edges)
    plt.title("Canny Edge detection")
    plt.show()

    # masked_edges = mask_edges(edges)
    masked_edges = edges
    plt.imshow(masked_edges)
    plt.title("Canny Edge detection - Mask")
    plt.show()

    accuracy_theta = np.pi/180
    accuracy_rho = 2
    min_votes = 200
    neighborhood_size = 1

    H = voting_hough_space(masked_edges,accuracy_rho,accuracy_theta)
    plt.imshow(H)
    plt.title("Hough Space")
    plt.show()

    local_maxima = retrieve_local_maxima(H,min_votes,neighborhood_size)    
    plt.imshow(H)
    plt.title("Hough Space - Local Maxima")
    plt.plot([y for (x,y) in  local_maxima],[x for (x,y) in  local_maxima], 'ro')
    plt.show()

    lines = hough_transform(masked_edges, accuracy_rho, accuracy_theta, min_votes, neighborhood_size)
    # lines = cv2.HoughLines(masked_edges, accuracy_rho, accuracy_theta, min_votes)
    # lines = [(line[0][0], line[0][1]) for line in lines]
    for (rho, theta) in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = rho*np.cos(theta)
        y0 = rho*np.sin(theta)
        pt1 =  (int(x0 + 10000*(-b)), int(y0 + 10000*(a)));
        pt2 =  (int(x0 - 10000*(-b)), int(y0 - 10000*(a)));
        cv2.line(img, pt1, pt2, (0,255,0), 4)

    cv2.imwrite('Terrain_propre1_output.png',img)
    cv2.imshow("img",img)
    cv2.waitKey(0)
