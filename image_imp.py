

import numpy as np

class Jpeg_img:
    
    def __init__(self,img,comp_per):
        ''' 
        Stores the image matrix in sparse matrix form after compresion
        Input: img: image matrix
        comp_per: compression ratio in percentage'''
        
        
        
        img = np.array(img)
        img1,self.shape = self.truncate(img)
        
        self.comp_per = comp_per
        
        self.c_img = self.img_compress(img1,comp_per)

    def truncate(self,img):
        
        m,n = img.shape
        m_2,n_2 = 2**int(np.log2(m)),2**int(np.log2(n))
        return img[:m_2,:n_2],(m_2,n_2)
        
        
    def img_compress(self,A, comp_per):
        ''' A: m x n matrix where m and n are powers of 2
            comp_per: percentage, ranges b/w [0 , 100)
            Returns: Compressed and transformed matrix of A represented in sparse matrix format,shape of sparse matrix

            Raises:
                Value error'''

        if comp_per >= 100 or comp_per < 0:
            raise ValueError("n has to be between [0 , 100)")

        
        y_A = np.fft.fft2(A)
        
        y_li = []
        m,n = y_A.shape
        
        
        
        for r in range(m):
            for c in range(n):
                y_li.append((r,c,y_A[r][c]))
        
        y_li.sort(key = lambda x: x[2]*x[2].conjugate())
        
        y_li = y_li[::-1]
        y_li = y_li[:int(len(y_li)*(100-comp_per)/100)]
        return y_li

    def render(self):
        
        y_li = self.c_img
        m,n = self.shape
        y_mat = [[0 for i in range(n)] for j in range(m)]

        for val in y_li:
            y_mat[val[0]][val[1]] = val[2]

        y_mat = np.array(y_mat)
        print("Starting conversion")
        A = np.fft.ifft2(y_mat).real
        
        print("Conversion Done")

        return A
    