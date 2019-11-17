import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from tqdm import tqdm

def make_file_list(dir, type):
    r"""读取文件夹下所有文件 

    Args:
        dir: 文件夹路径  str
        type: 文件类型  str
    Return:
        namelist: 包含符合条件的所有文件的列表  list
    """
    namelist=[]
    for filename in os.listdir(dir):
        if filename.endswith(type):
            namelist.append(dir+'/'+filename)
    return np.sort(namelist).tolist()

class GMM:
    r"""混合高斯模型

    Args:
        K: 高斯模型的个数  int
        alpha: 学习率  float
        T: 阈值 float
        initImg: 初始化图片  float  (H*W*C)
        initSigma: 初始化方差值  float
        initWeight: 初始化权重  float
    """
    def __init__(self, K, alpha, T, initImg, initSigma, initWeight):
        self.K = K
        self.alpha = alpha
        self.T = T
        self.initSigma = initSigma
        self.initWeight = initWeight
        self.imgSize = initImg.shape
        self.mu, self.sigma, self.weight = self.initG(initImg)
    
    def initG(self, initImg):
        r"""初始化模型的均值权重方差

        Args:
            initImg: 用来初始化的图片  float  (H*W*C)
        Return:
            mu: 均值矩阵  float  (K*H*W*C)
            sigma: 方差矩阵  float  (K*H*W*C)
            weight: 权重矩阵  float  (K*H*W*C)
        """
        Z = np.repeat(np.expand_dims(np.zeros(self.imgSize, dtype=float), axis=0), self.K-1,axis=0)
        O = np.expand_dims(np.ones(self.imgSize, dtype=float), axis=0)
        mu = np.expand_dims(initImg,axis=0)
        mu = np.concatenate((mu,Z),axis=0)
        sigma = np.concatenate((self.initSigma*O,Z),axis=0)
        weight = np.concatenate((O,Z),axis=0)
        return mu, sigma, weight

    @staticmethod
    def read_image(imgname):
        r"""读取图片，采用opencv方法，返回float类型，防止溢出
        Args: 
            imgname: 图片路径  str
        Return: 
            img: 图片  float  (H*W*C)
        """
        # imgname = '../WavingTrees/b'+str(i).zfill(5)+'.bmp'
        # img = plt.imread(imgname)
        img = cv2.imread(imgname)
        # print(img)
        return img.astype(float)

    @staticmethod
    def disp_imgs(*imgs, nx=1, ny=1, size=(160,120), scale=1, name='FIGURE', show_img=False):
        r"""显示多幅图像的算法，不足位置填充为空白

        Args:
            *imgs: img1, img2, ..., imgn  一组图像
            nx: 纵向显示的图片数 int
            ny: 横向显示图片数 int
            size: 图片大小 tuple
            scale: 尺度变换 float/int/double
            name: 窗口名称 str
        Return:
            imgbox: 打包好的图像集合 float 
        """
        n_img = len(imgs)
        iter_img = 0
        scaled_size = (np.ceil(size[0]*scale).astype(np.int),np.ceil(size[1]*scale).astype(np.int))

        for i in range(nx):
            for j in range(ny):
                if iter_img>=n_img:
                    add_img = cv2.resize((imgs[0]*0+255).astype(np.uint8), scaled_size) 
                else:
                    add_img = cv2.resize(imgs[iter_img].astype(np.uint8), scaled_size)
                if j == 0:
                    yimgs = add_img
                else:
                    yimgs = np.hstack([yimgs, add_img])
                iter_img += 1
            if i == 0:
                imgbox = yimgs
            else:
                imgbox = np.vstack([imgbox, yimgs])
        if show_img:
            cv2.namedWindow(name)
            cv2.imshow(name,imgbox)
            cv2.moveWindow(name,200,50)
            cv2.waitKey(0)  
            cv2.destroyAllWindows()
        return imgbox


    def matcher(self, img, match_first=True, mask=None):
        r"""判断是否匹配到某一分布

        Args:
            img: 图片  float  (H*W*C)
            match_first: 是否只保留第一个匹配的位置  boolean
            mask: 用于测试，mask是阈值遮罩矩阵  int  (K*H*W*C)
        Retuen:
            match: 匹配矩阵  int  (K*H*W*C)
            unmatched:  不匹配的位置  int  (H*W*C)
        """
        match = np.sum((np.square(self.mu-img)<(self.sigma*6.25)), axis=-1).astype(float)
        match[match<2.5] = 0.0
        match[match>2.5] = 1.0
        match = np.repeat(np.expand_dims(match,axis=-1), 3, axis=-1)

        if match_first:
            tmp = np.zeros(match[0].shape)
            for i in range(0, match.shape[0]):
                ntmp = tmp + match[i]
                match[i] -= tmp
                tmp = ntmp
            match[match>0.1] = 1.0
            match[match<=0.1] = 0.0

        if mask is not None:
            unmatched = 1.0 - np.max(match*mask, axis=0)
        else:
            unmatched = 1.0 - np.max(match, axis=0)

        return match, unmatched
    
    def updater(self, img, match, unmatched):
        r"""判断是否匹配到某一分布

        Args:
            img: 图片  float  (H*W*C)
            match: 匹配矩阵  int  (K*H*W*C)
            unmatched:  不匹配的位置  int  (H*W*C)
        """

        rho = self.alpha*(1/(np.sqrt(2*3.14*self.sigma+1e-10)) * np.exp(-0.5*np.square(self.mu-img)/(self.sigma+1e-10)))
        # rho[rho>1.0] = 1.0

        self.mu = (1.0-match)*self.mu + match*((1.0-rho)*self.mu+rho*img)
        self.sigma = (1.0-match)*self.sigma + match*((1.0-rho)*self.sigma+rho*np.square(self.mu-img))
        self.weight = (1-self.alpha)*self.weight + self.alpha*match

        index = np.argsort(-(self.weight/(np.expand_dims(np.mean(self.sigma),axis=-1)+1e-10)),axis=0) #/(np.expand_dims(np.mean(sigma),axis=-1)+1e-10)
        self.weight = np.take_along_axis(self.weight,index,axis=0)
        self.mu = np.take_along_axis(self.mu,index,axis=0)
        self.sigma = np.take_along_axis(self.sigma,index,axis=0)


        self.mu[-1] = img*unmatched + (1-unmatched)*self.mu[-1]
        self.sigma[-1] = np.expand_dims(np.ones(self.imgSize, dtype=float), axis=0)*self.initSigma*unmatched +(1-unmatched)*self.sigma[-1]
        self.weight[-1] = np.expand_dims(np.ones(self.imgSize, dtype=float), axis=0)*self.initWeight*unmatched +(1-unmatched)*self.weight[-1]

        self.weight = self.weight/(np.sum(self.weight,axis=0)+1e-10)

    def make_B_mask(self):
        r"""制作阈值选择遮罩

        Return:
            mask: 遮罩  int  (K*H*W*C)
        """
        mask = np.zeros(self.weight.shape)
        mask[0] = self.weight[0]
        for i in range(1, mask.shape[0]):
            mask[i] = mask[i-1]+self.weight[i]
        for i in range(mask.shape[0]-2,-1,-1):
            mask[i+1] = mask[i]
        #for i in range(0, mask.shape[0]):
        mask[mask>self.T] = 1
        mask[mask<=self.T] = 0
        mask = 1 - mask
        mask[0] = np.ones(mask[0].shape)
        return mask

    def train(self, imgnamelist, match_first=True, show_img=False):
        r""" 训练模型

        Args:
            imgnamelist: 图片名称列表  list contains str
            match_first: 是否只保留第一个匹配 boolean
        """
        
        for imgname in tqdm(imgnamelist):
            img = self.read_image(imgname)
            match, unmatched = self.matcher(img, match_first=match_first)
            self.updater(img, match, unmatched)
            self.disp_imgs(img, unmatched*255, ny=2, nx=1, size=(img.shape[1],img.shape[0]), scale=2, show_img=show_img)
        
    def test(self, imgnamelist, save_method='video', save_name='result', show_img=False):
        r""" 训练模型

        Args:
            imgnamelist: 图片名称列表  list contains str
            save_method: 保存方法 in 'video', 'image' 
            save_name: 结果名称
        """

        if save_method == 'video':
            fps = 8
            size = (self.imgSize[1]*4,self.imgSize[0]*4)
            fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
            outv = cv2.VideoWriter()
            outv.open('./'+save_name+'.mp4', fourcc, fps, size)

        mask = self.make_B_mask()

        kernel_ero = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  #MORPH_ELLIPSE
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  #MORPH_RECT
        iimg = 0
        for imgname in tqdm(imgnamelist):
            img = self.read_image(imgname)
            match, unmatched = self.matcher(img, match_first=False, mask=mask)
            eroded= cv2.erode(unmatched, kernel_ero) 
            dilated = cv2.dilate(eroded, kernel_dil)
            padded = dilated*img
            ibx = self.disp_imgs(img, unmatched*255, dilated*255, padded, ny=2, nx=2, size=(img.shape[1],img.shape[0]), scale=2, show_img=show_img)
            iimg += 1
            if save_method == 'video':
                outv.write(ibx)
            if save_method == 'image':
                res_pos = './'+save_name+'.png'
                cv2.imwrite(res_pos, ibx)
        
        if save_method == 'video':
            outv.release()
        
        



if __name__ == "__main__":
    # 实际运行时要修改文件夹的路径名称
    # 如果要显示中间的图片，则修改show_img的布尔值

    imgdir = r'../WavingTrees/'
    show_img = False
    imgnamelist = make_file_list(imgdir, 'bmp')

    trainlist = imgnamelist[:200]
    testlist  = imgnamelist[200:286]

    img1 = GMM.read_image(imgnamelist[0])
    model = GMM(7, 0.01, 0.9, img1, 100, 0.005)
    print("Training...")
    model.train(trainlist, show_img=show_img)
    print("Testing...")
    model.test(testlist, save_method='video', save_name='resultK'+str(model.K), show_img=show_img)
    model.test([imgnamelist[248]],save_method='image', save_name='resultK'+str(model.K), show_img=show_img)
    print("Finish!")
