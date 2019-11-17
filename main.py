import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def ReadImage(imgname):
    r"""读取图片，采用opencv方法，返回float类型，防止溢出
    Args: 
        imgname:图片路径
    output: 
        float 类型的图片
    """
    # imgname = '../WavingTrees/b'+str(i).zfill(5)+'.bmp'
    # img = plt.imread(imgname)
    img = cv2.imread(imgname)
    # print(img)
    return img.astype(float)

    

def init(img, K):
    r"""初始化，对均值方差权重按照赋初值
    Args:
        img: 一个float类型的图
        K: 高斯分布的个数
    Output：
        mu: 高斯分布的均值
    """
    Z = np.zeros((K-1,img.shape[0],img.shape[1],img.shape[2]), dtype=float)
    O = np.ones((1,img.shape[0],img.shape[1],img.shape[2]), dtype=float)
    mu = np.expand_dims(img,axis=0)
    mu = np.concatenate((mu,Z),axis=0)
    sigma = np.concatenate((100.0*O,Z),axis=0)
    weight = np.concatenate((O,Z),axis=0)
    # weight = np.mean(weight,axis=-1)
    # match = 0.0*weight
    return mu, sigma, weight


def train(imglist, K, alpha=0.05):
    img = ReadImage('/Users/dgc/Study/Lessons/视频分析前沿/Project1/WavingTrees/b00000.bmp')
    mu, sigma, weight = init(img, K)
    for i_img in imglist:
        imgname = '/Users/dgc/Study/Lessons/视频分析前沿/Project1/WavingTrees/b'+str(i_img).zfill(5)+'.bmp'
        # cmpname = '/Users/dgc/Study/Lessons/视频分析前沿/Project1/WavingTrees/b'+str(i_img+1).zfill(5)+'.bmp'
        #print(imgname)
        img = ReadImage(imgname)
        # print(len(img.shape))
        # cmpimg = ReadImage(cmpname)
        #match = (np.sum(np.square(mu-img),axis=-1)<np.sum(sigma*6.25,axis=-1)).astype(float)
        #match = (np.square(mu-img)<sigma*6.25)

        match = np.sum((np.square(mu-img)<(sigma*6.25)),axis=-1).astype(float)
        match[match<2.5] = 0.0
        match[match>2.5] = 1.0
        match = np.repeat(np.expand_dims(match,axis=-1),3,axis=-1)

        tmp = np.zeros(match[0].shape)

        for i in range(0, match.shape[0]):
            ntmp = tmp + match[i]
            match[i] -= tmp
            tmp = ntmp

        match[match>0.1] = 1.0
        match[match<=0.1] = 0.0

        # print(match[:,10,50,0])

        rho = alpha*(1/(np.sqrt(2*3.14*sigma+1e-10)) * np.exp(-0.5*np.square(mu-img)/(sigma+1e-10)))
        # rho[rho>1.0] = 1.0
        mu = (1.0-match)*mu + match*((1.0-rho)*mu+rho*img)
        sigma = (1.0-match)*sigma + match*((1.0-rho)*sigma+rho*np.square(mu-img))

        unmatched = 1.0-np.max(match, axis=0)

        # unmatched = (1-match[0])
        # for i in range(K):
        #    unmatched *= (1-match[i])

        weight = (1-alpha)*weight + alpha*match
        #weight = np.mean(weight,axis=-1)
        #print(weight)
        # print("img:\t", str(img[5,5,:]))
        # print("sigma:\n", str(sigma[0:K,5,5,:]))
        # print("cerr:\n",str(np.square(mu-img)[0:K,5,5,:]))
        # print("match:\n",str(match[0:K,5,5,:]))
        # #print("err:\t", str(np.square(img[5,5,:]-mu[0,5,5,:])))
        # print("weight:\n", str(weight[0:K,5,5,:]))
        # print("img:\t", str(i_img))

        index = np.argsort(-(weight/(np.expand_dims(np.mean(sigma),axis=-1)+1e-10)),axis=0) #/(np.expand_dims(np.mean(sigma),axis=-1)+1e-10)
        weight = np.take_along_axis(weight,index,axis=0)
        mu = np.take_along_axis(mu,index,axis=0)
        sigma = np.take_along_axis(sigma,index,axis=0)


        mu[K-1] = img*unmatched + (1-unmatched)*mu[K-1]
        sigma[K-1] = np.ones((1,img.shape[0],img.shape[1],img.shape[2]), dtype=float)*100.0*unmatched +(1-unmatched)*sigma[K-1]
        weight[K-1] = np.ones((1,img.shape[0],img.shape[1],img.shape[2]),dtype=float)*0.005*unmatched +(1-unmatched)*weight[K-1]

        weight = weight/(np.sum(weight,axis=0)+1e-10)

        cv2.namedWindow('FIGURE')
        #cv2.resizeWindow('test',1000, 1000)
        imgs = np.hstack([cv2.resize(img.astype(np.uint8),(img.shape[1]*3,img.shape[0]*3)), cv2.resize((unmatched*255).astype(np.uint8),(img.shape[1]*3,img.shape[0]*3))])
        cv2.imshow('FIGURE',imgs)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()

        # plt.figure(i_img,figsize=(10,4))
        # plt.subplot(1,2,1)
        # plt.imshow(img.astype(int))
        # plt.subplot(1,2,2)
        # plt.imshow(unmatched)
        # plt.show()
        # plt.pause(0.05)
        # plt.close()

    return mu, sigma, weight

        

def test(imglist, mu, sigma, weight, T, close=True):

    mask = np.zeros(weight.shape)
    mask[0] = weight[0]
    for i in range(1, mask.shape[0]):
        mask[i] = mask[i-1]+weight[i]
    for i in range(mask.shape[0]-2,-1,-1):
        mask[i+1] = mask[i]
    #for i in range(0, mask.shape[0]):
    mask[mask>T] = 1
    mask[mask<=T] = 0
    mask = 1 - mask
    mask[0] = np.ones(mask[0].shape)

    # print(mask[:,100,50,0])
    # print(weight[:,100,50,0])
    # print(np.sum(weight[:,100,50,0]))
    #mask[0:least] = np.ones(mask[0:least].shape)
        #mask[i] = (1-mask[i-1])*mask[i]
    kernel_ero = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  #MORPH_ELLIPSE
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  #MORPH_RECT

    imgname = '/Users/dgc/Study/Lessons/视频分析前沿/Project1/WavingTrees/b00000.bmp'
    img = ReadImage(imgname)

    fps = 8
    size = (img.shape[1]*6,img.shape[0]*6)
    fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
    outv = cv2.VideoWriter()
    outv.open('./result.mp4', fourcc, fps, size, True)

    for i_img in imglist:
        imgname = '/Users/dgc/Study/Lessons/视频分析前沿/Project1/WavingTrees/b'+str(i_img).zfill(5)+'.bmp'
        img = ReadImage(imgname)

        match = np.sum((np.square(mu-img)<(sigma*6.25)),axis=-1).astype(float)
        match[match<2.5] = 0.0
        match[match>2.5] = 1.0
        match = np.repeat(np.expand_dims(match,axis=-1),3,axis=-1)
        unmatched = 1 - np.max(match*mask, axis=0)

        # unmatched = (1-match[0])
        # for i in range(1, weight.shape[0]):
        #     unmatched *= (1-match[i]*mask[i])

        eroded= cv2.erode(unmatched, kernel_ero) 
        dilated = cv2.dilate(eroded, kernel_dil)

        #opening = cv2.morphologyEx(unmatched, cv2.MORPH_OPEN, kernel)
        #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        cv2.namedWindow('FIGURE')
        #cv2.resizeWindow('test',1000, 1000)
        imgs1 = np.hstack([cv2.resize(img.astype(np.uint8),(img.shape[1]*3,img.shape[0]*3)), cv2.resize((unmatched*255).astype(np.uint8),(img.shape[1]*3,img.shape[0]*3))])

        imgs2 = np.hstack([cv2.resize((dilated*255).astype(np.uint8),(img.shape[1]*3,img.shape[0]*3)), cv2.resize((dilated*img).astype(np.uint8),(img.shape[1]*3,img.shape[0]*3))])
        imgs = np.vstack([imgs1,imgs2])

        cv2.imshow('FIGURE',imgs)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()

        outv.write(imgs)

        # plt.figure(i_img,figsize=(10,7))
        # plt.subplot(2,2,1)
        # plt.imshow(img.astype(int))
        # plt.subplot(2,2,2)
        # plt.imshow(unmatched)
        # plt.subplot(2,2,3)
        # plt.imshow(dilated)
        # plt.subplot(2,2,4)
        # plt.imshow((dilated/255.0)*img.astype(int))
        # if close == False:
        #     plt.ioff()
        #     plt.show()
        # if close == True:
        #     plt.show()
        #     plt.pause(0.05)
        #     plt.close()
    outv.release()

if __name__ == "__main__":
    trainlist = range(0,200)
    testlist = range(200,286)
    T = 0.9
    plt.ion()
    mu,sigma,weight = train(trainlist, K=100, alpha=0.01)
    test(testlist, mu, sigma, weight, T)
    #test([247], mu, sigma, weight, T, False)