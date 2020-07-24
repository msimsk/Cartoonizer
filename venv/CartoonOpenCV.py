import numpy as np
import cv2
import os


# class Cartoonizer:
def Cartoon(im, select):
    output = np.array(im)
    w, h = output.shape[:2]
    # 1. adım
    # Gaussian Piramid boyutu
    numDown = 2
    imColor = im
    imRgb = imColor
    # bilateral filtresi daha hızlı sonuç verdiğinden kamerada kullanılacaktır
    if select == True:
        print("bilateral çalışıyor")
        numBilateral = 50
        for i in range(numDown):
            imColor = cv2.pyrDown(imColor)
        for i in range(numBilateral):
            imColor = cv2.bilateralFilter(imColor, d=9, sigmaColor=9, sigmaSpace=7)
            # print("-----------------------------")
        for i in range(numDown):
            imColor = cv2.pyrUp(imColor)
    # K-Means algoritması ile renk azaltma(yavaş)
    else:
        print("K-Means çalışıyor")
        for i in range(numDown):
            imColor = cv2.pyrDown(imColor)
        imColor = KMeans(imColor, h, w)
        for i in range(numDown):
            imColor = cv2.pyrUp(imColor)
    # 2. adım
    imGray = cv2.cvtColor(imRgb, cv2.COLOR_RGB2GRAY)
    imBlur = cv2.medianBlur(imGray, 3)
    # 3. adım
    imEdge = cv2.adaptiveThreshold(imBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
    # 4. adım
    imEdge = cv2.resize(imEdge, (h, w))
    imEdge = cv2.cvtColor(imEdge, cv2.COLOR_GRAY2RGB)
    imColor = cv2.resize(imColor, (h, w))
    imCartoon = cv2.bitwise_and(imColor, imEdge)
    return imCartoon


def KMeans(img, w, h):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # görüntü vektörel ve float 32 düründe olmalıdır
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    #  sonlandırma kriterleri burada belirlenir
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # küme sayımız
    K = 16
    #
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    res2 = cv2.cvtColor(res2, cv2.COLOR_LAB2BGR)
    res2 = cv2.resize(res2, (h,w))
    return res2

def cartoonCam():
    cap = cv2.VideoCapture("")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output1.avi', fourcc, 8, (640, 480))
    # Kamera bağlantısını kontrol et.
    if not cap.isOpened():
        print("Kamera bağlantısı başarısız!")
        exit()
    # while True:
    while True:
        # Kameradan görüntü al
        ret, frame = cap.read()
        # Görüntü başarıyla alındı mı kontrol et.
        if not ret:
            print("Bağlantıdan görüntü alınamadı!")
            break
        # Gri formatta okumak için
        # frame = KMeans(frame)
        imCartoon = Cartoon(frame, "True")
        # pencere boyutu
        height = 1280
        width = 720
        cv2.namedWindow("Cartoon", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cartoon", height, width)
        imCartoon = cv2.flip(imCartoon, 1)
        out.write(imCartoon)
        # Okunan görüntüyü ekranda göster.
        cv2.imshow('Cartoon', imCartoon)
        # q tuşuna basıldığında çık.
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    # img = cv2.imread("images/ben.jpg")
    # (h, w, r) = img.shape
    # if h / 2 < 1600:
    #     h -= int((h / 3))
    #
    # if w / 2 < 1200:
    #     w -= int((w / 3))
    # kmeans, cartoon ve orjinal resim
    # kmean = KMeans(img, h, w)
    # cartoon, cartoon1 = Cartoon(img, h, w)


    in_dir = './images/input'
    out_dir = './images/outputbill'

    # os.mkdir(out_dir)

    for f in os.listdir(in_dir):
        image = cv2.imread(os.path.join(in_dir, f))
        print('==============')
        print(f)
        # start_time = time.time()
        # output = cartoonize(image)
        # Cartoon = Cartoonizer()
        output = Cartoon(image, True)
        # end_time = time.time()
        # print("time: {0}s".format(end_time - start_time))
        name = os.path.basename(f)
        tmp = os.path.splitext(name)
        name = tmp[0] + "_cartoon" + tmp[1]
        name = os.path.join(out_dir, name)
        print("write to {0}".format(name))
        cv2.imwrite(name, output)
    # cartoonCam()