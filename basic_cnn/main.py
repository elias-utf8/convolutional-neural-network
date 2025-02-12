import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

url_image = r'https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Tux.svg/800px-Tux.svg.png'
url_image2 = r'https://github.com/anisayari/Youtube-apprendre-le-deeplearning-avec-tensorflow/blob/master/%234%20-%20CNN/pikachu.png?raw=true'

def charger_image(url_image):
    resp = requests.get(url_image, stream=True).raw
    image_array = np.asarray(bytearray(resp.read()), dtype="uint8")
    print(f'Shape of the image {image_array.shape}')
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    return image, image_array

def noir_et_blanc(image_array):
	img_nb = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
	(thresh, img_nb) = cv2.threshold(img_nb, 127, 255, cv2.THRESH_BINARY)
	plt.axis('off')
	plt.imshow(cv2.cvtColor(img_nb, cv2.COLOR_BGR2RGB))
	return img_nb


"""
Transforme une image en une version plus petite, en niveaux de gris et binaire, et affiche à la fois l'image et les valeurs des pixels
Représente l'entrée du CNN
"""
def transformation_image(image):
	res = cv2.resize(image, dsize=(40,40), interpolation=cv2.INTER_CUBIC)
	print(res.shape)
	res = cv2.cvtColor(res,cv2.COLOR_RGB2GRAY) #TO 3D to 1D
	print(res.shape)
	res = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)[1]
	d = res
	for row in range(0,40):
	    for col in range(0,40):
	        print('%03d ' %d[row][col],end=' ')
	    print('')
	plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
	plt.axis('off')
	plt.show()


"""
Simple matrice identité 
"""
def filtre_1(image_nb): 
	kernel = np.matrix([[0,0,0],[0,1,0],[0,0,0]])
	print(kernel)
	img_1 = cv2.filter2D(image_nb, -1, kernel)
	plt.axis('off')
	plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
	plt.show()
	return img_1

"""
Applique un filtre mettant en évidence les traits verticaux
"""
def filtre_2(image_nb):
	kernel = np.matrix([[-10,0,10],[-10,0,10],[-10,0,10]])
	print(kernel)
	img_1 = cv2.filter2D(image_nb, -1, kernel)
	plt.axis('off')
	plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
	plt.show()

"""
Applique un filtre mettant en évidence les traits horizontaux
"""
def filtre_3(image_nb):
	kernel = np.matrix([[10,10,10],[0,0,0],[-10,-10,-10]])
	print(kernel)
	img_1 = cv2.filter2D(image_nb, -1, kernel)
	plt.axis('off')
	plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
	plt.show()


image, image_array = charger_image(url_image)
transformation_image(image)
image_nb = noir_et_blanc(image_array)
filtre_1(image_nb)
filtre_2(image_nb)
filtre_3(image_nb)