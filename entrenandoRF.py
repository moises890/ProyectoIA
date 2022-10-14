import cv2
import os
import numpy as np
import time

def getmodel(metodo, facedata,label):
	if metodo == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
	if metodo == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
	if metodo == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

	print("Entrenando ( "+metodo+" )...")
	inicio = time.time()
	emotion_recognizer.train(facesData, np.array(labels))
	tiempoEntrenamiento = time.time()-inicio
	print("Tiempo de entrenamiento ( "+metodo+" ): ", tiempoEntrenamiento)

	# Almacenando el modelo obtenido
	emotion_recognizer.write("modelo"+metodo+".xml")


dataPath = 'Emociones encontradas' #Cambia a la ruta donde hayas almacenado Data
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print('Leyendo las imágenes')

	for fileName in os.listdir(personPath):
		print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))
		#image = cv2.imread(personPath+'/'+fileName,0)
		#cv2.imshow('image',image)
		#cv2.waitKey(10)
	label = label + 1

getmodel("LBPH",facesData,labels)